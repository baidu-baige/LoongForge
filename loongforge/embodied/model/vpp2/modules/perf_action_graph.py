# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in native action-layer forward/backward graphs; ZeRO and optimizer stay eager."""
import atexit
import json
import logging
import os
import time
from pathlib import Path

import torch

from .wan_video_dit import flash_attention, modulate, rope_apply

logger = logging.getLogger(__name__)
_health_records = []


class _ActionLayer(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.train(block.training)

    def forward(self, x, freqs, t_mod, context, context_mask, kv_mask, video_k, video_v):
        from .mot import MoT

        block = self.block
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = MoT._split_modulation(block, t_mod)
        z = modulate(block.norm1(x), shift, scale)
        q = rope_apply(block.self_attn.norm_q(block.self_attn.q(z)), freqs, block.num_heads)
        k = rope_apply(block.self_attn.norm_k(block.self_attn.k(z)), freqs, block.num_heads)
        v = block.self_attn.v(z)
        mixed = flash_attention(
            q, torch.cat([video_k, k], dim=1), torch.cat([video_v, v], dim=1),
            block.num_heads, ctx_mask=kv_mask,
        )
        return MoT._apply_expert_post_block(
            block, x, mixed, gate, shift_mlp, scale_mlp, gate_mlp,
            {"context": context, "mask": context_mask},
        )


class _ActionStack(torch.nn.Module):
    def __init__(self, blocks, mot=None):
        super().__init__()
        self.layers = torch.nn.ModuleList([_ActionLayer(block) for block in blocks])
        self.__dict__["_mot"] = mot

    def forward(self, x, freqs, t_mod, context, context_mask, kv_mask, *video_kv):
        # Keep the shared context fan-out in one graph so gradient readiness
        # matches eager ActionDiT and ZeRO bucket reduction order.
        mot = self.__dict__["_mot"]
        for idx, layer in enumerate(self.layers):
            expert = mot.mixtures["action"] if mot is not None else None
            if mot is None:
                x = layer(x, freqs, t_mod, context, context_mask, kv_mask,
                          video_kv[2 * idx], video_kv[2 * idx + 1])
                continue
            q, k, v, residual, gate, shift, scale, gate_mlp, checkpoint = mot._build_expert_attention_io(
                expert, layer.block, x, freqs, t_mod,
            )
            mixed = mot._mixed_attention(
                q, torch.cat([video_kv[2 * idx], k], dim=1),
                torch.cat([video_kv[2 * idx + 1], v], dim=1), kv_mask, expert.num_heads,
            )
            x = mot._apply_post_with_optional_checkpoint(
                layer.block, residual, gate, shift, scale, gate_mlp, checkpoint,
                mixed, {"context": context, "mask": context_mask},
            )
        return x


class _FullAction(torch.nn.Module):
    def __init__(self, mot, action, compact_visible_kv=False, unmasked_visible_kv=False):
        super().__init__()
        self.action = action
        self.inner = _ActionStack(action.blocks, mot)
        self.compact_visible_kv = compact_visible_kv
        self.unmasked_visible_kv = unmasked_visible_kv

    def forward(self, noisy_action, timestep, context, context_mask, attention_mask, *video_kv):
        pre = self.action.pre_dit(noisy_action, timestep, context, context_mask)
        prefix = video_kv[0].shape[1]
        tokens = self.inner(
            pre["tokens"], pre["freqs"], pre["t_mod"], pre["context"], pre["context_mask"],
            None if self.unmasked_visible_kv else (
                attention_mask if self.compact_visible_kv else attention_mask[prefix:, :]
            ), *video_kv,
        )
        return self.action.post_dit(tokens, pre)


class _ParameterEdge(torch.autograd.Function):
    """Keep eager leaf-gradient readiness order outside the captured compute."""

    @staticmethod
    def forward(ctx, parameter):
        return parameter.view_as(parameter)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def _ordered_graph(wrapper, static, ordered_parameters=None):
    # Native DiT backward reaches the leaves in this order. A single graph
    # node makes all leaves ready together; their heap order would otherwise
    # change ZeRO's buckets and BF16 ring-reduction summation order.
    per_block = (
        "ffn.2.bias", "ffn.2.weight", "ffn.0.bias", "ffn.0.weight",
        "cross_attn.o.bias", "cross_attn.o.weight", "cross_attn.v.bias", "cross_attn.v.weight",
        "cross_attn.norm_k.weight", "cross_attn.k.bias", "cross_attn.k.weight",
        "cross_attn.norm_q.weight", "cross_attn.q.bias", "cross_attn.q.weight",
        "self_attn.o.bias", "self_attn.o.weight", "self_attn.v.bias", "self_attn.v.weight",
        "self_attn.norm_k.weight", "self_attn.k.bias", "self_attn.k.weight",
        "self_attn.norm_q.weight", "self_attn.q.bias", "self_attn.q.weight", "modulation",
    )
    ordered = list(ordered_parameters) if ordered_parameters is not None else []
    if ordered_parameters is None:
        for layer in reversed(wrapper.layers):
            named = dict(layer.block.named_parameters())
            if set(named) != set(per_block) or not all(p.requires_grad for p in named.values()):
                raise ValueError("Action graph gradient order requires the original trainable DiT block")
            ordered.extend(named[name] for name in per_block)
    parameters = tuple(wrapper.parameters())
    if len(ordered) != len(parameters) or len(set(map(id, ordered))) != len(parameters):
        raise ValueError("Action graph does not accept shared block parameters")
    argument_count = len(static)

    def compute(*flat):
        return wrapper(*flat[:argument_count])

    # Parameters are explicit inputs to the callable, so the live call can
    # attach identity edges without copying or replacing parameter storage.
    graph = torch.cuda.make_graphed_callables(compute, static + parameters, num_warmup_iters=3)

    def replay(*args):
        edges = {id(p): _ParameterEdge.apply(p) for p in reversed(ordered)}
        return graph(*args, *(edges[id(p)] for p in parameters))

    return replay


@torch._dynamo.disable
def replay_full_action(mot, action, noisy_action, timestep, context, context_mask, video_cache, attention_mask,
                       visible_video_tokens=None, exact_visible_kv=False):
    """Capture ActionDiT pre/block/head compute together, keeping ZeRO eager."""
    # MoT owns the effective checkpoint branch: expert configuration alone
    # does not enable checkpointing while the action-stage MoT stays in eval.
    if not torch.is_grad_enabled() or mot.training:
        raise ValueError("Full action graphs require the original no-checkpoint action stage")
    if mot.use_video_tokens or mot.use_fixed_video_layers is not None or torch.is_autocast_enabled():
        raise ValueError("Full action graphs require direct K/V and the explicit BF16 path")
    if any(t.requires_grad for entry in video_cache for t in entry.values()):
        raise ValueError("Full action graphs require a frozen video producer")
    video_tokens = int(video_cache[0]["k"].shape[1])
    compact = visible_video_tokens is not None
    retained_video_tokens = video_tokens
    if compact:
        # The action-only caller derives this certificate from the very same
        # frame/patch geometry that constructs the standard prefix mask.
        visible_video_tokens = int(visible_video_tokens)
        if not 0 < visible_video_tokens <= video_tokens:
            raise ValueError("Invalid structurally visible video prefix")
        # Keep the final partial video tile separate from the action tile,
        # with the original action-key offset modulo 128. Packing both into
        # one tile changes BF16 reduction rounding. For the validated LIBERO
        # geometry: 1960 total / 1176 visible -> 1320 retained (144 masked).
        retained_video_tokens = (
            visible_video_tokens if exact_visible_kv else min(
                video_tokens, ((visible_video_tokens + 127) // 128) * 128 + video_tokens % 128,
            )
        )
        action_rows = attention_mask[video_tokens:, :]
        attention_mask = torch.cat((
            action_rows[:, :retained_video_tokens], action_rows[:, video_tokens:],
        ), dim=-1)
        video_cache = [dict(k=entry["k"][:, :retained_video_tokens],
                            v=entry["v"][:, :retained_video_tokens]) for entry in video_cache]
    args = (noisy_action, timestep, context, context_mask, attention_mask,
            *(t for entry in video_cache for t in (entry["k"], entry["v"])))
    signature = _signature(args)
    parameters = tuple(action.parameters())
    pointers = tuple(p.data_ptr() for p in parameters)
    state = mot.__dict__.setdefault("_perf_action_graph_state", {})
    if "full-action" not in state:
        if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in action.modules()):
            raise ValueError("Full action graph does not accept module hooks")
        wrapper = _FullAction(
            mot, action, compact_visible_kv=compact,
            unmasked_visible_kv=bool(exact_visible_kv and compact),
        )
        # Keep the original frequency values resident for pre_dit during capture.
        action.freqs = action.freqs.to(noisy_action.device)
        # Frozen KV from the first producer become the persistent graph inputs.
        # Keep original layouts; no second full KV allocation, including capture.
        static = tuple(
            t.detach().clone().requires_grad_(t.requires_grad)
            for i, t in enumerate(args)
        )
        native_order = ["head.bias", "head.weight"]
        block_order = (
            "ffn.2.bias", "ffn.2.weight", "ffn.0.bias", "ffn.0.weight",
            "cross_attn.o.bias", "cross_attn.o.weight", "cross_attn.v.bias", "cross_attn.v.weight",
            "cross_attn.norm_k.weight", "cross_attn.k.bias", "cross_attn.k.weight",
            "cross_attn.norm_q.weight", "cross_attn.q.bias", "cross_attn.q.weight",
            "self_attn.o.bias", "self_attn.o.weight", "self_attn.v.bias", "self_attn.v.weight",
            "self_attn.norm_k.weight", "self_attn.k.bias", "self_attn.k.weight",
            "self_attn.norm_q.weight", "self_attn.q.bias", "self_attn.q.weight", "modulation",
        )
        native_order.extend(f"blocks.{i}.{name}" for i in reversed(range(len(action.blocks))) for name in block_order)
        native_order.extend((
            "text_embedding.2.bias", "text_embedding.2.weight", "text_embedding.0.bias", "text_embedding.0.weight",
            "action_encoder.bias", "action_encoder.weight", "time_projection.1.bias", "time_projection.1.weight",
            "time_embedding.2.bias", "time_embedding.2.weight", "time_embedding.0.bias", "time_embedding.0.weight",
        ))
        named = dict(action.named_parameters())
        if set(native_order) != set(named) or not all(p.requires_grad for p in parameters):
            raise ValueError("Full action graph requires the original ActionDiT parameter layout")
        torch.cuda.synchronize()
        start = time.perf_counter()
        capture_start_unix = time.time()
        previous = torch._C._override_stale_capture_stream()
        try:
            torch.autograd.graph.set_override_stale_capture_stream(True)
            replay = _ordered_graph(wrapper, static, tuple(named[name] for name in native_order))
        except BaseException:
            raise
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(previous)
        torch.cuda.synchronize()
        record = {
            "scope": "complete ActionDiT pre/block/head compute", "action_layers": len(action.blocks),
            "capture_seconds": time.perf_counter() - start, "replays": 0, "signature": signature,
            "parameter_count": len(parameters), "parameter_storage_stable": True,
            "capture_stream_override_scoped": True,
            "capture_stream_override_restored": torch._C._override_stale_capture_stream() == previous,
            "shared_context_fanout_internal": True, "ordered_parameter_gradient_edges": True,
            "native_mot_helpers": True, "pre_projection_and_head_internal": True,
            "fallback": "none-fail-closed", "native_ops": True,
            "video_tokens_before_compaction": video_tokens,
            "video_tokens_per_layer": retained_video_tokens,
            "visible_video_tokens": visible_video_tokens,
            "visible_kv_compaction": compact,
            "attention_mask_and_backend": (
                "source visible-prefix unmasked SDPA"
                if exact_visible_kv and compact else "original masked SDPA retained"
            ),
            "capture_start_unix": capture_start_unix,
            "capture_end_unix": time.time(),
        }
        state["full-action"] = (replay, signature, pointers, record)
        _health_records.append(record)
        logger.info("VPP2 full action graph captured seconds=%.4f", record["capture_seconds"])
    replay, expected, expected_pointers, record = state["full-action"]
    if signature != expected or pointers != expected_pointers:
        record["parameter_storage_stable"] = False
        raise RuntimeError("Full action graph input signature or parameter storage changed")
    result = replay(*args)
    record["replays"] += 1
    return result


def _signature(args):
    return tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device), t.requires_grad) for t in args)


def replay_action_layers(mot, x, freqs, t_mod, context_payload, video_cache, attention_mask):
    """Capture once after distributed initialization; reject unsupported scope or storage drift."""
    if not torch.is_grad_enabled() or mot.training:
        raise ValueError("Action graphs require the validated action-only, no-checkpoint training path")
    if mot.use_video_tokens or mot.use_fixed_video_layers is not None:
        raise ValueError("Action graphs require direct per-layer video K/V")
    if context_payload is None or context_payload.get("context") is None or context_payload.get("mask") is None:
        raise ValueError("Action graphs require tensor context and mask")
    if torch.is_autocast_enabled():
        raise ValueError("Action graphs require the original explicit BF16 precision path without autocast")
    context, context_mask = context_payload["context"], context_payload["mask"]
    if any(v.requires_grad for entry in video_cache for v in entry.values()):
        raise ValueError("Action graphs require detached frozen-video cache")
    # Keep wrappers out of the model's module registry/state_dict and optimizer discovery.
    state = mot.__dict__.setdefault("_perf_action_graph_state", {})
    blocks = mot.mixtures["action"].blocks
    video_kv = tuple(t for entry in video_cache for t in (entry["k"], entry["v"]))
    args = (x, freqs, t_mod, context, context_mask, attention_mask, *video_kv)
    signature = _signature(args)
    if "inner-stack" not in state:
        if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in blocks.modules()):
            raise ValueError("Action graph capture does not accept module hooks")
        wrapper = _ActionStack(blocks)
        static = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in args)
        pointers = tuple(p.data_ptr() for p in blocks.parameters())
        torch.cuda.synchronize()
        start = time.perf_counter()
        previous_override = torch._C._override_stale_capture_stream()
        try:
            torch.autograd.graph.set_override_stale_capture_stream(True)
            replay = _ordered_graph(wrapper, static)
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(previous_override)
        torch.cuda.synchronize()
        record = {
            "scope": "complete inner action stack", "action_layers": len(blocks),
            "capture_seconds": time.perf_counter() - start,
            "replays": 0, "signature": signature,
            "static_strides": [list(t.stride()) for t in static],
            "parameter_count": len(pointers), "parameter_storage_stable": True,
            "fallback": "none-fail-closed", "native_ops": True,
            "capture_stream_override_scoped": True,
            "capture_stream_override_restored": torch._C._override_stale_capture_stream() == previous_override,
            "shared_context_fanout_internal": True,
            "ordered_parameter_gradient_edges": True,
        }
        state["inner-stack"] = (replay, signature, pointers, record)
        _health_records.append(record)
        logger.info("VPP2 native action stack graph captured layers=%d seconds=%.4f", len(blocks), record["capture_seconds"])
    wrapper, expected, pointers, record = state["inner-stack"]
    if signature != expected or tuple(p.data_ptr() for p in blocks.parameters()) != pointers:
        record["parameter_storage_stable"] = False
        raise RuntimeError("Action graph input signature or parameter storage changed")
    output = wrapper(*args)
    record["replays"] += 1
    return output


def _write_health():
    root = os.environ.get("OUTPUT_DIR")
    if not root:
        return
    data = {
        "rank": os.environ.get("RANK"), "requested": "native-action-inner-stack-forward-backward-graph",
        "effective": "native-cuda-graph", "fallback": "none-fail-closed",
        "layers": _health_records, "capture_count": len(_health_records),
        "capture_seconds": sum(r["capture_seconds"] for r in _health_records),
        "distributed_scope": "ZeRO leaf hooks, collectives and optimizer remain eager",
        "limitations": ["Fixed shapes and parameter storage only", "No higher-order gradients or concurrent replay"],
    }
    (Path(root) / ("action_graph_health_rank" + str(os.environ.get("RANK", "unknown")) + ".json")).write_text(
        json.dumps(data, indent=2) + "\n"
    )


atexit.register(_write_health)
