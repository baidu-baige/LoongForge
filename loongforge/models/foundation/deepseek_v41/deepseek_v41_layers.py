# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned DeepSeek-V4.1 residual streams and decoder blocks."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.hyper_connection import native_sinkhorn
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.utils import get_pg_rank, make_viewless_tensor
from torch import nn

from .deepseek_v41_attention import DeepseekV41Attention, DeepseekV41AttentionState
from .deepseek_v41_engram import DeepseekV41Engram, DeepseekV41NgramHash


def _rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    value_fp32 = value.float()
    normalized = value_fp32 * torch.rsqrt(value_fp32.square().mean(-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(value.dtype)


class DeepseekV41SwiGLU(nn.Module):
    """Use MCore's clamped activation through its MLP activation hook."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._limit = float(config.activation_func_clamp_value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return bias_swiglu_impl(hidden_states, None, clamp_value=self._limit)


class DeepseekV41SequentialMLP(SequentialMLP):
    """Keep the V4.1 clamp in MCore's sequential expert path."""

    def __init__(self, num_local_experts, config, submodules, pg_collection=None):
        # MCore's sequential weighted activation does not pass the clamp value.
        expert_config = copy(config)
        expert_config.use_te_activation_func = True
        submodules.activation_func = DeepseekV41SwiGLU
        super().__init__(num_local_experts, expert_config, submodules, pg_collection)


class DeepseekV41RMSNorm(nn.Module):
    """FP32 RMS normalization used by the released V4.1 reference."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _rms_norm(hidden_states, self.weight, self.eps)


@dataclass(frozen=True)
class DeepseekV41Mix:
    pre: torch.Tensor
    post: torch.Tensor
    residual: torch.Tensor


class DeepseekV41HyperConnection(nn.Module):
    """Single-pass mHC with the V4.1 parameter and coefficient contract."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.streams = config.hc_mult
        self.hidden_size = config.hidden_size
        self.eps = config.hc_eps
        self.norm_eps = config.layernorm_epsilon
        width = self.streams * (self.streams + 2)
        self.fn = nn.Parameter(torch.empty(width, self.streams * self.hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(width, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        for parameter in (self.fn, self.base, self.scale):
            parameter._keep_in_float32 = True
        config.init_method(self.fn)

    def forward(self, hidden_states: torch.Tensor) -> DeepseekV41Mix:
        flat = hidden_states.flatten(2).float()
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            projected = F.linear(flat, self.fn.float())
            projected = projected * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        scales = torch.cat(
            (
                self.scale.float()[0].expand(self.streams),
                self.scale.float()[1].expand(self.streams),
                self.scale.float()[2].expand(self.streams * self.streams),
            )
        )
        logits = torch.addcmul(self.base.float(), projected, scales)
        pre = torch.sigmoid(logits[..., : self.streams]) + self.eps
        post = 2 * torch.sigmoid(logits[..., self.streams : 2 * self.streams])
        residual = native_sinkhorn(
            logits[..., 2 * self.streams :].unflatten(-1, (self.streams, self.streams)),
            self.config.hc_sinkhorn_iters,
            self.eps,
        )
        return DeepseekV41Mix(pre, post, residual)

    def collapse(self, hidden_states: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return (pre.unsqueeze(-1) * hidden_states.float()).sum(2).to(hidden_states.dtype)

    def expand(
        self,
        output: torch.Tensor,
        residual: torch.Tensor,
        mix: DeepseekV41Mix,
    ) -> torch.Tensor:
        update = mix.post.unsqueeze(-1) * output.unsqueeze(-2)
        mixed = (mix.residual.unsqueeze(-1) * residual.unsqueeze(-2)).sum(2)
        return (update + mixed).to(output.dtype)


class DeepseekV41Router(TopKRouter):
    """DeepSeek-V4.1 router with a separate visual expert correction bias."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep the released visual routing bias in FP32.  It is checkpoint
        # state, not a trainable weight, but the Megatron BF16 wrapper casts
        # buffers before loading HF values.
        self.bias_vl = nn.Parameter(
            torch.zeros(
                self.num_experts,
                dtype=torch.float32,
                device=self.weight.device,
            ),
            requires_grad=False,
        )
        self.bias_vl._keep_in_float32 = True
        self._vision_mask = None

    def _set_bias_vl(self) -> None:
        """Keep the modality bias in fp32 across model wrappers."""
        if self.bias_vl.dtype != torch.float32:
            self.bias_vl.data = self.bias_vl.data.float()

    def set_vision_mask(self, vision_mask: torch.Tensor | None) -> None:
        self._vision_mask = vision_mask

    def routing(self, logits: torch.Tensor, input_ids: torch.Tensor | None = None):
        del input_ids
        self._set_bias_vl()
        sequence_length, batch_size = logits.shape[:2]
        flat_logits = self.apply_z_loss(logits).reshape(-1, self.config.num_moe_experts)
        if self.score_function != "sqrtsoftplus":
            raise ValueError(f"DeepSeek-V4.1 requires sqrtsoftplus routing, got {self.score_function!r}")
        # Keep the released FP32 score and top-k weight path through dispatch.
        # Casting the weights back to BF16 changes the routed expert sum even
        # when the selected expert IDs are unchanged.
        scores = torch.nn.functional.softplus(flat_logits.float()).sqrt()
        text_bias = (
            self.expert_bias
            if self.expert_bias is not None
            else torch.zeros(self.num_experts, dtype=torch.float32, device=scores.device)
        )
        vision_mask = self._vision_mask
        self._vision_mask = None
        if vision_mask is None:
            selection_bias = text_bias
        else:
            if vision_mask.shape != (batch_size, sequence_length):
                raise ValueError("DeepSeek-V4.1 vision routing mask must have shape [batch, sequence]")
            flat_vision = vision_mask.transpose(0, 1).reshape(-1).bool()
            selection_bias = torch.where(
                flat_vision.unsqueeze(-1),
                self.bias_vl.to(device=scores.device),
                text_bias.to(device=scores.device),
            )
        _, top_indices = torch.topk(scores + selection_bias, k=self.topk, dim=-1)
        weights = scores.gather(1, top_indices)
        if self.topk > 1 and self.config.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        if self.config.moe_router_topk_scaling_factor:
            weights = weights * self.config.moe_router_topk_scaling_factor
        routing_probs = torch.zeros_like(scores).scatter(1, top_indices, weights)
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(1, top_indices, True)
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)
        return routing_probs, routing_map


class DeepseekV41MoELayer(MoELayer):
    """Existing MCore MoE dispatcher with V4.1 modality-aware routing."""

    def __init__(self, *args, **kwargs):
        pg_collection = kwargs.get("pg_collection")
        super().__init__(*args, **kwargs)
        self.pg_collection = pg_collection
        self.router = DeepseekV41Router(
            config=self.config,
            pg_collection=pg_collection,
            layer_number=self.layer_number,
            is_mtp_layer=self.is_mtp_layer,
        )

    def set_v41_vision_mask(self, vision_mask: torch.Tensor | None) -> None:
        self.router.set_vision_mask(vision_mask)


class DeepseekV41Block(nn.Module):
    """One V4.1 block: Engram, CSA2, mHC and routed MoE."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int,
        pg_collection: ProcessGroupCollection,
        backend,
    ):
        super().__init__()
        self.config = config
        self.pg_collection = pg_collection
        self.layer_idx = layer_idx
        self.attn_norm = DeepseekV41RMSNorm(config.hidden_size, config.layernorm_epsilon, dtype=config.params_dtype)
        self.ffn_norm = DeepseekV41RMSNorm(config.hidden_size, config.layernorm_epsilon, dtype=config.params_dtype)
        self.attn = DeepseekV41Attention(config, layer_idx, backend, pg_collection=pg_collection)
        self.attn_hc = DeepseekV41HyperConnection(config)
        self.ffn_hc = DeepseekV41HyperConnection(config)
        from .deepseek_v41_layer_spec import get_deepseek_v41_moe_spec

        self.mlp = build_module(
            get_deepseek_v41_moe_spec(config, backend),
            config=config,
            layer_number=layer_idx + 1,
            pg_collection=pg_collection,
        )
        # Offline conversion shards the replicated Engram tables over TP and
        # repeats the same rows across EP ranks. Keep the runtime owner group
        # aligned with that checkpoint contract.
        engram_owner_group = getattr(pg_collection, "tp", None)
        self.engram = (
            DeepseekV41Engram(
                config,
                layer_idx,
                process_group=engram_owner_group,
            )
            if config.engram_enabled and layer_idx in config.engram_layer_ids
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
        state: DeepseekV41AttentionState,
        *,
        input_ids: torch.Tensor,
        engram_hash_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_mask: torch.Tensor | None = None,
        vision_token_types: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepseekV41AttentionState]:
        if self.engram is not None:
            if input_ids is None:
                raise ValueError("DeepSeek-V4.1 Engram requires input_ids")
            if engram_hash_ids is None:
                raise ValueError("DeepSeek-V4.1 Engram requires tokenizer-derived hash IDs")
            token_mask = None if image_mask is None else ~image_mask.bool()
            if attention_mask is not None:
                token_mask = attention_mask.bool() if token_mask is None else token_mask & attention_mask.bool()
            hidden_states = self.engram(
                hidden_states,
                engram_hash_ids,
                token_mask=token_mask,
            )

        attn_mix = self.attn_hc(hidden_states)
        collapsed = self.attn_hc.collapse(hidden_states, pre_mix)
        attended, state = self.attn(
            self.attn_norm(collapsed),
            state=state,
            position_ids=position_ids,
            attention_mask=attention_mask,
            sequence_ids=sequence_ids,
            vision_token_types=vision_token_types,
        )
        hidden_states = self.attn_hc.expand(attended, hidden_states, attn_mix)

        ffn_mix = self.ffn_hc(hidden_states)
        collapsed = self.ffn_hc.collapse(hidden_states, attn_mix.pre)
        mlp_input = self.ffn_norm(collapsed).transpose(0, 1).contiguous()
        if self.config.sequence_parallel:
            mlp_input = tensor_parallel.scatter_to_sequence_parallel_region(mlp_input, group=self.pg_collection.tp)
            if image_mask is not None:
                local_length = mlp_input.shape[0]
                start = get_pg_rank(self.pg_collection.tp) * local_length
                image_mask = image_mask[:, start : start + local_length]
        self.mlp.set_v41_vision_mask(image_mask)
        mlp_output, mlp_bias = self.mlp(
            mlp_input,
            input_ids=None,
        )
        if mlp_bias is not None:
            mlp_output = mlp_output + mlp_bias
        if self.config.sequence_parallel:
            mlp_output = tensor_parallel.gather_from_sequence_parallel_region(
                mlp_output, tensor_parallel_output_grad=False, group=self.pg_collection.tp
            )
        mlp_output = mlp_output.transpose(0, 1).contiguous()
        hidden_states = self.ffn_hc.expand(mlp_output, hidden_states, ffn_mix)
        return hidden_states, ffn_mix.pre, state


_CARRY_PRE = 0
_CARRY_INPUT_IDS = 1
_CARRY_IMAGE_MASK = 2
_CARRY_VISION_TOKEN_TYPES = 3
_CARRY_SEQUENCE_IDS = 4
_CARRY_POSITION_IDS = 5
_CARRY_COMPRESSED_KV = 6
_CARRY_COMPRESSED_LATENT = 7
_CARRY_COMPRESSED_POSITIONS = 8
_CARRY_COMPRESSED_VALID = 9
_CARRY_COMPRESSED_SEQUENCE_IDS = 10
_CARRY_INDEX_KEYS = 11
_CARRY_TOPK_INDICES = 12
_CARRY_CANDIDATES = 13
_CARRY_RATIO = 14
_CARRY_VALID_TOKENS = 15
_CARRY_HEADER_INTS = 1 + 8 * 16


def _carrier_int_units(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 2
    if dtype == torch.float32:
        return 1
    raise ValueError("DeepSeek-V4.1 PP carry requires bf16 or fp32 hidden states")


def _encode_int(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    value = value.detach().contiguous()
    if value.dtype == torch.bool:
        raw = value.to(torch.uint8).flatten()
        unit_bytes = torch.tensor([], dtype=dtype).element_size()
        raw = F.pad(raw, (0, (-raw.numel()) % unit_bytes))
        return raw.view(dtype)
    raw = value.to(torch.int32).flatten()
    return raw.view(dtype)


def _pack_float(value: torch.Tensor, dtype: torch.dtype, preserve_fp32: bool = False):
    if preserve_fp32 and dtype == torch.bfloat16 and value.dtype == torch.float32:
        high = value.to(dtype)
        low = (value - high.float()).to(dtype)
        return torch.cat((high.flatten(), low.flatten()))
    return value.to(dtype).contiguous().flatten()


def _unpack_float(value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, preserve_fp32: bool):
    if preserve_fp32 and dtype == torch.bfloat16:
        size = int(torch.tensor(shape).prod().item())
        return (value[:size].float() + value[size : 2 * size].float()).view(shape)
    return value.view(shape)


def _unpack_int(value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Decode int32 payloads without requiring an aligned bf16 storage offset."""
    size = int(torch.tensor(shape).prod().item())
    raw = value.view(torch.uint8)[: size * 4].clone().view(torch.int32)
    return raw.view(shape).long()


def _pack_v41_carry(
    hidden_states: torch.Tensor,
    pre_mix: torch.Tensor,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    image_mask: torch.Tensor | None,
    vision_token_types: torch.Tensor | None,
    sequence_ids: torch.Tensor | None,
    state: DeepseekV41AttentionState,
    valid_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    dtype = hidden_states.dtype
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("DeepSeek-V4.1 PP carry requires bf16 or fp32 hidden states")
    records = []

    def add(kind, value, *, key=0, boolean=False, preserve_fp32=False):
        if value is None:
            return
        shape = tuple(value.shape)
        if len(shape) > 4:
            raise ValueError(f"DeepSeek-V4.1 PP carry does not support shape {shape}")
        padded_shape = shape + (0,) * (4 - len(shape))
        if boolean or value.dtype in (torch.int32, torch.int64, torch.bool):
            payload = _encode_int(value, dtype)
        else:
            payload = _pack_float(value, dtype, preserve_fp32=preserve_fp32)
        records.append((kind, key, len(shape), *padded_shape, payload))

    add(_CARRY_PRE, pre_mix, preserve_fp32=True)
    add(_CARRY_INPUT_IDS, input_ids)
    add(_CARRY_POSITION_IDS, position_ids)
    add(_CARRY_IMAGE_MASK, image_mask, boolean=True)
    add(_CARRY_VISION_TOKEN_TYPES, vision_token_types)
    add(_CARRY_SEQUENCE_IDS, sequence_ids)
    add(_CARRY_COMPRESSED_KV, state.compressed_kv)
    add(_CARRY_COMPRESSED_LATENT, state.compressed_latent)
    add(_CARRY_COMPRESSED_POSITIONS, state.compressed_positions)
    add(_CARRY_COMPRESSED_VALID, state.compressed_valid, boolean=True)
    add(_CARRY_COMPRESSED_SEQUENCE_IDS, state.compressed_sequence_ids)
    add(_CARRY_INDEX_KEYS, state.index_keys)
    add(_CARRY_TOPK_INDICES, state.topk_indices)
    add(_CARRY_CANDIDATES, state.candidates, boolean=True)
    add(_CARRY_VALID_TOKENS, valid_tokens, boolean=True)
    if state.compression_ratio:
        records.append((_CARRY_RATIO, state.compression_ratio, 0, 0, 0, 0, 0, hidden_states.new_empty(0)))

    header = torch.zeros(_CARRY_HEADER_INTS, dtype=torch.int32, device=hidden_states.device)
    header[0] = len(records)
    payloads = [header.view(dtype)]
    offset = _CARRY_HEADER_INTS * _carrier_int_units(dtype)
    for index, (kind, key, ndim, d0, d1, d2, d3, payload) in enumerate(records):
        base = 1 + index * 8
        header[base : base + 8] = torch.tensor(
            [kind, key, ndim, d0, d1, d2, d3, payload.numel()],
            dtype=torch.int32,
            device=hidden_states.device,
        )
        payloads.append(payload)
        offset += payload.numel()
    # Rewrite the header after filling its records.
    payloads[0] = header.view(dtype)
    flat = torch.cat(payloads)
    rows = hidden_states.shape[0] * hidden_states.shape[1]
    flat = F.pad(flat, (0, (-flat.numel()) % rows))
    return torch.cat((hidden_states, flat.view(hidden_states.shape[0], hidden_states.shape[1], -1)), dim=-1)


def _unpack_v41_carry(
    hidden_states: torch.Tensor,
    base_hidden_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    DeepseekV41AttentionState,
]:
    dtype = hidden_states.dtype
    if hidden_states.shape[-1] == base_hidden_size:
        raise ValueError("DeepSeek-V4.1 PP carry is missing on a non-first pipeline stage")
    carry = hidden_states[..., base_hidden_size:].contiguous().flatten()
    header_units = _CARRY_HEADER_INTS * _carrier_int_units(dtype)
    header = carry[:header_units].view(torch.int32).tolist()
    count = header[0]
    values = {
        "pre": None,
        "input_ids": None,
        "position_ids": None,
        "image_mask": None,
        "vision_token_types": None,
        "sequence_ids": None,
        "valid_tokens": None,
    }
    state = {}
    ratio = 0
    offset = header_units
    for index in range(count):
        base = 1 + index * 8
        kind, key, ndim, d0, d1, d2, d3, length = header[base : base + 8]
        shape = tuple(x for x in (d0, d1, d2, d3)[:ndim])
        payload = carry[offset : offset + length]
        offset += length
        if kind == _CARRY_RATIO:
            ratio = key
        elif kind == _CARRY_PRE:
            values["pre"] = _unpack_float(payload, shape, dtype, preserve_fp32=True)
        elif kind in (
            _CARRY_INPUT_IDS,
            _CARRY_POSITION_IDS,
            _CARRY_VISION_TOKEN_TYPES,
            _CARRY_SEQUENCE_IDS,
            _CARRY_TOPK_INDICES,
        ):
            values_map = {
                _CARRY_INPUT_IDS: "input_ids",
                _CARRY_POSITION_IDS: "position_ids",
                _CARRY_VISION_TOKEN_TYPES: "vision_token_types",
                _CARRY_SEQUENCE_IDS: "sequence_ids",
            }
            raw = _unpack_int(payload, shape)
            if kind == _CARRY_TOPK_INDICES:
                state["topk_indices"] = raw
            else:
                values[values_map[kind]] = raw
        elif kind in (_CARRY_IMAGE_MASK, _CARRY_COMPRESSED_VALID, _CARRY_CANDIDATES, _CARRY_VALID_TOKENS):
            raw = payload.view(torch.uint8)[: int(torch.tensor(shape).prod().item())].bool().view(shape)
            if kind == _CARRY_IMAGE_MASK:
                values["image_mask"] = raw
            elif kind == _CARRY_COMPRESSED_VALID:
                state["compressed_valid"] = raw
            elif kind == _CARRY_VALID_TOKENS:
                values["valid_tokens"] = raw
            else:
                state["candidates"] = raw
        else:
            state_map = {
                _CARRY_COMPRESSED_KV: "compressed_kv",
                _CARRY_COMPRESSED_LATENT: "compressed_latent",
                _CARRY_COMPRESSED_POSITIONS: "compressed_positions",
                _CARRY_COMPRESSED_SEQUENCE_IDS: "compressed_sequence_ids",
                _CARRY_INDEX_KEYS: "index_keys",
            }
            if kind in state_map:
                if kind in (_CARRY_COMPRESSED_POSITIONS, _CARRY_COMPRESSED_SEQUENCE_IDS):
                    state[state_map[kind]] = _unpack_int(payload, shape)
                else:
                    state[state_map[kind]] = payload.view(shape)
    return (
        hidden_states[..., :base_hidden_size].contiguous(),
        values["pre"],
        values["input_ids"],
        values["position_ids"],
        values["image_mask"],
        values["vision_token_types"],
        values["sequence_ids"],
        values["valid_tokens"],
        DeepseekV41AttentionState(compression_ratio=ratio, **state),
    )


class DeepseekV41DecoderBlock(nn.Module):
    """A V4.1 decoder compatible with LoongForge's BaseGPTModel wrapper."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: ModuleSpec | None,
        pre_process: bool,
        post_process: bool,
        pg_collection: ProcessGroupCollection,
        vp_stage: Optional[int] = None,
    ):
        super().__init__()
        if getattr(config, "pipeline_model_parallel_layout", None) is not None:
            raise ValueError(
                "DeepSeek-V4.1 does not support an explicit PP layout; "
                "set the virtual chunk sizes with --num-virtual-stages-per-pipeline-rank instead"
            )
        backend = (
            TESpecProvider()
            if getattr(config, "transformer_impl", "local") == "transformer_engine"
            else LocalSpecProvider()
        )
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.pg_collection = pg_collection
        pp_rank = get_pg_rank(pg_collection.pp)
        self.layer_offset = get_transformer_layer_offset(config, vp_stage, pp_rank)
        self.num_layers_to_build = get_num_layers_to_build(config, vp_stage, pp_rank)
        self.first_layer = self.layer_offset
        self.last_layer = self.layer_offset + self.num_layers_to_build
        self.engram_hash = None
        if config.engram_enabled and any(
            self.first_layer <= layer < self.last_layer for layer in config.engram_layer_ids
        ):
            from loongforge.utils import get_tokenizer

            self.engram_hash = DeepseekV41NgramHash(config, get_tokenizer().hf_tokenizer())
        self.layers = nn.ModuleList(
            DeepseekV41Block(config, layer_idx, pg_collection, backend)
            for layer_idx in range(self.first_layer, self.last_layer)
        )
        self.final_layernorm = (
            DeepseekV41RMSNorm(config.hidden_size, config.layernorm_epsilon, dtype=config.params_dtype)
            if post_process
            else None
        )

    def set_input_tensor(self, input_tensor) -> None:
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        input_ids: torch.Tensor,
        v41_position_ids: torch.Tensor | None = None,
        v41_image_mask: torch.Tensor | None = None,
        v41_vision_token_types: torch.Tensor | None = None,
        v41_sequence_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states is None:
            hidden_states = self.input_tensor
        if hidden_states.ndim != 3:
            raise ValueError("DeepSeek-V4.1 decoder expects [sequence, batch, hidden]")
        sequence_parallel = bool(self.config.sequence_parallel)
        if sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=self.pg_collection.tp,
            )
        stream_hidden_size = self.config.hc_mult * self.config.hidden_size
        carried = hidden_states.shape[-1] > stream_hidden_size
        carried_input_ids = None
        if carried:
            (
                hidden_states,
                carried_pre_mix,
                carried_input_ids,
                carried_position_ids,
                carried_image_mask,
                carried_vision_token_types,
                carried_sequence_ids,
                carried_valid_tokens,
                state,
            ) = _unpack_v41_carry(hidden_states, stream_hidden_size)
            if carried_pre_mix is None:
                raise ValueError("DeepSeek-V4.1 PP carry did not contain mHC coefficients")
            pre_mix = carried_pre_mix
            if carried_input_ids is not None:
                input_ids = carried_input_ids
            if carried_position_ids is not None:
                v41_position_ids = carried_position_ids
            if carried_image_mask is not None:
                v41_image_mask = carried_image_mask
            if carried_vision_token_types is not None:
                v41_vision_token_types = carried_vision_token_types
            if carried_sequence_ids is not None:
                v41_sequence_ids = carried_sequence_ids
            if carried_valid_tokens is not None:
                attention_mask = carried_valid_tokens
        else:
            state = DeepseekV41AttentionState()
            pre_mix = None
        sequence, batch, _ = hidden_states.shape
        if input_ids.shape != (batch, sequence):
            raise ValueError("DeepSeek-V4.1 input_ids must be [batch, sequence]")
        if v41_position_ids is None:
            v41_position_ids = torch.arange(sequence, device=hidden_states.device).expand(batch, sequence)
        if v41_position_ids.shape != (batch, sequence):
            raise ValueError("DeepSeek-V4.1 position_ids must be [batch, sequence]")

        streams = self.config.hc_mult
        if carried:
            hidden_states = (
                hidden_states.transpose(0, 1).reshape(batch, sequence, streams, self.config.hidden_size).contiguous()
            )
        else:
            hidden_states = (
                hidden_states.transpose(0, 1)
                .unsqueeze(2)
                .expand(batch, sequence, streams, self.config.hidden_size)
                .contiguous()
            )
        if pre_mix is None:
            pre_mix = torch.zeros(
                batch,
                sequence,
                streams,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            pre_mix[..., 0] = 1
        valid = _binary_attention_mask(attention_mask, batch, sequence).to(hidden_states.device)
        token_mask = valid
        if v41_image_mask is not None:
            token_mask = token_mask & ~v41_image_mask.bool()
        hashes = (
            None
            if self.engram_hash is None
            else self.engram_hash(
                input_ids,
                token_mask=token_mask,
                sequence_ids=v41_sequence_ids,
            )
        )
        for layer in self.layers:
            hidden_states, pre_mix, state = layer(
                hidden_states,
                pre_mix,
                state,
                input_ids=input_ids,
                engram_hash_ids=(
                    None if hashes is None or layer.engram is None else hashes[:, :, layer.engram.hash_index]
                ),
                position_ids=v41_position_ids,
                attention_mask=valid,
                image_mask=v41_image_mask,
                vision_token_types=v41_vision_token_types,
                sequence_ids=v41_sequence_ids,
            )
        if not self.post_process:
            hidden_states = hidden_states.transpose(0, 1).reshape(sequence, batch, stream_hidden_size).contiguous()
            output = _pack_v41_carry(
                hidden_states,
                pre_mix,
                input_ids,
                v41_position_ids,
                v41_image_mask,
                v41_vision_token_types,
                v41_sequence_ids,
                state,
                valid_tokens=valid,
            )
        else:
            hidden_states = self.layers[-1].attn_hc.collapse(hidden_states, pre_mix)
            if self.final_layernorm is not None:
                hidden_states = self.final_layernorm(hidden_states)
            output = hidden_states.transpose(0, 1).float().contiguous()
        if sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output,
                group=self.pg_collection.tp,
            )
        return make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)


def _binary_attention_mask(attention_mask: torch.Tensor | None, batch: int, sequence: int) -> torch.Tensor:
    """Normalize LoongForge causal/padding masks to [batch, sequence] validity."""
    if attention_mask is None:
        return torch.ones(batch, sequence, dtype=torch.bool)
    if attention_mask.ndim == 2:
        if attention_mask.shape != (batch, sequence):
            raise ValueError("DeepSeek-V4.1 attention mask must be [batch, sequence]")
        return attention_mask.bool()
    if attention_mask.ndim == 4:
        if attention_mask.shape[0] not in (1, batch):
            raise ValueError("DeepSeek-V4.1 attention mask has invalid batch size")
        mask = attention_mask
        if mask.shape[-2] == 1:
            valid = ~mask[:, 0, 0].bool()
        else:
            valid = ~mask[:, 0].diagonal(dim1=-2, dim2=-1).bool()
        if valid.shape[0] == 1 and batch > 1:
            valid = valid.expand(batch, -1)
        if valid.shape != (batch, sequence):
            raise ValueError("DeepSeek-V4.1 attention mask has invalid shape")
        return valid
    raise ValueError("DeepSeek-V4.1 attention mask must be [B,S] or [B,1,S,S]")
