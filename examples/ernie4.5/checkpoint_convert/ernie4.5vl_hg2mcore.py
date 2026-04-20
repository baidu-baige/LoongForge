# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file
"""
ERNIE-4.5-VL HuggingFace → Megatron-Core checkpoint converter.

Usage (PP=1):
    python ernie4.5vl_hg2mcore.py \
        --load_hg_path    /path/to/hf_ckpt \
        --save_mcore_path /path/to/mcore_out \
        --load_mcore_path /path/to/mcore_template \
        --tp 1 --pp 1 \
        --num_vit_layers 32 --num_lm_layers 62 --num_experts 64

Usage (PP=8, custom pipeline layer split):
    python ernie4.5vl_hg2mcore.py \
        --load_hg_path    /path/to/hf_ckpt \
        --save_mcore_path /path/to/mcore_out \
        --load_mcore_path /path/to/mcore_template \
        --tp 1 --pp 8 \
        --num_vit_layers 32 --num_lm_layers 62 --num_experts 64 \
        --pp_layer_offsets 0,4,8,12,16,20,24,26

Notes:
    --pp_layer_offsets  Comma-separated list of global LM-layer start indices,
                        one per PP stage. Required when --pp > 1.
                        Each value is the index of the first LM layer in that
                        pipeline stage within the full HF model.
                        Length must equal --pp.
                        When --pp 1 this argument can be omitted (defaults to 0).
"""
import os
import re
import argparse
import torch
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Weight-mapping primitives
# ---------------------------------------------------------------------------

@dataclass
class TransformOp:
    """One HF-key → MCore-key mapping entry."""
    src_name: str
    dst_name: str
    forward_fn: Callable  = None   # filled by WeightMapping


class WeightMapping:
    """Accumulates TransformOps and applies a shared forward_fn."""

    def __init__(self, forward_fn: Callable):
        self.ops: List[TransformOp] = []
        self._default_fn = forward_fn

    def add(self, src: str, dst: str, forward_fn: Callable = None) -> None:
        fn = forward_fn if forward_fn is not None else self._default_fn
        self.ops.append(TransformOp(src_name=src, dst_name=dst, forward_fn=fn))


# ---------------------------------------------------------------------------
# Per-tensor transform functions
# ---------------------------------------------------------------------------

def copy_fn(ops: List[TransformOp], hf_sd: dict, expected_shape=None) -> torch.Tensor:
    """Direct copy, with optional transpose/reshape when shapes mismatch."""
    src = ops[0].src_name
    tensor = hf_sd[src]
    if expected_shape is not None and tensor.shape != expected_shape:
        if len(tensor.shape) == len(expected_shape) == 2:
            tensor = tensor.transpose(1, 0)
        else:
            tensor = tensor.reshape(expected_shape)
    return tensor


def split_fn(ops: List[TransformOp], hf_sd: dict, expected_shape=None) -> torch.Tensor:
    """Split one HF tensor into two MCore tensors via 'key@0' / 'key@1' suffix.

    Example: e_score_correction_bias [2, 64] → split → [1, 64] → reshape → [64]
    The reshape step matches the original forward_func behaviour.
    """
    raw, part = ops[0].src_name.split("@")
    result = hf_sd[raw].split([1, 1], dim=0)[int(part)]
    if expected_shape is not None and result.shape != expected_shape:
        if len(result.shape) == len(expected_shape):
            result = result.transpose(1, 0)
        else:
            result = result.reshape(expected_shape)
    return result


def merge_qkv_lm_fn(ops: List[TransformOp], hf_sd: dict, expected_shape=None) -> torch.Tensor:
    """Merge separate q_proj / k_proj / v_proj into Megatron linear_qkv.

    HF layout  : [q_proj, k_proj, v_proj]  (independent Linear layers)
    MCore layout: interleaved per head — [q0,k0,v0, q1,k1,v1, ..., qH,kH,vH]

    Concrete shapes for ERNIE-4.5-VL-28B-A3B (4 KV-heads, 20 Q-heads):
        q_proj : [2560, 2560]  (20 heads × 128)
        k_proj : [512,  2560]  (4  heads × 128)
        v_proj : [512,  2560]  (4  heads × 128)
    """
    assert len(ops) == 3, f"Expected 3 ops (q/k/v), got {len(ops)}"
    q_name, k_name, v_name = [op.src_name for op in ops]
    q = hf_sd[q_name]   # [num_q_heads * head_dim, hidden]
    k = hf_sd[k_name]   # [num_kv_heads * head_dim, hidden]
    v = hf_sd[v_name]   # [num_kv_heads * head_dim, hidden]

    num_kv_heads = 4
    num_q_per_kv = 5    # GQA ratio: 20 Q heads / 4 KV heads
    head_dim = 128
    hidden   = q.shape[-1]  # 2560

    q = q.reshape(num_kv_heads, num_q_per_kv, head_dim, hidden)
    k = k.reshape(num_kv_heads, 1,            head_dim, hidden)
    v = v.reshape(num_kv_heads, 1,            head_dim, hidden)
    # interleave: [kv_groups, q+k+v, head_dim, hidden] → flatten first two dims
    qkv = torch.cat([q, k, v], dim=1).reshape(-1, hidden)  # [(q+k+v)*heads, hidden]
    return qkv.contiguous()


def reorder_vit_qkv_weight(ops, hf_sd, expected_shape=None):
    """ViT QKV weight: [Q_all, K_all, V_all] → interleaved [q0,k0,v0, q1,k1,v1, ...]

    ERNIE-4.5-VL ViT config: embed_dim=1280, num_heads=16, head_dim=80
    """
    num_heads = 16
    head_dim  = 80
    embed_dim = 1280

    w = hf_sd[ops[0].src_name]                              # [3*1280, 1280]
    w = w.reshape(3, num_heads, head_dim, embed_dim)        # [3, H, D, E]
    w = w.permute(1, 0, 2, 3).reshape(3 * embed_dim, embed_dim)
    return w.contiguous()


def reorder_vit_qkv_bias(ops, hf_sd, expected_shape=None):
    """ViT QKV bias: [Q_all, K_all, V_all] → interleaved [q0,k0,v0, ...]"""
    num_heads = 16
    head_dim  = 80
    embed_dim = 1280

    b = hf_sd[ops[0].src_name]                              # [3*1280]
    b = b.reshape(3, num_heads, head_dim)                   # [3, H, D]
    b = b.permute(1, 0, 2).reshape(3 * embed_dim)
    return b.contiguous()


# ---------------------------------------------------------------------------
# Weight-map builders
# ---------------------------------------------------------------------------

def build_vision_mapping(args) -> WeightMapping:
    """Vision encoder (ViT) + resampler weight mapping."""
    m = WeightMapping(forward_fn=copy_fn)

    # --- patch embedding & final layernorm ---
    m.add("vision_model.patch_embed.proj.weight",
          "encoder_model.image_encoder.patch_embed.proj.weight")
    m.add("vision_model.ln.weight", "encoder_model.image_encoder.ln.weight")
    m.add("vision_model.ln.bias",   "encoder_model.image_encoder.ln.bias")

    # --- ViT transformer layers ---
    for i in range(args.num_vit_layers):
        hg = f"vision_model.blocks.{i}"
        mc = f"encoder_model.image_encoder.decoder.layers.{i}"

        m.add(f"{hg}.norm1.weight", f"{mc}.input_layernorm.weight")
        m.add(f"{hg}.norm1.bias",   f"{mc}.input_layernorm.bias")

        m.add(f"{hg}.attn.qkv.weight", f"{mc}.self_attention.linear_qkv.weight",
              forward_fn=reorder_vit_qkv_weight)
        m.add(f"{hg}.attn.qkv.bias",   f"{mc}.self_attention.linear_qkv.bias",
              forward_fn=reorder_vit_qkv_bias)

        m.add(f"{hg}.attn.proj.weight", f"{mc}.self_attention.linear_proj.weight")
        m.add(f"{hg}.attn.proj.bias",   f"{mc}.self_attention.linear_proj.bias")

        m.add(f"{hg}.norm2.weight", f"{mc}.pre_mlp_layernorm.weight")
        m.add(f"{hg}.norm2.bias",   f"{mc}.pre_mlp_layernorm.bias")

        m.add(f"{hg}.mlp.fc1.weight", f"{mc}.mlp.linear_fc1.weight")
        m.add(f"{hg}.mlp.fc1.bias",   f"{mc}.mlp.linear_fc1.bias")
        m.add(f"{hg}.mlp.fc2.weight", f"{mc}.mlp.linear_fc2.weight")
        m.add(f"{hg}.mlp.fc2.bias",   f"{mc}.mlp.linear_fc2.bias")

    # --- resampler ---
    hg_base = "model.resampler_model"
    mc_base = "encoder_model.image_encoder.resampler"
    for suffix in [
        ".mlp.bias", ".mlp.weight",
        ".spatial_linear.0.bias",   ".spatial_linear.0.weight",
        ".spatial_linear.2.bias",   ".spatial_linear.2.weight",
        ".spatial_linear.3.bias",   ".spatial_linear.3.weight",
        ".temporal_linear.0.bias",  ".temporal_linear.0.weight",
        ".temporal_linear.2.bias",  ".temporal_linear.2.weight",
        ".temporal_linear.3.bias",  ".temporal_linear.3.weight",
    ]:
        m.add(f"{hg_base}{suffix}", f"{mc_base}{suffix}")

    return m


def build_adapter_mapping(args) -> WeightMapping:
    """Image projector (adapter) weight mapping."""
    m = WeightMapping(forward_fn=copy_fn)
    hg = "model.resampler_model"
    mc = "encoder_model.image_projector"
    m.add(f"{hg}.mlp.weight",        f"{mc}.mlp.weight")
    m.add(f"{hg}.mlp.bias",          f"{mc}.mlp.bias")
    m.add(f"{hg}.after_norm.weight",  f"{mc}.after_norm.weight")
    return m


def build_language_mapping(args) -> WeightMapping:
    """Language model (LLM backbone) weight mapping."""
    m = WeightMapping(forward_fn=copy_fn)

    # --- embeddings & final norm ---
    m.add("model.embed_tokens.weight", "encoder_model.text_encoder.word_embeddings.weight")
    m.add("model.embed_tokens.weight", "foundation_model.embedding.word_embeddings.weight")
    m.add("model.embed_tokens.weight", "foundation_model.output_layer.weight")
    m.add("model.norm.weight",         "foundation_model.decoder.final_layernorm.weight")

    for i in range(args.num_lm_layers):
        hg = f"model.layers.{i}"
        mc = f"foundation_model.decoder.layers.{i}"

        # --- self attention ---
        m.add(f"{hg}.input_layernorm.weight",  f"{mc}.input_layernorm.weight")
        # q/k/v are merged by merge_qkv_lm_fn; all three must share the same dst_name
        m.add(f"{hg}.self_attn.q_proj.weight", f"{mc}.self_attention.linear_qkv.weight",
              forward_fn=merge_qkv_lm_fn)
        m.add(f"{hg}.self_attn.k_proj.weight", f"{mc}.self_attention.linear_qkv.weight",
              forward_fn=merge_qkv_lm_fn)
        m.add(f"{hg}.self_attn.v_proj.weight", f"{mc}.self_attention.linear_qkv.weight",
              forward_fn=merge_qkv_lm_fn)
        m.add(f"{hg}.self_attn.o_proj.weight", f"{mc}.self_attention.linear_proj.weight")
        m.add(f"{hg}.post_attention_layernorm.weight", f"{mc}.pre_mlp_layernorm.weight")

        if i == 0:
            # Layer 0: dense MLP (no MoE)
            m.add(f"{hg}.mlp.gate_proj.weight", f"{mc}.mlp.linear_fc1.weight")
            m.add(f"{hg}.mlp.up_proj.weight",   f"{mc}.mlp.linear_fc1_1.weight")
            m.add(f"{hg}.mlp.down_proj.weight",  f"{mc}.mlp.linear_fc2.weight")
        else:
            # Layers 1+: MultiTypeMoE
            _add_moe_layer_mapping(m, hg, mc, args.num_experts)

    return m


def _add_moe_layer_mapping(m: WeightMapping, hg: str, mc: str, num_experts: int) -> None:
    """Add weight mappings for one ErnieMultiTypeMoE layer."""

    # routers
    m.add(f"{hg}.mlp.gate.weight",   f"{mc}.mlp.text_moe_layer.router.weight")
    m.add(f"{hg}.mlp.gate.weight_1", f"{mc}.mlp.vision_moe_layer.router.weight")

    # expert bias (one tensor split into text / vision halves via '@' notation)
    bias_src = f"{hg}.mlp.moe_statics.e_score_correction_bias"
    m.add(f"{bias_src}@0", f"{mc}.mlp.text_moe_layer.router.expert_bias",
          forward_fn=split_fn)
    m.add(f"{bias_src}@1", f"{mc}.mlp.vision_moe_layer.router.expert_bias",
          forward_fn=split_fn)

    # shared experts
    se_hg = f"{hg}.mlp.shared_experts"
    se_mc = f"{mc}.mlp.shared_experts"
    m.add(f"{se_hg}.gate_proj.weight", f"{se_mc}.linear_fc1.weight")
    m.add(f"{se_hg}.up_proj.weight",   f"{se_mc}.linear_fc1_1.weight")
    m.add(f"{se_hg}.down_proj.weight", f"{se_mc}.linear_fc2.weight")

    # text experts  (HF expert 0..num_experts-1)
    for j in range(num_experts):
        _add_expert_mapping(m,
            src_expert=f"{hg}.mlp.experts.{j}",
            dst_expert=f"{mc}.mlp.text_moe_layer.experts.local_experts.{j}")

    # vision experts (HF expert num_experts..2*num_experts-1)
    for j in range(num_experts):
        _add_expert_mapping(m,
            src_expert=f"{hg}.mlp.experts.{j + num_experts}",
            dst_expert=f"{mc}.mlp.vision_moe_layer.experts.local_experts.{j}")


def _add_expert_mapping(m: WeightMapping, src_expert: str, dst_expert: str) -> None:
    """gate_proj / up_proj / down_proj for one expert."""
    m.add(f"{src_expert}.gate_proj.weight", f"{dst_expert}.linear_fc1.weight")
    m.add(f"{src_expert}.up_proj.weight",   f"{dst_expert}.linear_fc1_1.weight")
    m.add(f"{src_expert}.down_proj.weight", f"{dst_expert}.linear_fc2.weight")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _shard_dir(t: int, p: int, pp: int) -> str:
    return f"mp_rank_{t:02d}" if pp == 1 else f"mp_rank_{t:02d}_{p:03d}"


def load_mcore_checkpoints(load_path: str, tp: int, pp: int) -> list:
    state_dict = [[None] * tp for _ in range(pp)]
    for p in range(pp):
        for t in range(tp):
            ckpt = os.path.join(load_path, _shard_dir(t, p, pp), "model_optim_rng.pt")
            state_dict[p][t] = torch.load(ckpt, map_location="cpu", weights_only=False)
    return state_dict


def save_mcore_checkpoints(state_dict: list, save_path: str, tp: int, pp: int) -> None:
    root = Path(save_path) / "release"
    for p in range(pp):
        for t in range(tp):
            out_dir = root / _shard_dir(t, p, pp)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict[p][t], out_dir / "model_optim_rng.pt")
    (Path(save_path) / "latest_checkpointed_iteration.txt").write_text("release")


def load_huggingface_checkpoints(hg_path: str) -> dict:
    if not os.path.isdir(hg_path):
        raise FileNotFoundError(f"HuggingFace checkpoint directory not found: {hg_path}")
    files = [f for f in os.listdir(hg_path) if f.endswith(".safetensors")]
    print(f"Found {len(files)} safetensors files in {hg_path}")
    merged = {}
    for fname in sorted(files):
        path = os.path.join(hg_path, fname)
        print(f"  Loading {path}")
        merged.update(load_file(path))
    return merged


# ---------------------------------------------------------------------------
# Debug / summary printers
# ---------------------------------------------------------------------------

def _shape(t) -> str:
    return str(tuple(t.shape))


def print_hf_summary(hf_sd: dict, args) -> None:
    """Print a structured subset of HF weights for quick sanity-checking.

    Prints:
      - Global non-layer tensors (embeddings, final norm, resampler, projector)
      - ViT: first 2 layers
      - LM layer 0: dense MLP
      - LM layers 1+: first 2 MoE layers, each showing
          router / shared_experts / first 2 text experts / first 2 vision experts
    """
    W = 80
    print("\n" + "=" * W)
    print(f"  HuggingFace checkpoint summary")
    print(f"  ViT layers      : {args.num_vit_layers}")
    print(f"  LM layers       : {args.num_lm_layers}  (layer 0: dense MLP, layers 1-{args.num_lm_layers-1}: MoE)")
    print(f"  Experts per MoE : {args.num_experts} text + {args.num_experts} vision = {2*args.num_experts} total")
    print("=" * W)

    # ---- global / non-layer tensors ----
    print("\n[Global]")
    global_keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "vision_model.patch_embed.proj.weight",
        "vision_model.ln.weight",
        "vision_model.ln.bias",
        "model.resampler_model.mlp.weight",
        "model.resampler_model.after_norm.weight",
    ]
    for k in global_keys:
        if k in hf_sd:
            print(f"  {k:<60s} {_shape(hf_sd[k])}")

    # ---- ViT layers (first 2) ----
    print(f"\n[ViT blocks]  (showing 0 .. 1 of {args.num_vit_layers})")
    for i in range(min(2, args.num_vit_layers)):
        print(f"  -- block {i} --")
        for suffix in [
            "norm1.weight", "norm1.bias",
            "attn.qkv.weight", "attn.qkv.bias",
            "attn.proj.weight", "attn.proj.bias",
            "norm2.weight", "norm2.bias",
            "mlp.fc1.weight", "mlp.fc1.bias",
            "mlp.fc2.weight", "mlp.fc2.bias",
        ]:
            k = f"vision_model.blocks.{i}.{suffix}"
            if k in hf_sd:
                print(f"    {suffix:<40s} {_shape(hf_sd[k])}")

    # ---- LM layer 0: dense MLP ----
    print(f"\n[LM layer 0]  (dense MLP)")
    for suffix in [
        "input_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ]:
        k = f"model.layers.0.{suffix}"
        if k in hf_sd:
            print(f"  {suffix:<50s} {_shape(hf_sd[k])}")

    # ---- LM MoE layers (first 2 of layers 1+) ----
    moe_layers = [i for i in range(1, min(3, args.num_lm_layers))]
    print(f"\n[LM MoE layers]  (showing layers {moe_layers} of {list(range(1, args.num_lm_layers))})")
    for i in moe_layers:
        hg = f"model.layers.{i}"
        print(f"  -- layer {i} --")

        # attention (same as layer 0)
        for suffix in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
        ]:
            k = f"{hg}.{suffix}"
            if k in hf_sd:
                print(f"    {suffix:<50s} {_shape(hf_sd[k])}")

        # router & expert_bias
        for suffix in [
            "mlp.gate.weight", "mlp.gate.weight_1",
            "mlp.moe_statics.e_score_correction_bias",
        ]:
            k = f"{hg}.{suffix}"
            if k in hf_sd:
                print(f"    {suffix:<50s} {_shape(hf_sd[k])}")

        # shared experts
        for suffix in [
            "mlp.shared_experts.gate_proj.weight",
            "mlp.shared_experts.up_proj.weight",
            "mlp.shared_experts.down_proj.weight",
        ]:
            k = f"{hg}.{suffix}"
            if k in hf_sd:
                print(f"    {suffix:<50s} {_shape(hf_sd[k])}")

        # text experts: HF index 0..num_experts-1  → show first 2
        print(f"    [text experts 0..1 of {args.num_experts}]")
        for j in range(min(2, args.num_experts)):
            for w in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
                k = f"{hg}.mlp.experts.{j}.{w}"
                if k in hf_sd:
                    print(f"      experts.{j}.{w:<35s} {_shape(hf_sd[k])}")

        # vision experts: HF index num_experts..2*num_experts-1 → show first 2
        print(f"    [vision experts 0..1 of {args.num_experts}]")
        for j in range(min(2, args.num_experts)):
            hf_j = j + args.num_experts
            for w in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
                k = f"{hg}.mlp.experts.{hf_j}.{w}"
                if k in hf_sd:
                    print(f"      experts.{hf_j}.{w:<33s} {_shape(hf_sd[k])}")

    print("=" * W + "\n")


def print_mcore_summary(mcore: list, args) -> None:
    """Print a structured subset of MCore weights (shard [pp=0, tp=0]) for sanity-checking."""
    W = 80
    sd = mcore[0][0]["model"]

    print("\n" + "=" * W)
    print(f"  MCore checkpoint summary  (shard pp=0 tp=0)")
    print(f"  Total MCore keys : {sum(1 for k in sd if not k.endswith('extra_state'))}")
    print("=" * W)

    # helper: print a key if it exists in the mcore state dict
    def show(key):
        if key in sd:
            print(f"  {key:<70s} {_shape(sd[key])}")

    # ---- global ----
    print("\n[Global]")
    for k in [
        "encoder_model.text_encoder.word_embeddings.weight",
        "foundation_model.embedding.word_embeddings.weight",
        "foundation_model.output_layer.weight",
        "foundation_model.decoder.final_layernorm.weight",
        "encoder_model.image_encoder.patch_embed.proj.weight",
        "encoder_model.image_encoder.ln.weight",
        "encoder_model.image_encoder.ln.bias",
        "encoder_model.image_projector.mlp.weight",
        "encoder_model.image_projector.after_norm.weight",
    ]:
        show(k)

    # ---- ViT layers (first 2) ----
    print(f"\n[ViT decoder layers]  (showing 0..1 of {args.num_vit_layers})")
    for i in range(min(2, args.num_vit_layers)):
        mc = f"encoder_model.image_encoder.decoder.layers.{i}"
        print(f"  -- layer {i} --")
        for suffix in [
            "input_layernorm.weight", "input_layernorm.bias",
            "self_attention.linear_qkv.weight", "self_attention.linear_qkv.bias",
            "self_attention.linear_proj.weight", "self_attention.linear_proj.bias",
            "pre_mlp_layernorm.weight", "pre_mlp_layernorm.bias",
            "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
        ]:
            show(f"{mc}.{suffix}")

    # ---- LM layer 0: dense ----
    print(f"\n[LM layer 0]  (dense MLP)")
    mc0 = "foundation_model.decoder.layers.0"
    for suffix in [
        "input_layernorm.weight",
        "self_attention.linear_qkv.weight",
        "self_attention.linear_proj.weight",
        "pre_mlp_layernorm.weight",
        "mlp.linear_fc1.weight", "mlp.linear_fc1_1.weight", "mlp.linear_fc2.weight",
    ]:
        show(f"{mc0}.{suffix}")

    # ---- LM MoE layers (first 2 of layers 1+) ----
    moe_layers = [i for i in range(1, min(3, args.num_lm_layers))]
    print(f"\n[LM MoE layers]  (showing layers {moe_layers} of {list(range(1, args.num_lm_layers))})")
    for i in moe_layers:
        mc = f"foundation_model.decoder.layers.{i}"
        print(f"  -- layer {i} --")
        for suffix in [
            "input_layernorm.weight",
            "self_attention.linear_qkv.weight",
            "self_attention.linear_proj.weight",
            "pre_mlp_layernorm.weight",
            "mlp.text_moe_layer.router.weight",
            "mlp.text_moe_layer.router.expert_bias",
            "mlp.vision_moe_layer.router.weight",
            "mlp.vision_moe_layer.router.expert_bias",
            "mlp.shared_experts.linear_fc1.weight",
            "mlp.shared_experts.linear_fc1_1.weight",
            "mlp.shared_experts.linear_fc2.weight",
        ]:
            show(f"{mc}.{suffix}")

        print(f"    [text experts 0..1 of {args.num_experts}]")
        for j in range(min(2, args.num_experts)):
            for w in ["linear_fc1.weight", "linear_fc1_1.weight", "linear_fc2.weight"]:
                show(f"{mc}.mlp.text_moe_layer.experts.local_experts.{j}.{w}")

        print(f"    [vision experts 0..1 of {args.num_experts}]")
        for j in range(min(2, args.num_experts)):
            for w in ["linear_fc1.weight", "linear_fc1_1.weight", "linear_fc2.weight"]:
                show(f"{mc}.mlp.vision_moe_layer.experts.local_experts.{j}.{w}")

    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------

_QKV_ORDER = {"q_proj": 0, "k_proj": 1, "v_proj": 2}


def _shift_layer_idx(key: str, offset: int) -> str:
    """Shift the layer index in a foundation_model.decoder.layers.N key by *offset*.

    Used to convert a per-PP-stage local layer index back to the global layer
    index used in the HF → MCore weight mapping table.

    Example:
        _shift_layer_idx("foundation_model.decoder.layers.2.mlp.linear_fc1.weight", 4)
        → "foundation_model.decoder.layers.6.mlp.linear_fc1.weight"
    """
    if offset == 0:
        return key
    return re.sub(
        r"(foundation_model\.decoder\.layers\.)(\d+)",
        lambda m: f"{m.group(1)}{int(m.group(2)) + offset}",
        key,
    )


def _build_dst_to_ops(mappings: List[WeightMapping]) -> dict:
    """Group all TransformOps by dst_name, sorted q→k→v for merged QKV.

    Most dst keys map to exactly one op (direct copy).
    The only exception is linear_qkv.weight, where q/k/v share one dst
    and must arrive in q→k→v order for merge_qkv_lm_fn.
    """
    grouped = defaultdict(list)
    for m in mappings:
        for op in m.ops:
            grouped[op.dst_name].append(op)

    return {
        dst: sorted(ops, key=lambda op: _QKV_ORDER.get(op.src_name.split(".")[-2], 99))
        for dst, ops in grouped.items()
    }


def convert_hg2mcore(args) -> None:
    # ------------------------------------------------------------------
    # Parse / validate pp_layer_offsets
    # ------------------------------------------------------------------
    if args.pp_layer_offsets is None:
        if args.pp == 1:
            pp_offsets = [0]
        else:
            raise ValueError(
                "--pp_layer_offsets is required when --pp > 1.\n"
                "Provide a comma-separated list of global layer-index offsets, "
                "one per PP stage.\n"
                "Example: --pp 8 --pp_layer_offsets 0,4,8,12,16,20,24,26"
            )
    else:
        pp_offsets = [int(x) for x in args.pp_layer_offsets.split(",")]
        if len(pp_offsets) != args.pp:
            raise ValueError(
                f"--pp_layer_offsets has {len(pp_offsets)} entries "
                f"but --pp={args.pp}; lengths must match."
            )

    hf_sd   = load_huggingface_checkpoints(args.load_hg_path)
    mcore   = load_mcore_checkpoints(args.load_mcore_path, args.tp, args.pp)

    print_hf_summary(hf_sd, args)
    print_mcore_summary(mcore, args)

    mappings = [
        build_vision_mapping(args),
        build_adapter_mapping(args),
        build_language_mapping(args),
    ]
    dst_to_ops = _build_dst_to_ops(mappings)

    for p in range(args.pp):
        offset = pp_offsets[p]
        for t in range(args.tp):
            print(f"\n=== Converting shard [pp={p}, tp={t}]  (layer offset={offset}) ===")
            model_sd = mcore[p][t]["model"]
            for dst_key, original_tensor in model_sd.items():
                if dst_key.endswith("extra_state"):
                    continue

                # Translate local PP-stage layer index → global layer index
                # so it matches the keys in dst_to_ops built from the full
                # HF weight mapping.
                lookup_key = _shift_layer_idx(dst_key, offset)

                if lookup_key not in dst_to_ops:
                    raise KeyError(
                        f"No mapping found for MCore key: {dst_key!r}"
                        + (f" (shifted to: {lookup_key!r})" if offset else "")
                    )

                ops = dst_to_ops[lookup_key]
                converted = ops[0].forward_fn(ops, hf_sd, original_tensor.shape)

                assert converted.shape == original_tensor.shape, (
                    f"Shape mismatch for {dst_key!r}: "
                    f"expected {original_tensor.shape}, got {converted.shape}"
                )
                srcs = [op.src_name for op in ops]
                print(f"  {srcs} -> {dst_key}  {converted.shape}")
                model_sd[dst_key] = converted

    save_mcore_checkpoints(mcore, args.save_mcore_path, args.tp, args.pp)
    print(f"\nConversion done. Saved to: {args.save_mcore_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ERNIE-4.5-VL checkpoints from HuggingFace to Megatron-Core."
    )
    parser.add_argument("--load_hg_path",    required=True, help="HF checkpoint directory (input)")
    parser.add_argument("--load_mcore_path", required=True, help="MCore checkpoint directory (input, template)")
    parser.add_argument("--save_mcore_path", required=True, help="MCore checkpoint directory (output)")
    parser.add_argument("--tp",              type=int, required=True, help="Tensor parallel size")
    parser.add_argument("--pp",              type=int, required=True, help="Pipeline parallel size")
    parser.add_argument("--num_vit_layers",  type=int, required=True, help="Number of ViT layers")
    parser.add_argument("--num_lm_layers",   type=int, required=True, help="Number of LM layers")
    parser.add_argument("--num_experts",     type=int, default=64,    help="Experts per MoE layer")
    parser.add_argument(
        "--pp_layer_offsets",
        type=str,
        default=None,
        help=(
            "Comma-separated global layer-index offset for each PP stage, "
            "one value per stage. Required when --pp > 1. "
            "Each value is the index of the *first* LM layer assigned to that stage "
            "in the full HF model. "
            "Example: --pp 8 --pp_layer_offsets 0,4,8,12,16,20,24,26"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.load_mcore_path != args.save_mcore_path, "load/save mcore paths must differ"

    print(f"load_hg_path  : {args.load_hg_path}")
    print(f"save_mcore_path: {args.save_mcore_path}")
    print(f"tp={args.tp}  pp={args.pp}  vit_layers={args.num_vit_layers}  "
          f"lm_layers={args.num_lm_layers}  experts={args.num_experts}")

    convert_hg2mcore(args)
