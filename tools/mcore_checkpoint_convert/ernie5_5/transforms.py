# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Adapted from NVIDIA-NeMo/Megatron-Bridge at 8f580c64376877e9e2bc3140ce3476afc0f75d76.

"""ERNIE HF layouts: fused D-layer KV, RoPE row order and FP8/MXFP4 block scales.

The shared MXFP4 dequant path consumes ``<base>.weight`` + ``<base>.scale`` pairs,
while ERNIE-5.5-Flash-Preview ships the per-block E8M0 scale as
``<base>.weight_scale_inv``; ``normalize_ernie55_mxfp4_state_dict`` bridges the two.
"""

import torch

from mcore_checkpoint_convert.utils.utils import convert_fp8_to_bf16

_MXFP4_BLOCK_SIZE = 32
try:
    _MXFP4_SCALE_DTYPES = (torch.float8_e8m0fnu,)
except AttributeError:  # torch < 2.1 has no E8M0 dtype
    _MXFP4_SCALE_DTYPES = ()


def convert_ernie55_hf_tensors(state_dict, module, hf_config, to_hf=False):
    """Split/join D-layer KV and convert RoPE rows without checkpoint I/O."""
    if not to_hf:
        for name, value in list(state_dict.items()):
            if isinstance(value, torch.Tensor) and value.dtype == torch.float8_e4m3fn:
                scale_name = name + "_scale_inv"
                state_dict[name] = convert_fp8_to_bf16(value, state_dict.pop(scale_name))

    heads = module["d_num_attention_heads"]
    latent = module["d_kv_lora_rank"]
    nope = module["d_qk_head_dim"]
    value_dim = module["d_v_head_dim"]
    rope_dim = module["d_qk_pos_emb_head_dim"]
    adjacent = hf_config.get("mqa_latent_rope_adjacent_pairing", hf_config.get("rope_interleave", False))
    for layer_id, ratio in enumerate(module["compress_ratios"]):
        if ratio != -2:
            continue
        prefix = f"model.layers.{layer_id}.self_attn."
        kv_name = prefix + "kv_b_proj.weight"
        k_name, v_name = prefix + "k_b_proj", prefix + "v_b_proj"
        if to_hf:
            if k_name in state_dict or v_name in state_dict:
                key, value = state_dict.pop(k_name), state_dict.pop(v_name)
                if tuple(key.shape) != (heads * latent, nope):
                    raise ValueError(f"Invalid D-layer K projection shape: {key.shape}.")
                if tuple(value.shape) != (heads * value_dim, latent):
                    raise ValueError(f"Invalid D-layer V projection shape: {value.shape}.")
                key = key.reshape(heads, latent, nope).transpose(1, 2)
                value = value.reshape(heads, value_dim, latent)
                state_dict[kv_name] = torch.cat((key, value), dim=1).reshape(-1, latent).contiguous()
        elif kv_name in state_dict:
            kv = state_dict.pop(kv_name).reshape(heads, nope + value_dim, latent)
            state_dict[k_name] = kv[:, :nope].transpose(1, 2).reshape(heads * latent, nope).contiguous()
            state_dict[v_name] = kv[:, nope:].reshape(heads * value_dim, latent).contiguous()

        if adjacent:
            continue
        for leaf, offset in (("q_b_proj.weight", nope), ("kv_a_proj_with_mqa.weight", latent)):
            name = prefix + leaf
            if name in state_dict:
                state_dict[name] = _permute_rope_rows(state_dict[name], offset, rope_dim, inverse=to_hf)


def normalize_ernie55_mxfp4_state_dict(state_dict, c_config, args=None):
    """Rename Preview MXFP4 ``weight_scale_inv`` scales to the shared ``scale`` name.

    Only runs for ERNIE 5.5: the rename keys off 8-bit ``.weight`` tensors, which
    this converter only expects from Preview's MXFP4 export, and an already
    normalized pair must stay untouched so a repeated run cannot clobber it.
    This is the only stage that recognizes a Preview MXFP4 pair, so a missing
    ``--hf-dequantize-mxfp4`` is reported here; otherwise the still-packed int8
    expert weights would only fail later with an error unrelated to the missing
    flag. ``args=None`` keeps the rename-only behavior for callers that merely
    reshape keys.
    """
    module = c_config.get("module", {}) if c_config is not None else {}
    if module.get("model_type") != "ernie5_5":
        return
    scale_dtypes = (torch.uint8, torch.int8) + _MXFP4_SCALE_DTYPES
    for name, weight in list(state_dict.items()):
        if not name.endswith(".weight") or not isinstance(weight, torch.Tensor):
            continue
        if weight.dtype not in (torch.int8, torch.uint8):
            continue
        base = name[: -len(".weight")]
        scale_name = base + ".weight_scale_inv"
        scale = state_dict.get(scale_name)
        if not isinstance(scale, torch.Tensor) or scale.dtype not in scale_dtypes:
            continue
        if args is not None and not getattr(args, "hf_dequantize_mxfp4", False):
            raise ValueError(
                f"{name} and {scale_name} are an MXFP4 pair: ERNIE-5.5-Flash-Preview routed "
                f"experts are stored as int8-packed FP4 weights with E8M0 block scales and must "
                f"be materialized before conversion. Add --hf-dequantize-mxfp4 "
                f"--hf-dequantize-dtype bfloat16 to the command."
            )
        target_name = base + ".scale"
        if target_name in state_dict:
            continue
        # OCP MXFP4 packs two E2M1 values per byte and shares one E8M0 scale
        # per block, so rank and leading dims must match exactly and the last
        # scale dim must cover the unpacked width. Validate both before the
        # rename: a scalar scale would otherwise raise IndexError below instead
        # of a diagnostic naming the offending key.
        scale_matches_weight = (
            weight.dim() >= 1
            and scale.dim() == weight.dim()
            and tuple(scale.shape[:-1]) == tuple(weight.shape[:-1])
            and scale.shape[-1] * _MXFP4_BLOCK_SIZE == weight.shape[-1] * 2
        )
        if not scale_matches_weight:
            raise ValueError(
                f"MXFP4 pair for base '{base}' is inconsistent: scale {scale_name} has shape "
                f"{tuple(scale.shape)}, packed weight {name} has shape {tuple(weight.shape)}, "
                f"block size {_MXFP4_BLOCK_SIZE}."
            )
        state_dict[target_name] = state_dict.pop(scale_name)


def _permute_rope_rows(value, offset, rope_dim, inverse=False):
    if rope_dim % 2:
        raise ValueError(f"RoPE dimension must be even, got {rope_dim}.")
    block = offset + rope_dim
    if value.shape[0] % block:
        raise ValueError(f"RoPE rows {value.shape[0]} are not divisible by block {block}.")
    order = torch.arange(rope_dim, device=value.device).reshape(2, -1).T.flatten()
    if inverse:
        order = order.argsort()
    blocks = value.reshape(-1, block, *value.shape[1:]).clone()
    blocks[:, offset:] = blocks[:, offset:].index_select(1, order)
    return blocks.reshape_as(value)


__all__ = ["convert_ernie55_hf_tensors", "normalize_ernie55_mxfp4_state_dict"]
