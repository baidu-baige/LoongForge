# Copyright 2026 The LoongForge Authors.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA-NeMo/Megatron-Bridge at 8f580c64376877e9e2bc3140ce3476afc0f75d76.

"""ERNIE HF layouts: fused D-layer KV, RoPE row order and FP8 block scales."""

import torch

from mcore_checkpoint_convert.utils.utils import convert_fp8_to_bf16


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
    adjacent = hf_config.get(
        "mqa_latent_rope_adjacent_pairing", hf_config.get("rope_interleave", False)
    )
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
                state_dict[name] = _permute_rope_rows(
                    state_dict[name], offset, rope_dim, inverse=to_hf
                )


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


__all__ = ["convert_ernie55_hf_tensors"]
