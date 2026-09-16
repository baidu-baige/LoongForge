"""SDPA block-causal backend for packed NaViT sequences."""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

_mask_cache: Dict[Tuple[int, int, int, int, torch.dtype], Tensor] = {}


def _get_or_create_mask(
    T: int,
    frame_size: int,
    crop_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return an additive frame-causal mask."""
    key = (T, frame_size, crop_len, device.index if device.index is not None else 0, dtype)
    if key not in _mask_cache:
        idx = torch.arange(crop_len, device=device)
        q_frame = (idx // frame_size).view(crop_len, 1)
        kv_frame = (idx // frame_size).view(1, crop_len)
        allow = kv_frame <= q_frame
        mask = torch.zeros(crop_len, crop_len, dtype=dtype, device=device)
        mask.masked_fill_(~allow, float("-inf"))
        _mask_cache[key] = mask
    return _mask_cache[key]


def build_sdpa_block_causal_masks(
    shapes: List[Tuple[int, int, int]],
    crops_per_shape: List[int],
    n_prefix: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tensor]:
    """Pre-build one additive mask per shape group."""
    masks = []
    for (T, Hp, Wp) in shapes:
        frame_size = n_prefix + Hp * Wp
        crop_len = T * frame_size
        masks.append(_get_or_create_mask(T, frame_size, crop_len, device, dtype))
    return masks


def sdpa_attention_block_causal(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    shapes: List[Tuple[int, int, int]],
    crops_per_shape: List[int],
    n_prefix: int,
    masks: List[Tensor],
) -> Tensor:
    """Run block-causal SDPA over packed shape groups."""
    H, D = q.shape[1], q.shape[2]

    group_token_counts = []
    for (T, Hp, Wp), n_crops in zip(shapes, crops_per_shape):
        frame_size = n_prefix + Hp * Wp
        crop_len = T * frame_size
        group_token_counts.append(n_crops * crop_len)

    q_groups = q.split(group_token_counts, dim=0)
    k_groups = k.split(group_token_counts, dim=0)
    v_groups = v.split(group_token_counts, dim=0)

    out_groups = []
    for group_idx, ((T, Hp, Wp), B_i) in enumerate(zip(shapes, crops_per_shape)):
        frame_size = n_prefix + Hp * Wp
        crop_len = T * frame_size

        q_g = q_groups[group_idx].reshape(B_i, crop_len, H, D).transpose(1, 2)
        k_g = k_groups[group_idx].reshape(B_i, crop_len, H, D).transpose(1, 2)
        v_g = v_groups[group_idx].reshape(B_i, crop_len, H, D).transpose(1, 2)

        out_g = F.scaled_dot_product_attention(q_g, k_g, v_g, attn_mask=masks[group_idx])

        out_g = out_g.transpose(1, 2).reshape(B_i * crop_len, H, D)
        out_groups.append(out_g)

    return torch.cat(out_groups, dim=0)
