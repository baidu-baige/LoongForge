"""FlexAttention block-causal backend for packed NaViT sequences."""

from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

if torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled():
    torch.use_deterministic_algorithms(True, warn_only=True)

_flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
_flex_attention_eager = flex_attention

_block_mask_cache: Dict[Tuple[int, int, int, int], BlockMask] = {}


def _get_or_create_block_mask(
    T: int,
    frame_size: int,
    crop_len: int,
    B: int,
    device: torch.device,
) -> BlockMask:
    """Get cached BlockMask or create one for given (T, frame_size, B)."""
    key = (T, frame_size, B, device.index if device.index is not None else 0)
    if key not in _block_mask_cache:
        def mask_mod(b, h, q_idx, kv_idx):
            return (kv_idx // frame_size) <= (q_idx // frame_size)

        _block_mask_cache[key] = create_block_mask(
            mask_mod,
            B=B,
            H=None,
            Q_LEN=crop_len,
            KV_LEN=crop_len,
            device=device,
            BLOCK_SIZE=128,
        )
    return _block_mask_cache[key]


def build_flex_block_causal_masks(
    shapes: List[Tuple[int, int, int]],
    crops_per_shape: List[int],
    n_prefix: int,
    device: torch.device,
) -> List[BlockMask]:
    """Pre-build one BlockMask per shape group."""
    masks = []
    for (T, Hp, Wp), B_i in zip(shapes, crops_per_shape):
        frame_size = n_prefix + Hp * Wp
        crop_len = T * frame_size
        masks.append(_get_or_create_block_mask(T, frame_size, crop_len, B_i, device))
    return masks


def flex_attention_block_causal(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    shapes: List[Tuple[int, int, int]],
    crops_per_shape: List[int],
    n_prefix: int,
    block_masks: List[BlockMask],
    has_regional_compile: bool = True,
) -> Tensor:
    """Run block-causal FlexAttention over packed shape groups."""
    H, D = q.shape[1], q.shape[2]
    flex_fn = _flex_attention_eager if has_regional_compile else _flex_attention_compiled

    flex_kernel_options = None if not has_regional_compile else {"BACKEND": "FLASH"}

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

        out_g = flex_fn(
            q_g, k_g, v_g, block_mask=block_masks[group_idx],
            kernel_options=flex_kernel_options,
        )

        out_g = out_g.transpose(1, 2).reshape(B_i * crop_len, H, D)
        out_groups.append(out_g)

    return torch.cat(out_groups, dim=0)
