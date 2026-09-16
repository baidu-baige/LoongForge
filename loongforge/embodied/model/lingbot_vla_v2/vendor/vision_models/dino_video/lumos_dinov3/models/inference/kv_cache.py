"""Vendor module implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ..navit_video_vision_transformer import _make_prefix_rope


def validate_cls_pool(cls_pool: str) -> str:
    """Vendor interface."""
    if cls_pool not in {"mean", "last"}:
        raise ValueError(f"cls_pool must be 'mean' or 'last', got {cls_pool!r}")
    return cls_pool


BLOCK_CAUSAL_ATTENTION_MODES = frozenset(
    {"flex_block_causal", "sdpa_block_causal"}
)


def validate_block_causal_backbone(backbone) -> None:
    """Reject backbones whose attention is not frame-level causal."""
    mode = getattr(backbone, "attention_mode", None)
    if mode not in BLOCK_CAUSAL_ATTENTION_MODES:
        raise ValueError(
            f"causal inference requires a block-causal attention_mode "
            f"(one of {sorted(BLOCK_CAUSAL_ATTENTION_MODES)}), but the backbone has "
            f"attention_mode={mode!r}, which runs bidirectional attention. "
            f"Rebuild the backbone with a block-causal attention_mode."
        )


def resolve_block_indices(n: int | Sequence[int], depth: int) -> list[int]:
    """Vendor interface."""
    if isinstance(n, int):
        if not (0 < n <= depth):
            raise ValueError(f"n must be in [1, {depth}], got {n}")
        return list(range(depth - n, depth))

    indices = [int(i) for i in n]
    if not indices:
        raise ValueError("n sequence must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"n contains duplicate block indices: {indices}")
    if any(i < 0 or i >= depth for i in indices):
        raise ValueError(f"n contains out-of-range block index for depth {depth}: {indices}")
    return sorted(indices)


@dataclass
class LayerKVCache:
    """Vendor interface."""
    k: Tensor | None = None
    v: Tensor | None = None

    def append(self, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Vendor interface."""
        if k_new.ndim != 4 or v_new.ndim != 4:
            raise ValueError("KV tensors must have shape (B, H, L, D)")
        if k_new.shape != v_new.shape:
            raise ValueError(f"K/V shapes differ: {tuple(k_new.shape)} vs {tuple(v_new.shape)}")

        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            if self.k.shape[:2] != k_new.shape[:2] or self.k.shape[-1] != k_new.shape[-1]:
                raise ValueError(
                    f"New KV tensor {tuple(k_new.shape)} is incompatible with cache {tuple(self.k.shape)}"
                )
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

    def clear(self) -> None:
        """Vendor interface."""
        self.k = None
        self.v = None


def single_frame_rope(
    backbone,
    *,
    hp: int,
    wp: int,
    frame_idx: int,
    fps: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Vendor interface."""
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be non-negative, got {frame_idx}")

    n_prefix = 1 + backbone.n_storage_tokens
    if getattr(backbone, "rope_3d", False):
        sin_sp, cos_sp = backbone.rope_embed(H=hp, W=wp, T=frame_idx + 1, fps=fps)
        patches_per_frame = hp * wp
        head_dim = sin_sp.shape[-1]
        sin_frame_sp = sin_sp.view(frame_idx + 1, patches_per_frame, head_dim)[frame_idx]
        cos_frame_sp = cos_sp.view(frame_idx + 1, patches_per_frame, head_dim)[frame_idx]
        sin_prefix, cos_prefix = _make_prefix_rope(
            n_prefix,
            sin_frame_sp,
            cos_frame_sp,
            backbone.rope_embed.temporal_channel_mask,
            getattr(backbone, "rope_prefix_temporal", False),
        )
        return torch.cat([sin_prefix, sin_frame_sp], dim=0), torch.cat([cos_prefix, cos_frame_sp], dim=0)

    sin_sp, cos_sp = backbone.rope_embed(H=hp, W=wp)
    head_dim = sin_sp.shape[-1]
    sin_prefix = torch.zeros(n_prefix, head_dim, dtype=sin_sp.dtype, device=sin_sp.device)
    cos_prefix = torch.ones(n_prefix, head_dim, dtype=cos_sp.dtype, device=cos_sp.device)
    return torch.cat([sin_prefix, sin_sp], dim=0), torch.cat([cos_prefix, cos_sp], dim=0)
