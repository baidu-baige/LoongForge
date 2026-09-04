"""NaViT video backbone with packed-sequence attention."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .unified_video_vision_transformer import UnifiedVideoViT

logger = logging.getLogger("dinov3")


def _resolve_block(blk):
    """Return the underlying attention block."""
    inner = blk
    inner = getattr(inner, "_checkpoint_wrapped_module", inner)
    inner = getattr(inner, "_orig_mod", inner)
    return inner


def _resolve_intermediate_indices(return_intermediate, depth: int) -> Optional[List[int]]:
    """Vendor interface."""
    if return_intermediate is None:
        return None
    if isinstance(return_intermediate, int):
        if not (0 < return_intermediate <= depth):
            raise ValueError(f"return_intermediate must be in [1, {depth}], got {return_intermediate}")
        return list(range(depth - return_intermediate, depth))

    indices = [int(i) for i in return_intermediate]
    if not indices:
        raise ValueError("return_intermediate sequence must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"return_intermediate contains duplicate block indices: {indices}")
    if any(i < 0 or i >= depth for i in indices):
        raise ValueError(f"return_intermediate contains out-of-range block index for depth {depth}: {indices}")
    return sorted(indices)


@dataclass
class PackedOutput:
    """Output of NaviTVideoViT.forward_features_packed."""

    cls_tokens: Tensor  # (total_crops, D) — one CLS per crop (mean over T)
    patch_tokens: Tensor  # (total_patches, D) — all patches flat
    patch_cu_seqlens: Tensor  # (total_crops + 1,) — patch boundaries
    storage_tokens: Tensor  # (total_storage, D) — register tokens flat
    n_patches_per_crop: List[int]  # per-crop patch count
    intermediate_tokens: Optional[List[Tensor]] = None
    intermediate_indices: Optional[List[int]] = None
    shapes: Optional[List[Tuple[int, int, int]]] = None
    crops_per_shape: Optional[List[int]] = None


def _make_prefix_rope(
    n_prefix: int,
    sin_sp_frame: Tensor,
    cos_sp_frame: Tensor,
    t_mask: Tensor,
    prefix_temporal: bool,
) -> Tuple[Tensor, Tensor]:
    """Build per-frame RoPE for CLS/storage prefix tokens."""
    D = sin_sp_frame.shape[-1]
    sin_prefix = torch.zeros(n_prefix, D, dtype=sin_sp_frame.dtype, device=sin_sp_frame.device)
    cos_prefix = torch.ones(n_prefix, D, dtype=cos_sp_frame.dtype, device=cos_sp_frame.device)
    if prefix_temporal:
        sin_prefix[:, t_mask] = sin_sp_frame[0, t_mask]
        cos_prefix[:, t_mask] = cos_sp_frame[0, t_mask]
    return sin_prefix, cos_prefix


def _assemble_crops_with_prefix(
    sin_sp: Tensor,
    cos_sp: Tensor,
    T: int,
    patches_per_frame: int,
    n_prefix: int,
    t_mask: Tensor,
    prefix_temporal: bool,
) -> Tuple[Tensor, Tensor]:
    """Insert per-frame prefix RoPE into packed crop layout."""
    N = sin_sp.shape[0]
    D = sin_sp.shape[-1]
    P = patches_per_frame

    sin_btpd = sin_sp.view(N, T, P, D)
    cos_btpd = cos_sp.view(N, T, P, D)

    sin_pre = torch.zeros(N, T, n_prefix, D, dtype=sin_sp.dtype, device=sin_sp.device)
    cos_pre = torch.ones(N, T, n_prefix, D, dtype=cos_sp.dtype, device=cos_sp.device)
    if prefix_temporal:
        sin_pre[:, :, :, t_mask] = sin_btpd[:, :, 0:1, t_mask]
        cos_pre[:, :, :, t_mask] = cos_btpd[:, :, 0:1, t_mask]

    sin_frames = torch.cat([sin_pre, sin_btpd], dim=2)  # (N, T, n_prefix+P, D)
    cos_frames = torch.cat([cos_pre, cos_btpd], dim=2)
    frame_len = n_prefix + P
    sin_crop = sin_frames.reshape(N * T * frame_len, D)
    cos_crop = cos_frames.reshape(N * T * frame_len, D)
    return sin_crop, cos_crop


def _build_packed_rope(
    shapes: List[Tuple[int, int, int]],
    crops_per_shape: List[int],
    rope_embed,
    n_storage_tokens: int,
    device: torch.device,
    rope_3d: bool = False,
    fps_per_group: Optional[List[Optional[Tensor]]] = None,
    prefix_temporal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Construct per-token RoPE for a packed sequence."""
    sin_parts: List[Tensor] = []
    cos_parts: List[Tensor] = []
    n_prefix = 1 + n_storage_tokens

    for group_idx, (shape, n_crops) in enumerate(zip(shapes, crops_per_shape)):
        T, Hp, Wp = shape

        if rope_3d:
            group_fps = None
            if fps_per_group is not None and fps_per_group[group_idx] is not None:
                group_fps = fps_per_group[group_idx]  # (n_crops,) tensor

            if group_fps is None or torch.all(group_fps == group_fps[0]):
                fps_val = float(group_fps[0]) if group_fps is not None else None
                sin_sp, cos_sp = rope_embed(H=Hp, W=Wp, T=T, fps=fps_val)
                patches_per_frame = Hp * Wp
                t_mask = rope_embed.temporal_channel_mask

                sin_crop, cos_crop = _assemble_crops_with_prefix(
                    sin_sp.unsqueeze(0), cos_sp.unsqueeze(0), T,
                    patches_per_frame, n_prefix, t_mask, prefix_temporal,
                )
                sin_parts.append(sin_crop.repeat(n_crops, 1))
                cos_parts.append(cos_crop.repeat(n_crops, 1))
            else:
                t_mask = rope_embed.temporal_channel_mask
                patches_per_frame = Hp * Wp
                sin_list, cos_list = [], []
                for c in range(n_crops):
                    fps_val = float(group_fps[c])
                    sin_sp, cos_sp = rope_embed(H=Hp, W=Wp, T=T, fps=fps_val)
                    sin_list.append(sin_sp)
                    cos_list.append(cos_sp)
                sin_all = torch.stack(sin_list, dim=0)  # (n_crops, T*P, D)
                cos_all = torch.stack(cos_list, dim=0)
                sin_crop, cos_crop = _assemble_crops_with_prefix(
                    sin_all, cos_all, T, patches_per_frame, n_prefix,
                    t_mask, prefix_temporal,
                )
                sin_parts.append(sin_crop)
                cos_parts.append(cos_crop)
        else:
            sin_sp, cos_sp = rope_embed(H=Hp, W=Wp)
            D = sin_sp.shape[-1]

            sin_prefix = torch.zeros(n_prefix, D, dtype=sin_sp.dtype, device=device)
            cos_prefix = torch.ones(n_prefix, D, dtype=cos_sp.dtype, device=device)
            sin_frame = torch.cat([sin_prefix, sin_sp], dim=0)
            cos_frame = torch.cat([cos_prefix, cos_sp], dim=0)

            sin_crop = sin_frame.repeat(T, 1)
            cos_crop = cos_frame.repeat(T, 1)
            sin_parts.append(sin_crop.repeat(n_crops, 1))
            cos_parts.append(cos_crop.repeat(n_crops, 1))

    return torch.cat(sin_parts, dim=0), torch.cat(cos_parts, dim=0)


class NaviTVideoViT(UnifiedVideoViT):
    """UnifiedVideoViT with a packed-sequence forward path."""

    def forward(self, *args, packed_mode: bool = False, is_training: bool = False, **kwargs):
        """Dispatch to packed-mode forward when requested."""
        if packed_mode:
            return self.forward_features_packed(*args, **kwargs)
        return super().forward(*args, is_training=is_training, **kwargs)

    def forward_features_packed(
        self,
        crop_groups: List[Tensor],
        mask_groups: Optional[List[Optional[Tensor]]] = None,
        fps_groups: Optional[List[Optional[Tensor]]] = None,
        return_intermediate: Optional[int | Sequence[int]] = None,
    ) -> PackedOutput:
        """Pack crop groups into one sequence and run the transformer."""
        if mask_groups is None:
            mask_groups = [None] * len(crop_groups)

        per_crop_tokens: List[Tensor] = []
        shapes: List[Tuple[int, int, int]] = []
        crops_per_shape: List[int] = []
        n_patches_per_crop: List[int] = []
        n_prefix = 1 + self.n_storage_tokens

        for crops, masks in zip(crop_groups, mask_groups):
            tokens, shape = self.prepare_tokens_with_masks(crops, masks)
            T, Hp, Wp = shape
            n_patches = T * Hp * Wp
            B_i = tokens.shape[0]

            shapes.append(shape)
            crops_per_shape.append(B_i)
            for b in range(B_i):
                per_crop_tokens.append(tokens[b])  # (seq_len, D)
                n_patches_per_crop.append(n_patches)

        packed = torch.cat(per_crop_tokens, dim=0)  # (total_tokens, D)
        seq_lens = [t.shape[0] for t in per_crop_tokens]
        cu_seqlens = torch.zeros(
            len(seq_lens) + 1, dtype=torch.int32, device=packed.device
        )
        torch.cumsum(
            torch.tensor(seq_lens, dtype=torch.int32, device=packed.device),
            dim=0,
            out=cu_seqlens[1:],
        )
        max_seqlen = max(seq_lens)

        rope_sin, rope_cos = _build_packed_rope(
            shapes, crops_per_shape, self.rope_embed,
            self.n_storage_tokens, packed.device,
            rope_3d=getattr(self, "rope_3d", False),
            fps_per_group=fps_groups,
            prefix_temporal=getattr(self, "rope_prefix_temporal", False),
        )

        if self.attention_mode == "flex_block_causal":
            from ..layers.flex_attention_packed import (
                build_flex_block_causal_masks,
            )
            flex_extra = (
                shapes, crops_per_shape, n_prefix,
                build_flex_block_causal_masks(shapes, crops_per_shape, n_prefix, packed.device),
            )
        elif self.attention_mode == "sdpa_block_causal":
            from ..layers.sdpa_attention_packed import (
                build_sdpa_block_causal_masks,
            )
            flex_extra = (
                shapes, crops_per_shape, n_prefix,
                build_sdpa_block_causal_masks(
                    shapes, crops_per_shape, n_prefix, packed.device, packed.dtype
                ),
            )
        else:
            flex_extra = None
        packed_seq_info = (cu_seqlens, max_seqlen, rope_sin, rope_cos, flex_extra)
        for blk in self.blocks:
            _resolve_block(blk)._packed_seq_info = packed_seq_info
        intermediate_indices = _resolve_intermediate_indices(return_intermediate, len(self.blocks))
        intermediate_index_set = set(intermediate_indices or [])
        intermediate_tokens: Optional[List[Tensor]] = [] if intermediate_indices is not None else None
        for i, blk in enumerate(self.blocks):
            packed = blk(packed)
            if intermediate_tokens is not None and i in intermediate_index_set:
                intermediate_tokens.append(packed)

        total_crops = len(n_patches_per_crop)
        D = packed.shape[-1]
        cls_list: List[Tensor] = []
        storage_list: List[Tensor] = []
        patch_list: List[Tensor] = []

        offset = 0
        shape_idx = 0
        crop_in_shape = 0
        has_storage = self.n_storage_tokens > 0
        for crop_i in range(total_crops):
            while crop_in_shape >= crops_per_shape[shape_idx]:
                crop_in_shape = 0
                shape_idx += 1
            T, Hp, Wp = shapes[shape_idx]
            frame_size = n_prefix + Hp * Wp
            crop_len = T * frame_size

            crop_tokens = packed[offset : offset + crop_len].reshape(T, frame_size, D)
            cls_per_frame = crop_tokens[:, 0, :]  # (T, D)
            patches_per_frame = crop_tokens[:, n_prefix:, :]  # (T, Hp*Wp, D)

            cls_list.append(cls_per_frame.mean(dim=0))  # (D,)
            patch_list.append(patches_per_frame.reshape(-1, D))  # (T*Hp*Wp, D)

            if has_storage:
                storage_per_frame = crop_tokens[:, 1:n_prefix, :]  # (T, n_storage, D)
                storage_list.append(storage_per_frame.reshape(-1, D))  # (T*n_storage, D)

            offset += crop_len
            crop_in_shape += 1

        cls_normed = torch.stack(cls_list, dim=0)    # (total_crops, D), unnormed
        patches_normed = torch.cat(patch_list, dim=0)  # (total_patches, D), unnormed

        if self.n_storage_tokens > 0:
            storage_flat = torch.cat(storage_list, dim=0)
        else:
            storage_flat = torch.empty(0, D, dtype=patches_normed.dtype, device=patches_normed.device)

        patch_cu = torch.zeros(total_crops + 1, dtype=torch.int32, device=packed.device)
        torch.cumsum(
            torch.tensor(n_patches_per_crop, dtype=torch.int32, device=packed.device),
            dim=0,
            out=patch_cu[1:],
        )

        return PackedOutput(
            cls_tokens=cls_normed,
            patch_tokens=patches_normed,
            patch_cu_seqlens=patch_cu,
            storage_tokens=storage_flat,
            n_patches_per_crop=n_patches_per_crop,
            intermediate_tokens=intermediate_tokens,
            intermediate_indices=intermediate_indices,
            shapes=list(shapes) if intermediate_indices is not None else None,
            crops_per_shape=list(crops_per_shape) if intermediate_indices is not None else None,
        )


def navit_video_vit_small(patch_size=16, **kwargs):
    """Vendor interface."""
    return NaviTVideoViT(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, ffn_ratio=4, **kwargs)


def navit_video_vit_base(patch_size=16, **kwargs):
    """Vendor interface."""
    return NaviTVideoViT(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, ffn_ratio=4, **kwargs)


def navit_video_vit_large(patch_size=16, **kwargs):
    """Vendor interface."""
    return NaviTVideoViT(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4, **kwargs)


def navit_video_vit_giant2(patch_size=16, **kwargs):
    """Vendor interface."""
    return NaviTVideoViT(patch_size=patch_size, embed_dim=1536, depth=40, num_heads=24, ffn_ratio=4, **kwargs)
