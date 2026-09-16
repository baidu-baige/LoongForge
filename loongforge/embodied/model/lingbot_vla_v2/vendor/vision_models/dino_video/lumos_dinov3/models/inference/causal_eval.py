"""Vendor module implementation."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .kv_cache import (
    resolve_block_indices,
    validate_block_causal_backbone,
    validate_cls_pool,
)


class CausalEvalAdapter(nn.Module):
    """Vendor interface."""

    def __init__(self, backbone, *, cls_pool: str = "mean") -> None:
        """Vendor interface."""
        super().__init__()
        validate_block_causal_backbone(backbone)
        self.backbone = backbone
        self.cls_pool = validate_cls_pool(cls_pool)

    def _cls_norm(self):
        """Vendor interface."""
        if getattr(self.backbone, "untie_cls_and_patch_norms", False):
            return self.backbone.cls_norm
        return self.backbone.norm

    def _pool_raw_cls(self, raw_cls: Tensor) -> Tensor:
        """Vendor interface."""
        if self.cls_pool == "mean":
            return raw_cls.mean(dim=1)
        return raw_cls[:, -1]

    @torch.no_grad()
    def get_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        return_frame_class_tokens: bool = False,
        norm: bool = True,
        fps: float | Tensor | None = None,
    ) -> tuple:
        """Vendor interface."""
        if return_frame_class_tokens and not return_class_token:
            raise ValueError("return_frame_class_tokens requires return_class_token=True")
        if self.backbone.training:
            raise RuntimeError("CausalEvalAdapter requires backbone.eval()")
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise ValueError(f"expected x with shape (B,C,T,H,W) or (B,C,H,W), got {tuple(x.shape)}")

        blocks_to_take = resolve_block_indices(n, len(self.backbone.blocks))
        fps_groups = None
        if fps is not None:
            if isinstance(fps, Tensor):
                fps_tensor = fps.to(device=x.device, dtype=torch.float32)
                if fps_tensor.ndim == 0:
                    fps_tensor = fps_tensor.expand(x.shape[0])
                elif fps_tensor.shape != (x.shape[0],):
                    raise ValueError(
                        f"fps tensor must be scalar or shape ({x.shape[0]},), got {tuple(fps_tensor.shape)}")
            else:
                fps_tensor = torch.full((x.shape[0],), float(fps), device=x.device, dtype=torch.float32)
            fps_groups = [fps_tensor]
        packed = self.backbone.forward_features_packed(
            [x],
            fps_groups=fps_groups,
            return_intermediate=blocks_to_take,
        )
        if packed.intermediate_tokens is None or packed.shapes is None or packed.crops_per_shape is None:
            raise RuntimeError("forward_features_packed did not return intermediate metadata")
        if len(packed.shapes) != 1 or len(packed.crops_per_shape) != 1:
            raise RuntimeError("CausalEvalAdapter expects a single packed crop group")

        bsz, _, frames, _, _ = x.shape
        t, hp, wp = packed.shapes[0]
        crops_per_shape = packed.crops_per_shape[0]
        if crops_per_shape != bsz:
            raise RuntimeError(f"expected one crop group with {bsz} crops, got {crops_per_shape}")
        if t != frames:
            raise RuntimeError(f"packed shape T={t} differs from input T={frames}")

        n_prefix = 1 + self.backbone.n_storage_tokens
        frame_size = n_prefix + hp * wp
        cls_norm = self._cls_norm()
        patch_outputs = []
        class_tokens = []
        frame_class_tokens = []
        extra_tokens = []

        for raw in packed.intermediate_tokens:
            raw_frames = raw.reshape(bsz, t, frame_size, -1)
            dim = raw_frames.shape[-1]
            raw_cls = raw_frames[:, :, 0, :]
            raw_storage = raw_frames[:, :, 1:n_prefix, :]
            raw_patches = raw_frames[:, :, n_prefix:, :]

            pooled_cls = self._pool_raw_cls(raw_cls)
            per_frame_cls = raw_cls
            storage = raw_storage.reshape(bsz, t * self.backbone.n_storage_tokens, dim)
            patches = raw_patches.reshape(bsz, t * hp * wp, dim)

            if norm:
                pooled_cls = cls_norm(pooled_cls)
                per_frame_cls = cls_norm(per_frame_cls)
                if self.backbone.n_storage_tokens > 0:
                    storage = cls_norm(storage)
                patches = self.backbone.norm(patches)

            patch_outputs.append(patches)
            class_tokens.append(pooled_cls)
            frame_class_tokens.append(per_frame_cls)
            extra_tokens.append(storage)

        if reshape and t == 1:
            patch_outputs = [
                p.reshape(p.shape[0], hp, wp, -1).permute(0, 3, 1, 2).contiguous()
                for p in patch_outputs
            ]

        if not return_class_token and not return_extra_tokens:
            return tuple(patch_outputs)
        if return_class_token and not return_extra_tokens:
            if return_frame_class_tokens:
                return tuple(zip(patch_outputs, class_tokens, frame_class_tokens))
            return tuple(zip(patch_outputs, class_tokens))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(patch_outputs, extra_tokens))
        if return_frame_class_tokens:
            return tuple(zip(patch_outputs, class_tokens, extra_tokens, frame_class_tokens))
        return tuple(zip(patch_outputs, class_tokens, extra_tokens))
