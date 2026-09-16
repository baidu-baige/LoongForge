"""Vendor module implementation."""
from __future__ import annotations

import torch
from torch import nn


def _config_get(config, key, default=None):
    """Vendor interface."""
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class DinoVideoTeacher(nn.Module):
    """Vendor interface."""

    def __init__(
        self,
        ckpt_path: str,
        config_path: str | None = None,
        attention_mode: str = "flex_block_causal",
        input_size: int = 256,
        n_blocks: int = 1,
        device: str | torch.device = "cuda",
        cls_pool: str = "mean",
        effective_fps: float | None = None,
    ) -> None:
        """Vendor interface."""
        super().__init__()
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.attention_mode = attention_mode
        self.input_size = input_size
        self.n_blocks = n_blocks
        self.device = torch.device(device)
        self.cls_pool = cls_pool
        self.effective_fps = float(effective_fps) if effective_fps is not None else None
        if self.effective_fps is not None and self.effective_fps <= 0:
            raise ValueError(f"effective_fps must be positive, got {self.effective_fps}.")
        self.backbone = None
        self.adapter = None
        self.info = None

    def build(self):
        """Vendor interface."""
        from .lumos_dinov3.models.inference import CausalEvalAdapter
        from .lumos_dinov3.models.navit_eval import build_navit_for_eval

        backbone, info = build_navit_for_eval(
            self.ckpt_path,
            config_path=self.config_path,
            attention_mode=self.attention_mode,
            img_size=self.input_size,
            device=self.device,
        )
        adapter = CausalEvalAdapter(backbone, cls_pool=self.cls_pool)
        # Preserve RoPE buffer precision while casting weights to bf16.
        rope = backbone.rope_embed
        rope_buffers = {name: buf.detach().clone() for name, buf in rope.named_buffers()}
        adapter.to(device=self.device, dtype=torch.bfloat16)
        for name, buf in rope_buffers.items():
            rope.get_buffer(name).data = buf.to(device=self.device)
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad = False

        if self.attention_mode == "flex_block_causal":
            from .lumos_dinov3.models.navit_video_vision_transformer import _resolve_block
            for blk in backbone.blocks:
                _resolve_block(blk)._has_regional_compile = False
        self.backbone = backbone
        self.adapter = adapter
        self.info = info
        return self

    @torch.no_grad()
    def get_future_feature(
        self,
        video: torch.Tensor,
        return_cls: bool = False,
        return_current: bool = False,
        current_index: int = 0,
        fps: float | torch.Tensor | None = None,
    ):
        """Vendor interface."""
        if self.adapter is None:
            raise RuntimeError("DinoVideoTeacher must be built before feature extraction.")
        if video.ndim != 5:
            raise ValueError(f"expected video tensor [B,C,T,H,W], got {tuple(video.shape)}")
        if video.shape[2] < 2:
            raise ValueError("future-video distillation requires at least current and future frames.")

        video = video.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
        outputs = self.adapter.get_intermediate_layers(
            video,
            n=self.n_blocks,
            return_class_token=return_cls,
            return_frame_class_tokens=return_cls and return_current,
            norm=True,
            fps=self.effective_fps if fps is None else fps,
        )
        output = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
        if return_cls:
            if return_current:
                patches, cls_token, frame_cls_tokens = output
            else:
                patches, cls_token = output
        else:
            patches = output
        batch_size, _, frames, _, _ = video.shape
        if patches.shape[1] % frames != 0:
            raise RuntimeError(
                f"DINO patch token count {patches.shape[1]} is not divisible by T={frames}."
            )
        tokens_per_frame = patches.shape[1] // frames
        patches = patches.view(batch_size, frames, tokens_per_frame, patches.shape[-1])
        future_patches = patches[:, -1].detach().to(dtype=torch.bfloat16)
        if return_current:
            current_patches = patches[:, current_index].detach().to(dtype=torch.bfloat16)
        if return_cls:
            future_cls = cls_token.detach().to(dtype=torch.bfloat16)
            if return_current:
                if frame_cls_tokens.ndim != 3 or frame_cls_tokens.shape[1] != frames:
                    raise RuntimeError(
                        "DINO frame CLS tokens must have shape [B,T,D] when return_current=True."
                    )
                current_cls = frame_cls_tokens[:, current_index].detach().to(dtype=torch.bfloat16)
                future_cls = frame_cls_tokens[:, -1].detach().to(dtype=torch.bfloat16)
                return future_patches, future_cls, current_patches, current_cls
            return future_patches, future_cls
        if return_current:
            return future_patches, current_patches
        return future_patches


def build_dino_video_teacher(config) -> DinoVideoTeacher:
    """Vendor interface."""
    ckpt_path = _config_get(config, "ckpt_path")
    if not ckpt_path:
        raise ValueError("DINO video teacher requires video.ckpt_path.")

    teacher = DinoVideoTeacher(
        ckpt_path=ckpt_path,
        config_path=_config_get(config, "config_path", None),
        attention_mode=_config_get(config, "attention_mode", "flex_block_causal"),
        input_size=int(_config_get(config, "input_size", 256)),
        n_blocks=int(_config_get(config, "n_blocks", 1)),
        device=_config_get(config, "device", "cuda"),
        cls_pool=_config_get(config, "cls_pool", "mean"),
        effective_fps=_config_get(config, "effective_fps", _config_get(config, "fps", None)),
    )
    return teacher.build()
