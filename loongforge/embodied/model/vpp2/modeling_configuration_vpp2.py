# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration for the VPP2 LIBERO action-stage model."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _asset_path(*parts: str) -> str:
    """Return a portable default path rooted at ``VPP2_ASSETS_ROOT``."""
    root = os.environ.get("VPP2_ASSETS_ROOT", "./vpp2_assets")
    return os.path.join(root, *parts)


@dataclass(frozen=True)
class VPP2ModelConfig:
    """Architecture and checkpoint knobs for the pinned BASE implementation.

    The nested dictionaries intentionally mirror BASE's Hydra config.  They are
    kept as plain dictionaries because the transplanted Wan modules consume the
    same keyword names directly.
    """

    model_type: str = "vpp2"
    model_id: str = field(default_factory=lambda: _asset_path("Wan2.1-I2V-14B-480P"))
    video_checkpoint_path: str = field(
        default_factory=lambda: _asset_path("libero", "video_step010000.pt")
    )
    action_dit_pretrained_path: str = field(
        default_factory=lambda: _asset_path("libero", "action_init.pt")
    )
    dtype: str = "bfloat16"
    load_text_encoder: bool = False
    load_clip_encoder: bool = True
    tokenizer_model_id: str | None = None
    tokenizer_max_len: int = 512
    # Action-stage contract shared with the data pipeline.
    action_dim: int = 7
    action_horizon: int = 32
    proprio_dim: int | None = 8
    action_chunk_size: int = 1
    action_visible_video_frames: int = 3
    action_only_first_frame_fast_path: bool = True
    mot_checkpoint_mixed_attn: bool = True
    init_on_meta: bool = True
    video_dit_config: dict[str, Any] = field(default_factory=lambda: {
        "model_type": "i2v",
        "patch_size": [1, 2, 2],
        "text_len": 512,
        "in_dim": 36,
        "dim": 5120,
        "hidden_dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 40,
        "attn_head_dim": 128,
        "num_layers": 40,
        "eps": 1.0e-6,
        "qk_norm": True,
        "cross_attn_norm": True,
        "use_gradient_checkpointing": True,
        "video_attention_mask_mode": "bidirectional",
    })
    action_dit_config: dict[str, Any] = field(default_factory=lambda: {
        "action_dim": 7,
        "hidden_dim": 512,
        "ffn_dim": 2048,
        "num_layers": 40,
        "num_heads": 40,
        "attn_head_dim": 128,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1.0e-6,
        "use_gradient_checkpointing": True,
    })
    video_scheduler: dict[str, Any] = field(default_factory=lambda: {
        "train_sampling_strategy": "pure_noise",
        "train_shift": 5.0,
        "infer_shift": 3.0,
        "num_train_timesteps": 1000,
        "beta_alpha": 7.0,
        "beta_beta": 1.0,
    })
    action_scheduler: dict[str, Any] = field(default_factory=lambda: {
        "train_shift": 5.0,
        "infer_shift": 5.0,
        "num_train_timesteps": 1000,
    })
    loss: dict[str, Any] = field(default_factory=lambda: {
        "lambda_video": 1.0,
        "lambda_action": 1.0,
    })

    def __post_init__(self) -> None:
        if self.model_type != "vpp2":
            raise ValueError(f"VPP2ModelConfig.model_type must be 'vpp2', got {self.model_type!r}")
        if self.action_dim != int(self.action_dit_config["action_dim"]):
            raise ValueError("action_dim must match action_dit_config.action_dim")
        if self.proprio_dim is not None and self.proprio_dim <= 0:
            raise ValueError("proprio_dim must be positive or None")
        if self.action_chunk_size <= 0 or self.action_horizon % self.action_chunk_size:
            raise ValueError("action_horizon must be divisible by action_chunk_size")
        if self.action_visible_video_frames <= 0:
            raise ValueError("action_visible_video_frames must be positive")
