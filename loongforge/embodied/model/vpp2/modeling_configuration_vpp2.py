# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Typed VPP2 configuration for action-stage and joint Video/Action training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _asset_path(*parts: str) -> str:
    """Return a portable default path rooted at ``VPP2_ASSETS_ROOT``."""
    root = os.environ.get("VPP2_ASSETS_ROOT", "/workspace/vpp2/vpp2_assets")
    return os.path.join(root, *parts)


@dataclass(frozen=True)
class VPP2ModelConfig:
    """Architecture and checkpoint knobs for the pinned BASE implementation.

    The nested dictionaries intentionally mirror BASE's Hydra config.  They are
    kept as plain dictionaries because the transplanted Wan modules consume the
    same keyword names directly.
    """

    model_type: str = "vpp2"
    training_stage: str = "action"
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
    # Architecture and temporal contract shared with the data pipeline.
    action_dim: int = 7
    action_horizon: int = 32
    proprio_dim: int | None = 8
    action_chunk_size: int = 1
    action_visible_video_frames: int = 3
    action_only_first_frame_fast_path: bool = True
    mot_checkpoint_mixed_attn: bool = True
    # Reproduce fresh checkpoint-excluded projections and the training RNG stream.
    initialization_seed: int | None = None
    disable_train_autocast: bool = False
    # Default-off: omit a structurally all-True mask in the frozen video path.
    perf_video_attention_mask_elision: bool = False
    # Direct counterparts of the community environment switches.  The
    # existing mask-elision/graph switches remain separate so their measured
    # LoongForge implementations are preserved.
    perf_video_prefill_unmasked_sdpa: bool = False
    perf_action_visible_kv_sdpa: bool = False
    perf_video_rope_fused: bool = False
    # The accepted implementation uses eager frozen-video prefill.
    perf_video_execution: str = "eager"
    perf_action_execution: str = "eager"
    # Additional frozen encoders; preserve dtype, causal state and parameters.
    perf_compile_clip: str = "off"
    perf_compile_vae_encode: str = "off"
    # Compact only keys excluded by the standard action mask, inside the
    # existing full ActionDiT graph. The video producer still runs every layer.
    perf_action_graph_visible_kv: bool = False
    # Explicit startup warmup, before the trainer's step1 timer (LIBERO B2).
    perf_warmup_before_training: bool = False
    # Default-off: omit redundant empty fills in the serial frozen producer.
    perf_video_skip_uninitialized_fill: bool = False
    # Default-off: native BF16 LayerNorm input/output, FP32 native statistics.
    perf_video_native_layer_norm: bool = False
    # Default-off RMS pointwise fusion with original native FP32 mean/rsqrt.
    perf_video_rms_pointwise_fused: bool = False
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
        for name in ("perf_compile_clip", "perf_compile_vae_encode"):
            if getattr(self, name) not in {"off", "native_graph"}:
                raise ValueError(f"{name} must be off or native_graph")
        if (self.perf_compile_clip != "off" or self.perf_compile_vae_encode != "off") and (
            self.training_stage != "action" or self.init_on_meta
        ):
            raise ValueError("Frozen encoder compilation requires materialized action-stage training")
        if self.perf_action_graph_visible_kv and (
            self.training_stage != "action" or self.perf_action_execution != "native_graph"
            or self.action_chunk_size != 1
        ):
            raise ValueError("Graph K/V compaction requires the standard full action graph")
        if self.perf_action_visible_kv_sdpa and (
            self.training_stage != "action" or self.action_chunk_size != 1
            or self.perf_action_execution not in {"eager", "native_graph"}
        ):
            raise ValueError("Action visible K/V requires the standard action-only direct-K/V path")
        if self.perf_video_execution != "eager":
            raise ValueError("Only eager frozen-video execution is supported")
        if self.perf_action_execution not in {"eager", "native_graph"}:
            raise ValueError("Unsupported action execution mode")
        if self.perf_video_attention_mask_elision and (
            self.training_stage != "action" or self.dtype != "bfloat16"
            or self.perf_action_execution != "native_graph"
            or not self.perf_video_rope_fused
            or not self.perf_video_native_layer_norm
            or not self.perf_video_rms_pointwise_fused
            or self.perf_video_skip_uninitialized_fill is False
            or self.action_chunk_size != 1
            or int(self.video_dit_config.get("num_layers", 0)) != 40
            or self.video_dit_config.get("video_attention_mask_mode") not in {"bidirectional", "three_frame"}
        ):
            raise ValueError("Mask elision requires the validated frozen BF16 native-LN/RMS action graph path")
        if self.perf_video_rms_pointwise_fused and (
            not self.perf_video_native_layer_norm
            or self.training_stage != "action" or self.dtype != "bfloat16"
            or self.perf_action_execution not in {"eager", "native_graph"}
            or not self.perf_video_rope_fused
            or self.action_chunk_size != 1
            or int(self.video_dit_config.get("num_layers", 0)) != 40
        ):
            raise ValueError("Video RMS fusion requires the validated frozen BF16 native-LN path")
        if self.perf_video_native_layer_norm and (
            self.training_stage != "action" or self.dtype != "bfloat16"
            or self.perf_action_execution not in {"eager", "native_graph"}
            or not self.perf_video_rope_fused
            or self.action_chunk_size != 1
            or int(self.video_dit_config.get("num_layers", 0)) != 40
        ):
            raise ValueError("Native video LayerNorm requires the validated frozen BF16 action path")
        if self.perf_video_skip_uninitialized_fill and (
            self.training_stage != "action" or self.perf_action_execution not in {"eager", "native_graph"}
            or not self.perf_video_rope_fused or self.dtype != "bfloat16"
        ):
            raise ValueError("Scoped allocation requires the frozen BF16 action path")
        if self.model_type != "vpp2":
            raise ValueError(f"VPP2ModelConfig.model_type must be 'vpp2', got {self.model_type!r}")
        if self.training_stage not in {"action", "joint"}:
            raise ValueError("VPP2 training_stage must be action or joint")
        if self.training_stage == "joint" and self.action_only_first_frame_fast_path:
            raise ValueError("Joint training requires action_only_first_frame_fast_path=False")
        if self.action_dim != int(self.action_dit_config["action_dim"]):
            raise ValueError("action_dim must match action_dit_config.action_dim")
        if self.proprio_dim is not None and self.proprio_dim <= 0:
            raise ValueError("proprio_dim must be positive or None")
        if self.action_chunk_size <= 0 or self.action_horizon % self.action_chunk_size:
            raise ValueError("action_horizon must be divisible by action_chunk_size")
        if self.action_visible_video_frames <= 0:
            raise ValueError("action_visible_video_frames must be positive")
