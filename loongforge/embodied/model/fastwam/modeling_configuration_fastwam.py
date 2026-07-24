# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM policy configuration for LoongForge.

YAML / config relationship
--------------------------
- The YAML file (``configs/models/embodied/fastwam.yaml``, ``model:`` section) is the
  user-facing knob: edit it to override any field for a specific run.
- This dataclass provides all defaults and is the single source of truth.
  At startup, OmegaConf merges the YAML ``model:`` section on top of a structured
  default built from this class, then materialises the result into a frozen instance.

Usage rules (must follow)
-------------------------
1. Always read fields via direct attribute access: ``model_cfg.action_dim``.
2. Never use ``getattr(cfg, "x", default)`` or ``cfg.get("x", default)``:
   - a default supplied there creates a second source of truth and hides the real one;
   - a misspelled field should raise ``AttributeError`` immediately, not silently
     return a fallback.
3. To add or change a model-structure parameter, edit only this dataclass
   (one authoritative definition).

Shared fields
-------------
Fields used by both model and data pipeline (``action_dim``, ``action_horizon``,
``proprio_dim``, etc.) are defined here once. ``FastWAMDataConfig`` does not
duplicate them; the data side reads them from the ``model_cfg`` instance
passed alongside.

Nested architecture configs
---------------------------
``video_dit_config``, ``action_dit_config``, ``video_scheduler``,
``action_scheduler``, and ``loss`` carry fixed architecture params for the
Wan2.2-5B backbone.  They are **not** exposed in the YAML; override per-key
after construction when needed for ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FastWAMModelConfig:
    """FastWAM model-structure config (maps 1:1 to YAML ``model:`` section)."""

    # ── Identity ──────────────────────────────────────────────────────────────
    model_type: str = "fastwam"
    variant: str = "joint"          # base | uncond | joint | idm

    # ── Model IDs ─────────────────────────────────────────────────────────────
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"

    # ── Task dimensions (shared with data side) ───────────────────────────────
    action_dim: int = 7
    action_horizon: int = 16
    proprio_dim: int | None = 8
    max_action_dim: int | None = None

    # ── Training knobs ────────────────────────────────────────────────────────
    tokenizer_max_len: int = 128
    load_text_encoder: bool = False
    mot_checkpoint_mixed_attn: bool = True
    skip_dit_load_from_pretrain: bool = False
    action_dit_pretrained_path: str | None = (
        "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    )
    redirect_common_files: bool = True
    dtype: str = "bfloat16"

    # ── Nested architecture configs (fixed for Wan2.2-5B, not in YAML) ────────
    video_dit_config: dict[str, Any] = field(default_factory=lambda: {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48, "out_dim": 48,
        "hidden_dim": 3072, "ffn_dim": 14336,
        "freq_dim": 256, "text_dim": 4096,
        "num_heads": 24, "attn_head_dim": 128, "num_layers": 30,
        "eps": 1.0e-6, "seperated_timestep": True,
        "require_clip_embedding": False, "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": True,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": 7,
        "action_group_causal_mask_mode": "group_diagonal",
    })
    action_dit_config: dict[str, Any] = field(default_factory=lambda: {
        "action_dim": 7,
        "hidden_dim": 1024, "ffn_dim": 4096,
        "num_heads": 24, "attn_head_dim": 128, "num_layers": 30,
        "text_dim": 4096, "freq_dim": 256, "eps": 1.0e-6,
        "use_gradient_checkpointing": True,
    })
    video_scheduler: dict[str, Any] = field(default_factory=lambda: {
        "train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000,
    })
    action_scheduler: dict[str, Any] = field(default_factory=lambda: {
        "train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000,
    })
    loss: dict[str, Any] = field(default_factory=lambda: {
        "lambda_video": 1.0, "lambda_action": 1.0,
    })

    def __post_init__(self) -> None:
        # ── Validate variant ──────────────────────────────────────────────────
        valid_variants = {"base", "uncond", "joint", "idm"}
        if self.variant not in valid_variants:
            raise ValueError(
                f"FastWAMModelConfig.variant must be one of {sorted(valid_variants)}, "
                f"got {self.variant!r}"
            )

        # ── Validate num_video_frames constraint via data config ───────────────
        # (num_video_frames lives in DataConfig; validation happens there)

        # ── Validate action_scheduler has required keys ────────────────────────
        required = {"train_shift", "infer_shift", "num_train_timesteps"}
        missing = required - set(self.action_scheduler.keys())
        if missing:
            raise ValueError(
                f"action_scheduler missing required keys: {sorted(missing)}"
            )

        # ── Sync mot_checkpoint_mixed_attn → nested dit configs ───────────────
        # frozen=True prevents direct assignment; use object.__setattr__ on the
        # mutable dicts themselves (the dict objects are not frozen).
        self.video_dit_config["use_gradient_checkpointing"] = self.mot_checkpoint_mixed_attn
        self.action_dit_config["use_gradient_checkpointing"] = self.mot_checkpoint_mixed_attn

        # ── Sync action_dim → nested dit configs ──────────────────────────────
        self.video_dit_config["action_dim"] = self.action_dim
        self.action_dit_config["action_dim"] = self.action_dim
