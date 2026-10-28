# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLDA model structure, shared task dimensions, and teacher configuration."""

from dataclasses import dataclass, field
from math import isqrt


@dataclass(frozen=True)
class VJEPA2Config:
    """V-JEPA2 encoder architecture; weights are loaded separately."""

    img_size: int = 256
    patch_size: int = 16
    embed_dim: int = 1408
    depth: int = 40
    num_heads: int = 22
    num_frames: int = 2
    tubelet_size: int = 2
    mlp_ratio: float = 48 / 11
    use_sdpa: bool = True
    use_rope: bool = True
    use_silu: bool = False
    wide_silu: bool = True
    qkv_bias: bool = True
    uniform_power: bool = True
    norm_layer_eps: float = 1e-6


@dataclass(frozen=True)
class VLDAModelConfig:
    """Defaults for ``model:`` in the VLDA YAML; overrides use dotlist syntax."""

    model_type: str = "vlda"
    stage: int = 1
    action_dim: int = 7
    state_dim: int = 8
    action_horizon: int = 10
    max_action_dim: int = 32
    max_state_dim: int = 32
    discrete_state_input: bool = True
    crop_action_sequences: int | None = None
    delta_token_num: int = 16
    delta_latent_dim: int = 256
    contrast_dim: int = 256
    align_loss_weight: float = 0.01
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    disable_train_autocast: bool = True
    attention_backend: str = "eager"
    compile_model: bool = False
    compile_mode: str = "max-autotune-no-cudagraphs"
    ema_start: float = 0.99925
    ema_end: float = 0.99925
    world_inference_steps: int = 3
    action_inference_steps: int = 10
    vjepa: VJEPA2Config = field(default_factory=VJEPA2Config)
    vjepa_checkpoint: str = ""
    vjepa_online_key: str = "encoder"
    vjepa_teacher_key: str = "target_encoder"

    def __post_init__(self):
        if self.attention_backend not in ("eager", "flex"):
            raise ValueError("VLDA attention_backend must be eager or flex")
        if self.stage not in (1, 2):
            raise ValueError("VLDA stage must be 1 or 2")
        if self.action_horizon <= 0 or not 0 < self.action_dim <= self.max_action_dim:
            raise ValueError("VLDA requires a positive horizon and action_dim <= max_action_dim")
        if not 0 < self.state_dim <= self.max_state_dim:
            raise ValueError("VLDA requires state_dim <= max_state_dim")
        if self.delta_token_num <= 0 or isqrt(self.delta_token_num) ** 2 != self.delta_token_num:
            raise ValueError("delta_token_num must be a positive square for pooled future loss")
        if self.delta_token_num > (self.vjepa.img_size // self.vjepa.patch_size) ** 2:
            raise ValueError("delta_token_num exceeds the number of V-JEPA spatial patches")
        if self.crop_action_sequences is not None and self.crop_action_sequences <= 0:
            raise ValueError("crop_action_sequences must be positive")
        if not 0 <= self.ema_start <= 1 or not 0 <= self.ema_end <= 1:
            raise ValueError("EMA momentum must be between zero and one")
        if min(self.world_inference_steps, self.action_inference_steps) <= 0:
            raise ValueError("Inference step counts must be positive")
