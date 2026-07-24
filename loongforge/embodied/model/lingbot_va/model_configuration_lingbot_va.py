# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the native PyTorch LingBot-VA backend."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class LingBotVAModelConfig:
    """Native LingBot-VA model and training configuration."""

    model_type: str = "lingbot_va"
    requires_tokenizer: bool = False
    latent_in_channels: int = 48
    latent_out_channels: int = 48
    latent_patch_size: Tuple[int, int, int] = (1, 2, 2)
    latent_space_scale: float = 0.5
    latent_time_scale: float = 1.0
    num_latent_frames: int = 13
    max_latent_height: int = 128
    max_latent_width: int = 256
    vae_temporal_compress: int = 1
    vae_spatial_compress: int = 8
    num_layers: int = 30
    hidden_size: int = 3072
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 24
    action_dim: int = 30
    text_dim: int = 4096
    freq_dim: int = 256
    norm_epsilon: float = 1e-6
    rope_max_seq_len: int = 1024
    cross_attn_norm: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_backend: str = "unfused"
    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None
    lingbot_va_num_train_timesteps: int = 1000
    lingbot_va_snr_shift: float = 5.0
    lingbot_va_action_snr_shift: float = 1.0
    lingbot_va_sigma_min: float = 0.0
    lingbot_va_extra_one_step: bool = True
    lingbot_va_noisy_cond_prob: float = 0.5
    lingbot_va_action_noisy_cond_prob: float = 0.0
    lingbot_va_video_loss_weight: float = 1.0
    lingbot_va_action_loss_weight: float = 1.0
    lingbot_va_window_size: int = 64
    lingbot_va_chunk_size: int = 4
    lingbot_va_use_flex_attention: bool = False
    lingbot_va_diffusers_checkpoint_path: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.latent_patch_size) != 3:
            raise ValueError("latent_patch_size must contain exactly three dimensions")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
