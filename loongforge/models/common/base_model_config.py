# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""model configuration classes."""

from transformers import PretrainedConfig

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_config import MLATransformerConfig
import dataclasses
from typing import Optional, List, Dict, Set
import torch
from collections import defaultdict


@dataclasses.dataclass
class BasePeftModelConfig():    
    """configuration class for the peft transformer"""
    target_modules: List[str] = dataclasses.field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    exclude_modules: List[str] = dataclasses.field(default_factory=list)
    canonical_mapping: Dict[str, Set] = dataclasses.field(default_factory=lambda: defaultdict(set))


@dataclasses.dataclass
class BaseModelConfig(TransformerConfig, PretrainedConfig):
    """Base configuration class for LoongForge models."""

    freeze: bool = False
    model_type: str = None
    model_spec: Optional[List[str]] = None
    peft_config: Optional[BasePeftModelConfig] = None
    convert_file: str = None


    # --- FP8 dynamic policy (benchmark-driven selective FP8) ---
    fp8_dynamic_policy_path: Optional[str] = None
    """Path to a JSON policy file exported by the TE layer benchmark.
    When set together with ``selective_fp8=True``, the init-time FP8 decision
    is driven by benchmark-derived per-module-kind token thresholds."""

    fp8_dynamic_num_tokens: int = 0
    """Effective per-GPU token count for FP8 dynamic policy decisions.
    When left at 0 (default), auto-computed from ``args.seq_length * args.micro_batch_size``."""

    def __post_init__(self):
        PretrainedConfig.__init__(self)
        TransformerConfig.__post_init__(self)


@dataclasses.dataclass
class BaseModelMLAConfig(MLATransformerConfig, PretrainedConfig):
    """Base configuration class for LoongForge models with multi-latent attention."""

    freeze: bool = False
    model_type: str = None
    model_spec: Optional[List[str]] = None
    peft_config: Optional[BasePeftModelConfig] = None
    convert_file: str = None


    # --- FP8 dynamic policy (same as BaseModelConfig) ---
    fp8_dynamic_policy_path: Optional[str] = None
    fp8_dynamic_num_tokens: int = 0

    def __post_init__(self):
        PretrainedConfig.__init__(self)
        MLATransformerConfig.__post_init__(self)


@dataclasses.dataclass
class BaseModelStditConfig(TransformerConfig):
    """configuration class for the stdit transformer"""
    model_type: str = None

    num_latent_frames: int = 0
    """Number of frames in the latent."""

    max_latent_height: int = 0
    """Maximum height of the latent."""

    max_latent_width: int = 0
    """Maximum width of the latent."""

    latent_in_channels: int = 0
    """Number of channels in the input latent."""

    latent_out_channels: int = 0
    """Number of channels in the output latent."""

    latent_patch_size: tuple = (1, 1, 1)
    """Patch size of the latent."""

    latent_space_scale: float = 1.0
    """Space scale of the latent."""

    latent_time_scale: float = 1.0
    """Time scale of the latent."""

    caption_channels: int = 0
    """Number of channels in the caption."""

    max_text_length: int = 0
    """Maximum text length."""

    max_image_length: int = 0
    """Maximum image length."""

    max_video_length: int = 0
    """Maximum video length."""

    def __post_init__(self):
        super().__post_init__()

        if len(self.latent_patch_size) != 3:
            raise ValueError(
                f'latent_patch_size: {self.latent_patch_size} must have three dimensions.'
            )
        if self.latent_patch_size[1] != self.latent_patch_size[2]:
            raise ValueError(
                f'latent_patch_size: {self.latent_patch_size} must have equal height and width.'
            )
        if self.latent_time_scale is None:
            self.latent_time_scale = 1.0 / self.latent_patch_size[0]

        elif self.latent_time_scale != 1.0 / self.latent_patch_size[0]:
            raise ValueError(
                f'latent_time_scale: {self.latent_time_scale} must be 1.0 / latent_patch_size[0].'
            )

        if self.latent_space_scale is None:
            self.latent_space_scale = 1.0 / self.latent_patch_size[1]
        elif self.latent_space_scale != 1.0 / self.latent_patch_size[1]:
            raise ValueError(
                f'latent_space_scale: {self.latent_space_scale} must be 1.0 / latent_patch_size[1].'
            )

