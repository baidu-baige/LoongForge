# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V-4.6 language model config."""

from dataclasses import dataclass

from loongforge.models.foundation.qwen3_5.qwen3_5_config import Qwen35Config
from loongforge.utils.constants import VisionLanguageModelFamilies


@dataclass
class MiniCPMV46Config(Qwen35Config):
    """MiniCPM-V-4.6 text backbone.

    The layer structure matches Qwen3.5 text closely, but MiniCPM uses plain
    1D RoPE position ids rather than Qwen3-VL style M-RoPE.
    """

    gated_delta_rule_backend: str = "auto"
    gated_norm_backend: str = "auto"
    causal_conv_backend: str = "auto"
    linear_backend: str = "transformer_engine"
    full_attention_linear_backend: str | None = None
    linear_attention_linear_backend: str | None = None
    mlp_linear_backend: str | None = None
    torch_cross_entropy_at_tp1: bool = False

    model_type = VisionLanguageModelFamilies.MINICPM_V_4_6
