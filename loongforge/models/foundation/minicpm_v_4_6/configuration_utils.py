# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Configuration initialization helpers for MiniCPM-V-4.6."""

from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig


def initialize_pretrained_config(config: PretrainedConfig) -> None:
    """Support both Transformers PretrainedConfig initialization protocols."""
    post_init = getattr(PretrainedConfig, "__post_init__", None)
    if post_init is not None:
        post_init(config)
    else:
        PretrainedConfig.__init__(config)


def initialize_transformer_config(config: TransformerConfig) -> None:
    initialize_pretrained_config(config)
    TransformerConfig.__post_init__(config)
