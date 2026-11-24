# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for native Megatron Core DeepSeek-V4 attention."""

from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridAttention,
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSubmodules,
)

__all__ = [
    "DSv4HybridAttention",
    "DSv4HybridSelfAttention",
    "DSv4HybridSelfAttentionSubmodules",
]
