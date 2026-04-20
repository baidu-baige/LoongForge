# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Omni-owned experimental attention variants for DeepSeek."""

from .dsa_fused import (
    DSAIndexerFused,
    DSAIndexerFusedSubmodules,
    DSAttentionFused,
    DSAttentionFusedSubmodules,
)
from .multi_latent_attention import MLASelfAttentionFused

__all__ = [
    "DSAIndexerFused",
    "DSAIndexerFusedSubmodules",
    "DSAttentionFused",
    "DSAttentionFusedSubmodules",
    "MLASelfAttentionFused",
]
