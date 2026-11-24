# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for native Megatron Core DeepSeek-V4 CSA/HCA."""

import torch

from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
)


def _unscale_indexer_grad(grad, scale_inv):
    """Dequantize an indexer grad while zeroing rows on the quantizer's eps floor."""
    epsilon_scale_floor = 2.0 ** -48 * 1.5
    scale_inv = scale_inv.unsqueeze(-1)
    usable = torch.isfinite(scale_inv) & (scale_inv > epsilon_scale_floor)
    safe_scale_inv = torch.where(usable, scale_inv, torch.ones_like(scale_inv))
    return (grad / safe_scale_inv).masked_fill(~usable.expand_as(grad), 0.0)


__all__ = [
    "CompressedSparseAttention",
    "CompressedSparseAttentionSubmodules",
    "Compressor",
    "CompressorSubmodules",
    "CSAIndexer",
    "CSAIndexerSubmodules",
]
