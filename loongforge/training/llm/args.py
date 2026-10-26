# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LLM-only Megatron arguments."""

import argparse


def add_llm_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register LLM-only arguments on *parser*."""
    group = parser.add_argument_group(
        title="LLM Configuration",
        description="Arguments for LLM attention and DSA kernels",
    )
    group.add_argument(
        "--enable-fa-within-mla",
        action="store_true",
        help="Enable FlashAttention for Multi-Head Latent Attention (MLA) models. "
             "Since MLA has different QK and V head dimensions, FlashAttention is disabled "
             "by default. This option aligns head dimensions via padding to enable FlashAttention. "
             "Default: False"
    )
    group.add_argument(
        "--use-dsa-fused",
        action="store_true",
        help="Force use of Omni fused DSA implementation for DeepSeek models. "
             "Default: False"
    )
    group.add_argument(
        "--dsa-kernel-backend",
        type=str,
        default="auto",
        choices=["auto", "sm100", "sm80", "torch_ref", "tilelang"],
        help="Kernel backend for the fused DSA path. 'auto' picks by compute "
             "capability: sm100 (FP8 indexer + FlashMLA) on SM90/SM100, sm80 "
             "(BF16 indexer, Ampere has no FP8 tensor cores) on A100/A800. "
             "'torch_ref' runs the PyTorch reference implementations -- correct "
             "but slow and memory hungry, intended for numerical validation and "
             "for machines with no DSA kernels installed. 'tilelang' forces the "
             "TileLang sparse-MLA backward. Default: auto"
    )
    group.add_argument(
        "--use-dsa-sp-first",
        action="store_true",
        help="Use SP-First partitioning for DSA fused path. Eliminates All-to-All "
             "communication by making projection layers hold full-head weights and "
             "operate on sequence-split data. Requires --use-dsa-fused and "
             "--sequence-parallel. Default: False"
    )
    group.add_argument(
        "--absorb-backend",
        type=str,
        default="te",
        choices=["te", "torch"],
        help="Backend for MLA absorb projections. "
             "'te' uses TEGroupedLinear, 'torch' uses torch.einsum with sliced weights. "
             "Default: te"
    )
    return parser
