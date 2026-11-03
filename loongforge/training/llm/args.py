# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Language model arguments shared by LLM and VLM runs."""

import argparse
import warnings

from megatron.core.transformer.enums import AttnBackend

from loongforge.engines.megatron import print_rank_0


def add_llm_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register shared language model arguments on *parser*."""
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


def _validate_llm_model_args(args):
    """Set the MLA attention backend before tokenizer validation."""
    if args.enable_fa_within_mla:
        args.attention_backend = AttnBackend.flash
        args.padding_v_to_qk_dim = True
        print_rank_0(
            "--enable-fa-within-mla is enabled, setting attention backend to FlashAttention",
            args.rank,
        )


def _validate_extra_training_args(args):
    """Validate training arguments"""

    if args.use_dsa_fused:
        current_variant = getattr(args, "experimental_attention_variant", None)
        if current_variant != "dsa":
            raise ValueError(
                "--use-dsa-fused requires experimental_attention_variant='dsa'. "
                f"Got experimental_attention_variant={current_variant!r}."
            )
        print_rank_0(
            "INFO: --use-dsa-fused is enabled; Omni fused DSA modules will be used.",
            args.rank,
        )
    if getattr(args, "use_dsa_sp_first", False):
        if not args.use_dsa_fused:
            raise ValueError(
                "--use-dsa-sp-first requires --use-dsa-fused to be enabled."
            )
        if not args.sequence_parallel:
            raise ValueError(
                "--use-dsa-sp-first requires --sequence-parallel to be enabled, "
                "since the scheme relies on S/TP partitioning."
            )
        print_rank_0(
            "INFO: --use-dsa-sp-first is enabled; All-to-All communication will be "
            "eliminated in the DSA fused path.",
            args.rank,
        )
    if getattr(args, "dsa_indexer_topk_freq", 1) > 1:
        if getattr(args, "experimental_attention_variant", None) != "dsa":
            raise ValueError(
                "--dsa-indexer-topk-freq > 1 (IndexShare) requires "
                "experimental_attention_variant='dsa'."
            )
        if getattr(args, "overlap_moe_expert_parallel_comm", False):
            raise ValueError(
                "DSA IndexShare is not supported with --overlap-moe-expert-parallel-comm: "
                "the fine-grained a2a path calls TransformerLayer._forward_attention directly "
                "and does not thread the per-forward index-share carrier."
            )
        print_rank_0(
            "INFO: DSA IndexShare is enabled "
            f"(topk_freq={args.dsa_indexer_topk_freq}, "
            f"skip_topk_offset={args.dsa_indexer_skip_topk_offset}); "
            "each pipeline stage must start on a computing layer.",
            args.rank,
        )
    if args.num_experts is None and args.moe_token_dispatcher_type in ['allgather', 'alltoall_seq']:
        args.moe_token_dispatcher_type = 'alltoall'
        warnings.warn(
            f"Since num_experts is {args.num_experts}, moe_token_dispatcher_type argument is not applicable. "
            f"Setting it to 'alltoall' to pass transformer config validation."
        )
