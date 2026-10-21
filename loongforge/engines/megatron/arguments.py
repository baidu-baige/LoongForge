# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge engine arguments for the Megatron backend.

This module defines the command-line arguments that configure backend
execution: model wrapping, tokenizer construction, parallelism, precision,
checkpoint format, and the HF bridge. Recipe, task, data, logging, and
lifecycle arguments live in ``loongforge/training/megatron/args.py``; both
providers register on the same parser.
"""

import argparse

from loongforge.chat_templates.registry import get_support_templates


def loongforge_extra_train_args_provider(parser: argparse.ArgumentParser):
    """Add Loongforge-specific arguments to the argument parser.

    This function serves as the main entry point for adding all Loongforge-specific
    training arguments, organized by functional groups.

    Args:
        parser: The base argument parser to extend.

    Returns:
        The modified argument parser with all Loongforge arguments added.
    """
    parser.conflict_handler = "resolve"
    parser = _add_extra_model_args(parser)
    parser = _add_extra_tokenizer_args(parser)
    parser = _add_model_config_args(parser)
    parser = _add_extra_training_args(parser)
    parser = _add_extra_parallel_args(parser)
    parser = _add_extra_bridge_args(parser)

    return parser


def _add_extra_bridge_args(parser):
    """Add bridge arguments"""
    group = parser.add_argument_group(title='extra-bridge')

    # Arguments for defining manner of converting FP8 checkpoint.
    # Both kebab-case and snake_case option strings are registered so callers
    # may use either convention (Megatron args are typically kebab, convert-tool
    # args are snake_case).
    group.add_argument('--fp8-force-no-requant', '--fp8_force_no_requant', dest='fp8_force_no_requant',
                       action='store_true',
                       help=("If enabled, in converting FP8 checkpoint, skip the `dequantize + re-quantize`, "
                             "directly chunk/concate the quantized data.")
    )
    group.add_argument('--force-pow-2-scales', '--force_pow_2_scales', dest='force_pow_2_scales',
                       action='store_true',
                       help=("Define whether to force destination checkpoint's scale to be power-of-two.")
    )
    group.add_argument('--amax-epsilon', '--amax_epsilon', dest='amax_epsilon',
                       type=float, default=0.0,
                       help=("Epsilon value added to the amax calculation to avoid divised by zero "
                             "when converting to FP8. Only used in Transformer Engine FP8 conversion.")
    )

    # Arguments for online dequantization of compressed-tensors INT4 HF checkpoints
    # during Bridge online loading (load_hf_checkpoint_online). Mirrors the offline
    # convert-tool flags in tools/mcore_checkpoint_convert/arguments.py so that INT4-quantized
    # HF weights (e.g. Kimi K2.6) can be loaded directly without offline conversion.
    group.add_argument('--hf-dequantize-int4', '--hf_dequantize_int4', dest='hf_dequantize_int4',
                       action='store_true',
                       help=("If enabled, on-the-fly dequantize compressed-tensors packed INT4 "
                             "weights to bf16/fp32 during HF checkpoint online loading.")
    )
    group.add_argument('--hf-dequantize-mxfp4', '--hf_dequantize_mxfp4', dest='hf_dequantize_mxfp4',
                       action='store_true',
                       help=("If enabled, on-the-fly dequantize MXFP4 packed FP4 weights "
                             "(E2M1 two-per-byte in uint8 + per-block E8M0 scale, block_size=32, "
                             "e.g. DeepSeek-V4 routed experts) to bf16/fp32 during HF checkpoint "
                             "online loading.")
    )
    group.add_argument('--hf-dequantize-dtype', '--hf_dequantize_dtype', dest='hf_dequantize_dtype',
                       default='bfloat16',
                       help=("Target dtype for INT4 dequantization (bfloat16 or float32). "
                             "Default: bfloat16")
    )
    group.add_argument('--hf-quant-config-file', '--hf_quant_config_file', dest='hf_quant_config_file',
                       default=None,
                       help=("Path to config.json carrying the compressed-tensors quantization_config; "
                             "used to locate the INT4-packed weight keys. Default: None")
    )
    # fp8 conversion args used by the mcore converter during Bridge online loading
    # (bf16->fp8 when training is fp8). Mirrors tools/mcore_checkpoint_convert/arguments.py.
    # Defaults match TransformerEngine-based training; no need to pass explicitly.
    group.add_argument('--quant-method', '--quant_method', dest='quant_method',
                       default='te', choices=['te', 'pt'],
                       help="fp8 quantization backend for bf16->fp8 conversion during bridge load. "
                            "Default: te (TransformerEngine)")
    group.add_argument('--pretrain-as-fp8', '--pretrain_as_fp8', dest='pretrain_as_fp8',
                       action='store_true',
                       help="Treat the loaded checkpoint as fp8 for pretrain (bf16 saved ckpt + pretrain fp8). "
                            "Default: False")

    # Read only by the HF bridge (tools/mcore_bridge) when it loads HF weights
    # online or merges adapters for --save-hf.
    group.add_argument(
        "--lora-alpha",
        type=int,
        help="LoRA alpha used by the HF export bridge when --save-hf merges adapters. "
             "Training LoRA shape comes from the YAML peft_config and must match."
    )
    group.add_argument(
        "--lora-dim",
        type=int,
        help="LoRA rank used by the HF export bridge when --save-hf merges adapters. "
             "Training LoRA shape comes from the YAML peft_config and must match."
    )
    group.add_argument(
        "--encoder-tensor-model-parallel-size",
        type=int,
        default=None,
        help="Encoder tensor-parallel size the HF bridge uses to shard encoder weights "
             "for heterogeneous TP. Default: None (use --tensor-model-parallel-size)",
    )
    return parser


def _add_extra_model_args(parser: argparse.ArgumentParser):
    """Add arguments that select engine-side model behavior."""
    group = parser.add_argument_group(
        title="Model Configuration",
        description="Arguments for attention backend selection"
    )

    group.add_argument(
        "--enable-fa-within-mla",
        action="store_true",
        help="Enable FlashAttention for Multi-Head Latent Attention (MLA) models. "
             "Since MLA has different QK and V head dimensions, FlashAttention is disabled "
             "by default. This option aligns head dimensions via padding to enable FlashAttention. "
             "Default: False"
    )

    return parser


# =============================================================================
# Tokenizer Arguments
# =============================================================================

def _add_extra_tokenizer_args(parser: argparse.ArgumentParser):
    """Add arguments for tokenizer configuration.

    These arguments control tokenizer type selection, special token handling,
    and vocabulary configuration.
    """
    group = parser.add_argument_group(
        title="Tokenizer Configuration",
        description="Arguments for tokenizer initialization and behavior"
    )

    group.add_argument(
        "--tokenizer-type",
        type=str,
        default=None,
        choices=["NullTokenizer", "HFTokenizer"],
        help="Tokenizer implementation to use. "
             "'NullTokenizer': no tokenization (data is pre-tokenized), "
             "'HFTokenizer': HuggingFace tokenizers library. "
             "When None, Loongforge automatically determines the appropriate tokenizer. "
             "Narrows the upstream choices on purpose: pre-tokenized NullTokenizer data "
             "is not supported. Default: None"
    )

    group.add_argument(
        "--hf-tokenizer-path",
        type=str,
        default=None,
        help="Path to HuggingFace tokenizer. Accepts either: "
             "(1) Model ID from huggingface.co (e.g., 'meta-llama/Llama-2-7b-hf'), or "
             "(2) Local directory path containing tokenizer files. "
             "Default: None"
    )

    group.add_argument(
        "--use-fast-tokenizer",
        action="store_true",
        dest="use_fast_tokenizer",
        help="Use the fast (Rust-based) tokenizer implementation when --tokenizer-type=HFTokenizer. "
             "Requires tokenizers library. Default: False"
    )

    group.add_argument(
        "--split-special-tokens",
        action="store_true",
        help="Split special tokens (e.g., <s>, </s>) into separate tokens during tokenization "
             "when --tokenizer-type=HFTokenizer. Default: False"
    )

    group.add_argument(
        "--padding-side",
        default="right",
        choices=["left", "right"],
        help="Side on which to add padding tokens. "
                "'left': padding tokens are added before the sequence. "
                "'right': padding tokens are added after the sequence. "
                "For most training setups, right padding is preferred as it keeps "
                "valid tokens aligned at the beginning of the sequence and simplifies "
                "label masking and loss computation. "
                "Default: right"
    )

    group.add_argument(
        "--additional-special-tokens",
        type=str,
        default=None,
        help="Comma-separated list of additional special tokens to add to the tokenizer. "
             "Example: '<|im_start|>,<|im_end|>'. Default: None"
    )

    group.add_argument(
        "--vocab-size-in-config-file",
        type=int,
        default=None,
        help="Vocabulary size from HuggingFace config file. Used when the tokenizer's "
             "vocab size differs from model's embedding size. Default: None"
    )

    # --padded-vocab-size comes from upstream megatron.training.arguments.
    # The tokenizer and the global chat template are built together at
    # initialization, so these two arguments stay with the tokenizer.
    group.add_argument(
        "--chat-template",
        type=str,
        choices=get_support_templates(),
        default=None,
        help="Chat template name to format instruction data. Templates define "
             "how prompts and responses are structured. "
             "Examples: 'deepseek', 'qwen', 'llama3.1', 'empty'. Default: None"
    )

    group.add_argument(
        "--chat-template-kwargs",
        type=str,
        default=None,
        help="Optional JSON object or path to a JSON file containing extra kwargs "
             "for Hugging Face chat templates. Only valid with registered "
             "`*-hf` chat templates. Example: '{\"enable_thinking\": false}'. "
             "Default: None"
    )

    return parser


# =============================================================================
# Model Config Arguments
# =============================================================================

def _add_model_config_args(parser: argparse.ArgumentParser):
    """Add arguments copied by name into the model config.

    ``build_model_config`` copies every attribute of the parsed namespace whose
    name matches a field of the target config class, so these must be
    registered before the config is built.
    """
    group = parser.add_argument_group(
        title="Model Config Overrides",
        description="Arguments that override model structure fields from the command line"
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
    group.add_argument(
        "--latent-in-channels",
        type=int,
        default=None,
        help="Number of input channels in latent space. Used by video diffusion models. "
             "Default: None"
    )
    group.add_argument(
        "--latent-out-channels",
        type=int,
        default=None,
        help="Number of output channels in latent space. Used by video diffusion models. "
             "Default: None"
    )
    group.add_argument(
        "--latent-patch-size",
        type=tuple,
        default=(1, 1, 1),
        help="Patch size (time, height, width) for latent space tokenization. "
             "Default: (1, 1, 1)"
    )
    group.add_argument(
        "--latent-space-scale",
        type=float,
        default=1.0,
        help="Spatial scaling factor for latent space. Default: 1.0"
    )
    group.add_argument(
        "--latent-time-scale",
        type=float,
        default=1.0,
        help="Temporal scaling factor for latent space. Default: 1.0"
    )
    group.add_argument(
        "--num-latent-frames",
        type=int,
        default=None,
        help="Number of frames in the latent video representation. Default: None"
    )
    group.add_argument(
        "--max-latent-height",
        type=int,
        default=None,
        help="Maximum height of video in latent space. Default: None"
    )
    group.add_argument(
        "--max-latent-width",
        type=int,
        default=None,
        help="Maximum width of video in latent space. Default: None"
    )
    return parser


def _add_extra_training_args(parser: argparse.ArgumentParser):
    """Add arguments that configure how the engine executes a step.

    Training-phase, EMA, export, and data-recipe switches live in
    ``loongforge/training/megatron/args.py``. Pipeline sequence-length
    handling, the checkpoint format, loss reduction, and the MTP embedding copy
    stay here because they describe backend execution.
    """
    group = parser.add_argument_group(
        title="Training Configuration",
        description="Arguments for step execution and checkpoint management"
    )

    group.add_argument(
        "--variable-seq-lengths",
        action="store_true",
        help="Use per-batch variable sequence lengths. Ignored for VLM families, "
             "which always enable it. SFT may force it off for tp-comm-overlap or "
             "MoE all-gather/all-to-all-seq dispatch. Default: False"
    )

    group.add_argument(
        "--ckpt-format",
        default="torch",
        choices=["torch", "torch_dist", "zarr", "fsdp_dtensor"],
        help="Checkpoint format for saving/loading model weights: "
             "'torch': standard PyTorch format, "
             "'torch_dist': distributed checkpoint format, "
             "'zarr': Zarr-based format for large models, "
             "'fsdp_dtensor': FSDP with DTensor format. "
             "Default: torch (upstream defaults to torch_dist)"
    )

    group.add_argument(
        "--loss-reduction-all-gather",
        action="store_true",
        help="Gather losses from all GPUs during loss reduction. Default: False"
    )

    group.add_argument(
        "--legacy-reporting-loss-reduction",
        action="store_true",
        help="Use legacy loss reduction method for backward compatibility. Default: False"
    )

    group.add_argument(
        "--should-get-embedding-weights-for-mtp",
        action="store_true",
        help="For models such as GLM-5, MTP does not have separate embedding weights,"
             "and in pipeline scenarios, weights need to be copied from the first PP stage. "
             "Default: False"
    )

    return parser


# =============================================================================
# Parallel Arguments
# =============================================================================

def _add_extra_parallel_args(parser):
    """Add arguments for distributed parallel execution.

    These arguments control model parallelism (tensor, pipeline, context),
    data parallelism balancing, and distributed training configuration.
    """
    group = parser.add_argument_group(
        title="Parallel Configuration",
        description="Arguments for distributed and parallel training"
    )

    # Context parallelism (deprecated argument for backward compatibility)
    group.add_argument(
        "--context-parallel-ulysses-degree",
        type=int,
        default=1,
        help="[LEGACY] Degree of Ulysses-style context parallelism. "
             "Retained for backward compatibility with older Loongforge versions. Default: 1"
    )
    # Deprecated pipeline parallelism arguments
    group.add_argument(
        '--custom-pipeline-layers',
        type=str,
        default=None,
        help="[DEPRECATED: Use --pipeline-model-parallel-layout] "
             "Comma-separated layer counts per pipeline stage for imbalanced PP. "
             "Example: '19,20,20,21' for 4 stages with different layer counts. "
             "Default: None"
    )

    group.add_argument(
        '--custom-virtual-pipeline-layers',
        type=str,
        default=None,
        help="[DEPRECATED: Use --pipeline-model-parallel-layout] "
             "Layer counts for virtual pipeline with interleaved scheduling. "
             "Example: '19,20,20,21' for 2 virtual chunks across 4 stages. "
             "Default: None"
    )

    # Encoder heterogeneous data parallelism
    group.add_argument(
        '--enable-encoder-hetero-dp',
        action="store_true",
        default=False,
        help="Enable heterogeneous data parallelism for encoder layers. "
             "Allows different DP degrees for encoder vs decoder. Default: False"
    )

    group.add_argument(
        '--enable-full-hetero-dp',
        default=False,
        action="store_true",
        help="Enable full heterogeneous data parallelism. Default: False"
    )

    group.add_argument(
        '--full-hetero-dp-cpu-offload',
        default=False,
        action="store_true",
        help="Offload full hetero DP intermediate embeddings and gradients to "
             "pinned CPU memory to reduce GPU memory usage. Uses async CUDA "
             "streams for transfers. Requires --enable-full-hetero-dp. "
             "Default: False"
    )

    # Data parallelism load balancing
    group.add_argument(
        '--use-vlm-dp-balance',
        action='store_true',
        help="Enable dynamic load balancing across data parallel ranks for VLM models. "
             "Adjusts computation distribution based on runtime statistics. Default: False"
    )

    group.add_argument(
        '--use-vit-dp-balance',
        action='store_true',
        help="Enable dynamic load balancing across data parallel ranks for ViT models. "
             "Adjusts computation distribution based on runtime statistics. Default: False"
    )

    group.add_argument(
        '--vlm-dp-balance-warmup-iters',
        nargs='+',
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        help="Iteration indices to collect statistics for VLM DP balance coefficient warmup. "
             "Default: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]"
    )

    group.add_argument(
        '--dp-balance-verbose',
        action='store_true',
        default=False,
        help="Print DP balance diagnostic information including imbalance ratio, "
             "per-DP-rank load distribution, and reorder decisions. Default: False"
    )

    group.add_argument(
        '--dp-balance-max-len-ratio-vlm',
        type=float,
        default=1.2,
        help="Maximum sequence length ratio for VLM-level DP load balancing. "
             "Limits each DP rank's maximum sequence length to (avg_len * ratio). "
             "The constraint is always on for VLM balancing: the disable value (None) "
             "cannot be passed on the command line. Default: 1.2"
    )

    group.add_argument(
        '--dp-balance-max-len-ratio-vit',
        type=float,
        default=None,
        help="Maximum sequence length ratio for ViT-level DP load balancing. "
             "Limits each DP rank's maximum sequence length to (avg_len * ratio). "
             "Leave unset to disable the constraint (default for ViT mode). Default: None"
    )

    group.add_argument(
        '--dp-balance-trigger-threshold-vlm',
        type=float,
        default=0.2,
        help="Minimum imbalance ratio threshold for triggering VLM-level DP load balancing. "
             "Balancing is skipped when imbalance ratio is below this threshold. "
             "imbalance_ratio = (max_load / avg_load) - 1. Default: 0.2"
    )

    group.add_argument(
        '--dp-balance-trigger-threshold-vit',
        type=float,
        default=0.2,
        help="Minimum imbalance ratio threshold for triggering ViT-level DP load balancing. "
             "Balancing is skipped when imbalance ratio is below this threshold. "
             "imbalance_ratio = (max_load / avg_load) - 1. Default: 0.2"
    )
    return parser
