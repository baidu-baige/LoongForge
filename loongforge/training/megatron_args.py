# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Training Arguments Module

This module defines command-line arguments for LoongForge training pipeline,
including model configuration, tokenizer settings, SFT data processing,
video processing, multimodal training, and parallel execution.
"""

import argparse
import importlib
import os
import warnings

import torch
from megatron.core.rerun_state_machine import RerunStateMachine
from megatron.core.transformer.enums import AttnBackend
from megatron.training.arguments import validate_args
from megatron.training.utils import warn_rank_0

from loongforge.data import constants as data_constants
from loongforge.data.chat_templates.template import get_support_templates
from loongforge.engines.megatron import (
    convert_custom_pipeline_to_layout,
    get_device_arch_version,
    is_torch_min_version,
    print_rank_0,
)
from loongforge.engines.megatron.tokenizer import get_default_tokenizer
from loongforge.engines.megatron.utils.runtime import get_default_sft_dataset_config
from loongforge.training.diffusion.args import (
    _align_wan_packing_seq_length,
    _validata_extra_custom_args,
    add_diffusion_args,
)
from loongforge.training.llm.args import (
    _validate_extra_training_args,
    _validate_llm_model_args,
    add_llm_args,
)
from loongforge.training.vlm.args import (
    _validate_extra_multimodal_args,
    _validate_vlm_model_args,
    add_vlm_args,
)


# =============================================================================
# Tensor Logging Arguments (llm-inspector)
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
             "Default: None"
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

    group.add_argument(
        "--padded-vocab-size",
        type=int,
        default=None,
        help="Explicitly specify padded vocabulary size. Used for models with custom "
             "vocabulary padding requirements. Default: None"
    )

    return parser


# =============================================================================
# Training Arguments
# =============================================================================

def _extend_cuda_graph_scope_choices(parser: argparse.ArgumentParser):
    """Add ``'per_microbatch'`` to the choices of Megatron's ``--cuda-graph-scope``.

    Avoids patching the upstream Megatron parser. Looks up the action by its
    option string and mutates its ``choices``/``help`` in place. Silently
    no-ops if the upstream definition has changed and the action can't be
    found, so we never break parsing.
    """
    extra_help = (
        ' "per_microbatch" scope (LoongForge extension) captures one CUDA graph per '
        'micro-batch with eager RNG between sub-graphs for bit-exact '
        'alignment with pure eager. Only supported with --cuda-graph-impl=local.'
    )
    for action in parser._actions:
        if "--cuda-graph-scope" in getattr(action, "option_strings", ()):
            choices = list(action.choices or [])
            if "per_microbatch" not in choices:
                choices.append("per_microbatch")
                action.choices = choices
            if action.help and "per_microbatch" not in action.help:
                action.help = action.help.rstrip() + extra_help
            return


def _add_extra_training_args(parser: argparse.ArgumentParser):
    """Add arguments for training configuration.

    These arguments control training phase, checkpointing, logging,
    and EMA (Exponential Moving Average).
    """
    group = parser.add_argument_group(
        title="Training Configuration",
        description="Arguments for training loop and checkpoint management"
    )

    group.add_argument(
        "--variable-seq-lengths",
        action="store_true",
        help="[DEPRECATED] This flag is ignored. Variable sequence length support "
             "is now automatic. Default: False"
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
             "Default: torch"
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

    group.add_argument(
        "--cuda-graph-pad-length",
        type=int,
        default=None,
        dest="cuda_graph_pad_length",
        help="Fixed sequence length for CUDA graph mode. "
             "When set (e.g. --cuda-graph-pad-length 220), the data collator pads "
             "all token sequences to this length (padding='max_length', truncation=True) "
             "so all batches have identical tensor shapes — a prerequisite for CUDA graph "
             "capture. When None (default), dynamic padding is used. "
             "Should be set to a value slightly larger than the maximum actual token "
             "length in the dataset to minimize wasted computation. "
             "NOTE: currently only consumed by the gr00t SFT pipeline "
             "(Gr00tN1d6DataCollator); other model pipelines ignore this flag.",
    )

    group.add_argument(
        "--force-all-weight-decay",
        action="store_false",
        default=None,
        help="Force all parameters into weight-decay group, overriding default "
             "exclusions for bias and LayerNorm. Default: None (use model defaults)"
    )

    # Extend Megatron's --cuda-graph-scope to accept "per_microbatch" without patching
    # the upstream parser. Mutates the already-registered action's choices/help
    # in place (this provider runs after add_megatron_arguments).
    _extend_cuda_graph_scope_choices(parser)

    return parser


def _add_extra_parallel_args(parser):
    """Add shared context and pipeline parallel arguments."""
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

    return parser


# =============================================================================
# Training run argument groups
# =============================================================================


def _add_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add run-selection, phase, and export arguments."""
    group = parser.add_argument_group(
        title="Run Configuration",
        description="Arguments that select the model config and describe the run lifecycle",
    )
    group.add_argument(
        "--config-file",
        type=str,
        required=False,
        help="Path to YAML configuration file containing model and training settings. "
             "Example: /path/to/config.yaml. When provided, other arguments may be "
             "overridden by config file values. Default: None",
    )
    group.add_argument(
        "--model-name",
        type=str,
        default=None,
        required=False,
        help="Name identifier for the model architecture. Must match a registered model "
             "in the MODEL_CONFIG_REGISTRY. "
             "Example: 'llama2-7b', 'qwen2-7b'. Default: None",
    )
    group.add_argument(
        "--training-phase",
        type=str,
        default=data_constants.TrainingPhase.PRETRAIN,
        choices=[data_constants.TrainingPhase.PRETRAIN, data_constants.TrainingPhase.SFT],
        help="Training phase: 'pretrain' for pre-training, 'sft' for supervised fine-tuning. "
             "Default: pretrain",
    )
    group.add_argument(
        "--save-hf",
        type=str,
        default='false',
        choices=['true', 'false'],
        help="Save HF checkpoint at the end of training. Choices: [true, false]. Default: true.",
    )
    group.add_argument(
        "--save-hf-path",
        type=str,
        default=None,
        help="Path to save the HF model checkpoint. If not specified, will save to <save>/release_hf_weights/",
    )
    group.add_argument(
        "--random-fallback-cpu",
        action="store_true",
        help="Generate random numbers on CPU then move to target device. "
             "Useful for hardware that has issues with on-device random number generation. "
             "Default: False"
    )

    return parser


def _add_freeze_and_init_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add parameter-freezing and initialization-source arguments."""
    group = parser.add_argument_group(
        title="Freeze and Initialization",
        description="Arguments for freezing parameters and choosing the starting weights",
    )
    group.add_argument(
        "--freeze-parameters",
        type=str,
        nargs="*",
        default=[],
        help="List of parameter name prefixes to freeze during training. "
             "Frozen parameters will not be updated by the optimizer. "
             "Example: --freeze-parameters 'embed' 'lm_head' freezes embedding and output layers. "
             "Default: [] (no freezing)",
    )
    group.add_argument(
        "--freeze-parameters-regex",
        type=str,
        default=None,
        help="Regular expression pattern to match parameter names to freeze. "
             "Example: '.*bias.*' freezes all bias parameters. "
             "Takes precedence over --freeze-parameters if both are specified. "
             "Default: None",
    )
    group.add_argument(
        "--allow-missing-adapter-checkpoint",
        action="store_true",
        help="Allow model loading to proceed even when adapter checkpoint is missing. "
             "Useful when starting training from a base model without adapter weights. "
             "Default: False (missing adapter checkpoint raises error)",
    )
    group.add_argument('--lora-alpha', type=int, help="Lora alpha for LoRA fine tuning.")

    group.add_argument('--lora-dim', type=int, help="Lora dim for LoRA fine tuning.")

    group.add_argument(
        "--trainable-modules",
        default=["all"],
        nargs="*",
        help="List of modules to train (unfrozen). Options: 'all', 'language_model', "
             "'adapter', 'vision_model', 'language_expert_linear', 'vision_expert_linear'. "
             "Default: ['all']"
    )

    return parser


def _add_ema_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add Exponential Moving Average arguments for the training loop."""
    group = parser.add_argument_group(
        title="EMA",
        description="Arguments for Exponential Moving Average of model parameters",
    )
    group.add_argument(
        "--enable-ema",
        action="store_true",
        help="Enable Exponential Moving Average (EMA) of model parameters. "
             "EMA provides smoothed parameter estimates for more stable inference. "
             "Default: False",
    )
    group.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="Decay rate for EMA parameter updates. Higher values = slower update. "
             "Default: 0.9999",
    )
    group.add_argument(
        "--save-ema",
        type=str,
        default=None,
        help="Directory path to save EMA checkpoints. Defaults to ${args.save}/ema. "
             "Default: None",
    )
    group.add_argument(
        "--load-ema",
        type=str,
        default=None,
        help="Directory path to load EMA checkpoint from. Defaults to ${args.load}/ema. "
             "Default: None",
    )
    return parser


def _add_recipe_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add data-recipe and preprocessing arguments."""
    group = parser.add_argument_group(
        title="Data Recipe",
        description="Arguments for SFT sequence packing and dataloader state",
    )
    group.add_argument(
        "--packing-sft-data",
        action="store_true",
        help="Pack multiple short sequences into a single training sample to improve "
             "GPU utilization. Default: False",
    )
    group.add_argument(
        "--dataloader-save",
        type=str,
        default=None,
        help="Path to save Energon dataloader state for resumable training. Default: None",
    )
    return parser


def _add_sft_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments for supervised fine-tuning data configuration.

    These arguments control dataset selection, preprocessing, packing,
    and training behavior for SFT tasks.
    """
    group = parser.add_argument_group(
        title="SFT Data Configuration",
        description="Arguments for supervised fine-tuning data processing",
    )

    group.add_argument(
        "--sft-dataset-config",
        type=str,
        default=None,
        help="Path to YAML file containing dataset configurations. "
             "Defines dataset formats and processing rules. "
             "Default: configs/data/sft_dataset_config.yaml",
    )
    group.add_argument(
        "--sft-dataset",
        nargs="*",
        default=None,
        help="List of dataset names for combined train/valid/test splits. "
             "Names must be defined in --sft-dataset-config. "
             "Examples: 'dataset1' or 'dataset1 dataset2'. "
             "Mutually exclusive with --sft-train-dataset/--sft-valid-dataset/--sft-test-dataset. "
             "Default: None",
    )
    group.add_argument(
        "--sft-train-dataset",
        nargs="*",
        default=None,
        help="List of training dataset names. Used with --train-data-path. "
             "Follows same naming rules as --sft-dataset. Default: None",
    )
    group.add_argument(
        "--sft-valid-dataset",
        nargs="*",
        default=None,
        help="List of validation dataset names. Used with --valid-data-path. "
             "Follows same naming rules as --sft-dataset. Default: None",
    )
    group.add_argument(
        "--sft-test-dataset",
        nargs="*",
        default=None,
        help="List of test dataset names. Used with --test-data-path. "
             "Follows same naming rules as --sft-dataset. Default: None",
    )
    group.add_argument(
        "--sft-sort-batch",
        action="store_true",
        help="Sort dataset samples by length (smallest to largest) for more efficient "
             "padding. Sorting occurs after packing if --packing-sft-data is enabled. ",
    )
    group.add_argument(
        "--sft-data-streaming",
        action="store_true",
        help="Enable streaming mode for large datasets that don't fit in memory. "
             "Data is loaded on-demand during training. Default: False",
    )
    group.add_argument(
        "--streaming-buffer-size",
        type=int,
        default=16384,
        help="Buffer size for random sampling in streaming mode. Larger buffers "
             "provide more randomization but use more memory. Default: 16384",
    )
    group.add_argument(
        "--sft-data-mix-strategy",
        type=str,
        choices=["concat", "interleave_under", "interleave_over"],
        default="concat",
        help="Strategy for mixing multiple SFT datasets: "
             "'concat': concatenate datasets sequentially; "
             "'interleave_under': interleave datasets and stop when the shortest dataset is exhausted; "
             "'interleave_over': interleave datasets until the longest dataset is exhausted.",
    )
    group.add_argument(
        "--sft-num-preprocess-workers",
        type=int,
        default=None,
        help="Number of worker processes for data preprocessing. Only applies to "
             "non-streaming mode. More workers speed up preprocessing but use more memory. "
             "Default: None (auto-detect)",
    )
    group.add_argument(
        "--train-on-prompt",
        action="store_true",
        help="Include prompt tokens in loss computation. By default, loss is computed "
             "only on response tokens. Default: False",
    )
    group.add_argument(
        "--history-mask-loss",
        action="store_true",
        help="Compute loss only on the last turn response in multi-turn conversations, "
             "masking the loss for tokens from previous turns. Default: False",
    )
    group.add_argument(
        "--is-tokenized-data",
        action="store_true",
        help="Indicate that input data is already tokenized. Skips tokenization step "
             "in data processing pipeline. Default: False",
    )
    group.add_argument(
        "--enable-discard-sample",
        action="store_true",
        help="Discard samples that exceed --seq-length instead of truncating. "
             "Useful for ensuring consistent sequence lengths. Default: False",
    )
    group.add_argument(
        "--packing-buffer-size",
        type=int,
        default=10000,
        help="Size of the sample buffer used for sequence packing. "
             "Samples are first loaded into this buffer and then packed into sequences "
             "by the packing algorithm. Effective only when --packing-sft-data is enabled.",
    )
    group.add_argument(
        "--use-fixed-seq-lengths",
        action="store_true",
        help="Pad all sequences to exactly --seq-length. Currently only supported "
             "for language models. Default: False",
    )
    return parser


def _add_loop_logging_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments for training-loop logging."""
    group = parser.add_argument_group(
        title="Training Loop Logging",
        description="Arguments for progress and memory logging in the training loop",
    )
    group.add_argument(
        "--no-detail-log",
        action="store_false",
        dest="log_detail",
        help="Disable detailed timing logs during training. Default: enabled",
    )
    group.add_argument(
        "--detail-log-interval",
        type=int,
        default=20,
        help="Interval (iterations) between detailed timing log reports. "
             "Only effective when timing-log-level is 0. Default: 20",
    )
    group.add_argument(
        "--log-memory-stats",
        action="store_true",
        default=False,
        help="Log GPU memory statistics (allocated/peak) during training. Default: False",
    )
    return parser


def _add_log_tensor_args(parser):
    """Add arguments for tensor logging and debugging via llm-inspector.

    These arguments interface with the llm-inspector library for debugging
    model training. When enabled (via --enable-log-tensor), the system will
    register hooks to capture tensor statistics during training.

    Required dependency: llm-inspector

    Usage:
        These arguments are typically used together:
        1. --enable-log-tensor: Enable the tensor logging feature
        2. --log-tensor-name-pattern: Filter which modules to log (regex)
        3. --log-tensor-stage: Choose when to log (init/forward/backward)
        4. --log-tensor-iter-pattern: Specific iterations to log
        5. --log-tensor-mbs-pattern: Specific micro-batches to log
        6. --log-tensor-layer-pattern: Specific layers to log
        7. --log-tensor-rank: Which GPU ranks to log
        8. --save-tensor: Save tensors to disk (vs just print norms)
        9. --save-tensor-dir: Directory to save tensor files

    Example:
        python train.py --enable-log-tensor \\
            --log-tensor-name-pattern ".*attention.*" \\
            --log-tensor-stage "forward,backward" \\
            --log-tensor-iter-pattern "0,100,1000" \\
            --save-tensor \\
            --save-tensor-dir "./tensor_logs"

    Note: If llm-inspector is not installed, these arguments will be ignored
    silently. The feature is controlled by HAS_INSPECTOR flag in training_utils.py.
    """
    group = parser.add_argument_group(
        title="Tensor Logging (llm-inspector)",
        description="Arguments for debugging tensor statistics via llm-inspector library. "
                    "Requires llm-inspector package to be installed.",
    )
    group.add_argument(
        "--enable-log-tensor",
        action="store_true",
        help="[llm-inspector] Enable tensor logging for debugging. When enabled, tensor statistics "
             "will be traced during training using the llm-inspector library. "
             "Requires llm-inspector to be installed. Default: False",
    )
    group.add_argument(
        "--log-tensor-name-pattern",
        type=str,
        default=None,
        help="[llm-inspector] Regex pattern to filter module names for tensor logging. "
             "When None (default), logs all modules. "
             "Example: '.*attention.*' to log only attention modules. Default: None",
    )
    group.add_argument(
        "--log-tensor-stage",
        type=str,
        default="forward",
        choices=["init", "forward", "backward"],
        help="[llm-inspector] Training stage at which to log tensors. "
             "'init': model initialization, 'forward': forward pass, 'backward': backward pass. "
             "Multiple stages can be specified comma-separated, e.g., 'forward,backward'. "
             "Default: forward",
    )
    group.add_argument(
        "--log-tensor-iter-pattern",
        type=str,
        default=None,
        help="[llm-inspector] Comma-separated iteration indices at which to log tensors. "
             "Example: '8,15,20' logs tensors at iterations 8, 15, and 20. "
             "When None, logs at all iterations. Default: None",
    )
    group.add_argument(
        "--log-tensor-mbs-pattern",
        type=str,
        default=None,
        help="[llm-inspector] Comma-separated micro-batch indices at which to log tensors. "
             "Example: '0,2,4' logs tensors for micro-batches 0, 2, and 4. "
             "Default: None",
    )
    group.add_argument(
        "--log-tensor-layer-pattern",
        type=str,
        default=None,
        help="[llm-inspector] Comma-separated layer indices at which to log tensors. "
             "Example: '0,5,10' logs tensors for layers 0, 5, and 10. "
             "Default: None",
    )
    group.add_argument(
        "--log-tensor-rank",
        type=str,
        default="0",
        help="[llm-inspector] Comma-separated GPU ranks at which to log tensors. "
             "Example: '0,1,2,4' logs tensors on ranks 0, 1, 2, and 4. "
             "Default: 0",
    )

    # Tensor saving options
    group.add_argument(
        "--save-tensor",
        action="store_true",
        help="[llm-inspector] Save logged tensors to disk files for offline analysis. "
             "When False, only prints tensor norms to log file. Default: False",
    )
    group.add_argument(
        "--save-tensor-dir",
        type=str,
        default="",
        help="[llm-inspector] Directory path to save tensor files. Required when --save-tensor is enabled. "
             "Default: '' (current directory)",
    )
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
    return parser


def loongforge_extra_train_args_provider(parser: argparse.ArgumentParser):
    """Register all Megatron training arguments on *parser*."""
    parser.conflict_handler = "resolve"
    parser = _add_run_args(parser)
    parser = _add_freeze_and_init_args(parser)
    parser = _add_extra_tokenizer_args(parser)
    parser = _add_sft_args(parser)
    parser = _add_recipe_args(parser)
    parser = add_vlm_args(parser)
    parser = add_diffusion_args(parser)
    parser = _add_extra_training_args(parser)
    parser = _add_ema_args(parser)
    parser = add_llm_args(parser)
    parser = _add_extra_parallel_args(parser)
    parser = _add_loop_logging_args(parser)
    parser = _add_log_tensor_args(parser)
    parser = _add_extra_bridge_args(parser)
    return parser


def validate_loongforge_extra_args(args, config):
    """ "Validate LoongForge extra arguments"""
    _validate_extra_model_args(args, config)
    _validate_extra_tokenizer_args(args)
    _validate_extra_sft_args(args)
    _validate_extra_training_args(args)
    _validate_extra_multimodal_args(args)
    _validata_extra_custom_args(args)
    _validata_extra_parallel_args(args)

    # megatron one_logger is not supported in loongforge
    args.enable_one_logger = False


def validate_megatron_args(args):
    """Validate Megatron arguments"""
    _align_wan_packing_seq_length(args)
    _validate_legacy_pipeline_args(args)
    validate_args(args)


# YAML fields may use a different argparse destination.
_YAML_KEY_ALIASES = {
    "fp8_param": "fp8_param_gather",
}


def _validate_extra_model_args(args, config):
    """Setup model config based on the given model name."""
    if config is not None:
        # Collect dataclass field names from the config _target_ class (if resolvable)
        # to allow YAML-only fields through even when not in argparse namespace.
        _config_class_fields = set()
        _target = config.get('_target_') if hasattr(config, 'get') else getattr(config, '_target_', None)
        if _target:
            try:
                import importlib
                import dataclasses as _dc
                _mod_path, _cls_name = _target.rsplit('.', 1)
                _mod = importlib.import_module(_mod_path)
                _cls = getattr(_mod, _cls_name)
                if _dc.is_dataclass(_cls):
                    _config_class_fields = {f.name for f in _dc.fields(_cls)}
            except Exception:
                pass

        for key in config:
            attr = _YAML_KEY_ALIASES.get(key, key)
            if hasattr(args, attr) or attr in _config_class_fields:
                setattr(args, attr, config[key])
                suffix = f" (from YAML '{key}')" if attr != key else ""
                print_rank_0(f"  {attr} = {config[key]}{suffix}", args.rank)

        print_rank_0(
            "---------------- End of configuration ----------------", args.rank
        )

    _validate_vlm_model_args(args)
    _validate_llm_model_args(args)


def _validate_extra_tokenizer_args(args):
    """Setup tokenizer based on the given model name."""
    if args.tokenizer_type is None:
        args.tokenizer_type = get_default_tokenizer(args.model_family)
        assert (
            args.tokenizer_type is not None
        ), "No default tokenizer found for the given model name, please set --tokenizer-type"

        print_rank_0(f"Configure tokenizer to {args.tokenizer_type}", args.rank)

    if args.additional_special_tokens is not None:
        args.additional_special_tokens = [
            token.strip() for token in args.additional_special_tokens.split(",")
        ]


def _validate_extra_sft_args(args):
    """Validate SFT arguments"""
    if args.training_phase != data_constants.TrainingPhase.SFT:
        return

    if args.tokenizer_type != "HFTokenizer":
        raise ValueError(
            "--tokenizer-type should be HFTokenizer when training phase is sft"
        )

    args.dataloader_type = "external"
    print_rank_0(
        "INFO: Set dataloader type to external since --training-phase=SFT", args.rank
    )

    if args.chat_template is None:
        raise ValueError("--chat-template is required when training phase is sft")

    if args.train_on_prompt and args.history_mask_loss:
        raise ValueError(
            "--train-on-prompt and --history-mask-loss cannot both be True at the same time"
        )

    if args.sft_dataset_config is None:
        # set default sft-dataset-config
        default_config = get_default_sft_dataset_config()
        if default_config is not None:
            args.sft_dataset_config = default_config
            print_rank_0(
                f"WARNING: --sft-dataset-config is not specified, setup to default config ({default_config})",
                args.rank,
            )
        else:
            raise ValueError(
                "--sft-dataset-config is not specified, and "
                "the default config does not exist, please setup it"
            )
    if args.sft_data_streaming:
        assert (
            args.sft_sort_batch is None or not args.sft_sort_batch
        ), '--sft-sort-batch" cannot be used together with --sft-data-streaming'

    if args.use_fixed_seq_lengths:
        args.variable_seq_lengths = False
    else:
        # Defaults to True but enforced as fixed-length for specific features (e.g., tp-comm-overlap/ moe allgather)
        args.variable_seq_lengths = True
        if args.tp_comm_overlap:
            # tp_comm_overlap requires fixed-length
            args.variable_seq_lengths = False

        if (
            args.num_experts is not None
            and args.num_experts > 0
            and args.moe_token_dispatcher_type in ["allgather", "alltoall_seq"]
        ):
            # allgather or alltoall_seq requires fixed-length
            args.variable_seq_lengths = False

    if args.packing_sft_data:
        if args.micro_batch_size > 1:
            args.micro_batch_size = 1
            print_rank_0(
                "WARING: Setting args.micro_batch_size to 1 since packing_sft_data is enabled",
                args.rank,
            )

        if args.context_parallel_size > 1:
            if (
                args.context_parallel_ulysses_degree < args.context_parallel_size
                and args.cp_comm_type == "allgather"
            ):
                args.cp_comm_type = "p2p"
                print_rank_0(
                    "WARNING: Setting args.cp_comm_type to p2p since ring attention "
                    "does not support all gather while packing_sft_data is enabled",
                    args.rank,
                )

    if args.padding_side == "left":
        args.padding_side = "right"
        print_rank_0(
            "WARING: Setting args.padding_side to right when run sft.", args.rank
        )

    # NOTE on ordering: this validator runs BEFORE megatron's `validate_args`,
    # so `args.sft_chunkpipe_mode` and `args.data_parallel_size` are not yet
    # populated. We are already inside the SFT branch (early-return above),
    # which is exactly the condition under which `sft_chunkpipe_mode` is later
    # set to True in megatron arguments.py, so we don't need to re-check it.
    # `data_parallel_size` is recomputed locally with megatron's formula.
    if getattr(args, "enable_chunkpipe", False):
        # A-series (Ampere / sm80) fused-DSA backend only supports ChunkPipe in
        # pretrain; the SFT ChunkPipe path is not implemented for it yet.
        from loongforge.models.common.experimental_attention_variant.dsa_kernel_backend import (
            BACKEND_SM80,
            resolve_backend_name,
        )

        if resolve_backend_name(args) == BACKEND_SM80:
            raise NotImplementedError(
                "SFT ChunkPipe is not supported on the sm80 (A-series / Ampere) "
                "fused-DSA backend yet; only pretrain is supported. Use "
                "--training-phase pretrain, or select a non-sm80 "
                "--dsa-kernel-backend."
            )
        if args.sft_data_streaming:
            raise NotImplementedError(
                "SFT chunkpipe does not support --sft-data-streaming."
            )
        if getattr(args, "dataloader_save", None):
            raise NotImplementedError(
                "SFT chunkpipe does not support --dataloader-save. "
                "Please resume from iteration checkpoints via consumed_train_samples."
            )
        if args.context_parallel_size != 1:
            raise NotImplementedError(
                "SFT chunkpipe temporarily requires context_parallel_size == 1. "
                f"Got context_parallel_size={args.context_parallel_size}."
            )
        if args.sequence_parallel and args.chunksize % args.tensor_model_parallel_size != 0:
            raise ValueError(
                "SFT chunkpipe requires chunksize to be divisible by "
                "tensor_model_parallel_size when sequence_parallel is enabled. "
                f"Got chunksize={args.chunksize}, "
                f"tensor_model_parallel_size={args.tensor_model_parallel_size}."
            )
        if getattr(args, "mtp_num_layers", 0) and args.mtp_num_layers > 0:
            if args.overlap_grad_reduce:
                raise NotImplementedError(
                    "SFT chunkpipe + MTP temporarily does not support "
                    "--overlap-grad-reduce because MTP checkpoint/recompute can "
                    "trigger duplicate grad-ready marking on output/shared embedding weights."
                )

        # SFT chunkpipe per-sample loss path: default when
        # --calculate-per-token-loss is not set. Requires the new 2-tuple
        # reporting format (legacy_reporting_loss_reduction=False); the new
        # path returns a scalar S/(N*G), not the 2-tuple [loss, num_tokens]
        # the legacy path expects.
        if not args.calculate_per_token_loss:
            assert not args.legacy_reporting_loss_reduction, (
                "SFT chunkpipe per-sample loss path requires "
                "legacy_reporting_loss_reduction=False. "
                "Either drop --legacy-reporting-loss-reduction or pass "
                "--calculate-per-token-loss to fall back to the "
                "token-weighted path."
            )
            print_rank_0(
                "INFO: SFT chunkpipe per-sample loss path enabled (default). "
                "Pass --calculate-per-token-loss to fall back to the "
                "token-weighted loss path.",
                args.rank,
            )

        # Decide whether to enable the heterogeneous-DP synthesis path.
        # attn_dp != expert_dp happens when EP > 1 or ETP differs from TP, in
        # which case the LPT partition cannot guarantee identical group-size
        # multisets across DP ranks; without synthesis the MoE All2All would
        # desync. The sampler reads `args.chunkpipe_enable_synthesis` at
        # construction time and switches to `_equal_size_partition`.
        tp = args.tensor_model_parallel_size
        pp = args.pipeline_model_parallel_size
        cp = args.context_parallel_size
        ep = getattr(args, "expert_model_parallel_size", 1) or 1
        etp = getattr(args, "expert_tensor_parallel_size", None)
        if etp is None:
            etp = tp
        attn_dp = args.world_size // (tp * pp * cp)
        expert_dp = args.world_size // (etp * pp * ep)
        vpp = args.num_virtual_stages_per_pipeline_rank or 1
        if vpp != 1 and attn_dp != expert_dp:
            raise NotImplementedError(
                f"SFT chunkpipe with virtual pipeline parallelism (vpp={vpp}) is temporarily not supported "
                f"when attn_dp ({attn_dp}) != expert_dp ({expert_dp}). "
                f"Please set vpp=1, or ensure attn_dp == expert_dp."
            )
        args.chunkpipe_enable_synthesis = (attn_dp != expert_dp)
        if args.chunkpipe_enable_synthesis:
            print_rank_0(
                f"INFO: SFT chunkpipe synthesis path enabled "
                f"(attn_dp={attn_dp}, expert_dp={expert_dp}). "
                f"Group-size multisets will be aligned across DP ranks via "
                f"short-pool synthesis with all-or-nothing rollback.",
                args.rank,
            )


def _validata_extra_parallel_args(args):
    """Validate parallel arguments"""
    # check cp, NOTE: maybe removed in the future
    if args.context_parallel_size > 1:
        if (
            args.context_parallel_ulysses_degree is None
            or args.context_parallel_ulysses_degree < 1
        ):
            # not set
            return

        assert (
            args.hierarchical_context_parallel_sizes is None
        ), "ERROR: Cannot specify both hierarchical_context_parallel_sizes and context_parallel_ulysses_degree"

        assert (
            args.context_parallel_ulysses_degree <= args.context_parallel_size
            and args.context_parallel_size % args.context_parallel_ulysses_degree == 0
        ), "ERROR: context_parallel_ulysses_degree must less than context_parallel_size and divisible by it"

        # only cp
        if args.context_parallel_ulysses_degree == 1:
            # just use cp
            assert (
                "a2a" not in args.cp_comm_type
            ), "p2p or allgather are allowed for non-ulysses context parallel"
        # only ulysses
        elif args.context_parallel_ulysses_degree == args.context_parallel_size:
            # just use all2all
            args.cp_comm_type = "a2a"
            print_rank_0(
                "Setting cp_comm_type to a2a because context_parallel_ulysses_degree equals "
                "to context_parallel_size",
                args.rank,
            )
        else:
            cp_degree = (
                args.context_parallel_size // args.context_parallel_ulysses_degree
            )
            args.cp_comm_type = "a2a+p2p"
            args.hierarchical_context_parallel_sizes = [
                args.context_parallel_ulysses_degree,
                cp_degree,
            ]

    # check tp overlap
    if args.tp_comm_overlap:
        if importlib.util.find_spec("torch_xmlir") is None and args.fp16:
            args.tp_comm_overlap = False
            print_rank_0(
                "Disabling tp comm overlap since fp16 is not supported on GPU",
                args.rank,
            )


def _validate_legacy_pipeline_args(args):
    """Validate parallel arguments"""
    # Uneven virtual pipeline parallelism
    assert not (
        args.custom_pipeline_layers is not None
        and args.pipeline_model_parallel_layout is not None
    ), 'custom_pipeline_layers and pipeline_model_parallel_layout cannot be set at the same time'

    # convert custom_pipeline_layers to pipeline_model_parallel_layout
    if args.custom_pipeline_layers is not None:
        assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, \
            'The layer partition mode conflicts.'

        pp_splits = []
        if args.custom_pipeline_layers.find(',') != -1:
            pp_splits = [int(s) for s in args.custom_pipeline_layers.split(',')]

        assert len(pp_splits) == args.pipeline_model_parallel_size, (
            f"the number of elements in --custom-pipeline-layers must be equal to "
            f"pipeline size {args.pipeline_model_parallel_size}")

        assert args.num_layers == sum(pp_splits), \
            f"the sum of --custom-pipeline-layers must be equal to {args.num_layers}"

        if args.num_virtual_stages_per_pipeline_rank is not None:
            assert all(x >= args.num_virtual_stages_per_pipeline_rank for x in pp_splits), \
                f"when num_virtual_stages_per_pipeline_rank is {args.num_virtual_stages_per_pipeline_rank}, \
                each element in custom_pipeline_layers must be >= num_virtual_stages_per_pipeline_rank"

        args.pipeline_model_parallel_layout = convert_custom_pipeline_to_layout(
            custom_pipeline_layers=args.custom_pipeline_layers,
            num_virtual_stages_per_pipeline_rank=args.num_virtual_stages_per_pipeline_rank,
            mtp_num_layers=args.mtp_num_layers,
        )

        # To pass the megatron validation
        if args.num_virtual_stages_per_pipeline_rank is not None:
            args.num_virtual_stages_per_pipeline_rank = None

    # add loongforge for custom virtual pipeline layers check
    if args.custom_virtual_pipeline_layers is not None:
        assert args.pipeline_model_parallel_size > 1, \
            "custom_virtual_pipeline_layers is only supported when pipeline_model_parallel_size > 1"
        assert args.num_virtual_stages_per_pipeline_rank is not None, \
                "num_virtual_stages_per_pipeline_rank should be set when custom_virtual_pipeline_layers is set"
        assert args.custom_pipeline_layers is None, \
                "custom_pipeline_layers should not be set when custom_virtual_pipeline_layers is set"

        custom_vpp_splits = []
        if args.custom_virtual_pipeline_layers.find(',') != -1:
            custom_vpp_splits = [int(s) for s in args.custom_virtual_pipeline_layers.split(',') if s.strip()]

        assert sum(custom_vpp_splits) == args.num_layers, (
            f"the sum of --custom-virtual-pipeline-layers must be equal to {args.num_layers}"
        )

        assert len(custom_vpp_splits) == args.pipeline_model_parallel_size * \
            args.num_virtual_stages_per_pipeline_rank, (
            f"the number of elements in --custom-virtual-pipeline-layers must be equal to "
            f"pipeline size {args.pipeline_model_parallel_size} * num_virtual_stages_per_pipeline_rank "
            f"{args.num_virtual_stages_per_pipeline_rank}"
        )

        args.pipeline_model_parallel_layout = convert_custom_pipeline_to_layout(
            custom_virtual_pipeline_layers=args.custom_virtual_pipeline_layers,
            num_virtual_stages_per_pipeline_rank=args.num_virtual_stages_per_pipeline_rank,
            mtp_num_layers=args.mtp_num_layers,
        )

        # To pass the megatron validation
        if args.num_virtual_stages_per_pipeline_rank is not None:
            args.num_virtual_stages_per_pipeline_rank = None


def validate_custom_model_args(name, args):
    """Validate non foundational model arguments"""
    _validate_custom_model_args(name, args)


def _check_arg_is_not_none(args, arg):
    """Check if an argument is not None."""
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


# Adapted from megatron/training/arguments.py
def _validate_custom_model_args(name, args, defaults={}):
    """Validate non foundational model arguments"""
    if args.custom_pipeline_recompute_layers is not None:
        assert args.recompute_granularity == "full", \
            "recompute-granularity should be full, when custom-pipeline-recompute-layers is set."

        pp_recompute_splits = []
        if args.custom_pipeline_recompute_layers.find(',') != -1:
            pp_recompute_splits = [int(s) for s in args.custom_pipeline_recompute_layers.split(',')]

        assert len(pp_recompute_splits) == args.pipeline_model_parallel_size, (
            f"the number of elements in --custom-pipeline-recompute-layers must be equal to "
            f"pipeline size {args.pipeline_model_parallel_size}")

        args.custom_pipeline_recompute_layers = pp_recompute_splits
    # Temporary model parallel size. Added by loongforge
    if args.pipeline_model_parallel_size > 1:
        warnings.warn(f"WARNING: Now for {name}, we only support pipeline model parallel size 1.")
        args.pipeline_model_parallel_size = 1

    if args.sequence_parallel:
        warnings.warn(f"WARNING: Now for {name}, we do not support sequence parallel.")
        args.sequence_parallel = False

    if args.expert_model_parallel_size > 1:
        warnings.warn(f"WARNING: Now for {name}, we only support expert model parallel size 1.")
        args.expert_model_parallel_size = 1

    if args.num_virtual_stages_per_pipeline_rank is not None:
        warnings.warn(f"WARNING: Now for {name}, we do not support num_virtual_stages_per_pipeline_rank.")
        args.num_virtual_stages_per_pipeline_rank = None

    if args.context_parallel_ulysses_degree is not None:
        warnings.warn(f"WARNING: Now for {name}, we do not support context_parallel_ulysses_degree.")
        args.context_parallel_ulysses_degree = 1

    if args.context_parallel_size > 1:
        warnings.warn(f"WARNING: Now for {name}, we only support context parallel size 1.")
        args.context_parallel_size = 1

    if args.tp_comm_overlap:
        warnings.warn(f"WARNING: Now for {name}, we do not support tp_comm_overlap.")
        args.tp_comm_overlap = False

    if getattr(args, 'pipeline_model_parallel_layout', None) is not None:
        warnings.warn(f"WARNING: Now for {name}, we do not support pipeline_model_parallel_layout.")
        args.pipeline_model_parallel_layout = None

    if args.num_query_groups is None:
        # To pass transformer config post init check
        warnings.warn(f"WARNING: Now for {name}, the num_query_groups is None, using default value"
                      " tensor_model_parallel_size to pass the transformer config check.")
        args.num_query_groups = args.tensor_model_parallel_size

    if args.num_attention_heads is None:
        # To pass transformer config post init check
        warnings.warn(f"WARNING: Now for {name}, the num_attention_heads is None, using default value"
                      " tensor_model_parallel_size to pass the transformer config check.")
        args.num_attention_heads = args.tensor_model_parallel_size

    if args.moe_router_topk is not None:
        warnings.warn(f"WARNING: Now for {name}, we do not support moe_router_topk.")
        args.moe_router_topk = None
    if args.moe_router_group_topk is not None:
        warnings.warn(f"WARNING: Now for {name}, we do not support moe_router_group_topk.")
        args.moe_router_group_topk = None

    # Temporary
    assert args.non_persistent_ckpt_type in ['global', 'local', None], \
        'Currently only global and local checkpoints are supported'
    if args.non_persistent_ckpt_type == 'local':
        try:
            from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import \
                LocalCheckpointManager  # noqa: F401
        except ModuleNotFoundError as e:
            raise RuntimeError('nvidia_resiliency_ext is required for local checkpointing') from e

    # Set args.use_dist_ckpt from args.ckpt_format.
    if args.use_legacy_models:
        assert args.ckpt_format == "torch", \
            "legacy model format only supports the 'torch' checkpoint format."
    args.use_dist_ckpt = args.ckpt_format != "torch"


    total_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size

    # Total model size.
    assert args.world_size % total_model_size == 0, (
        f"world size ({args.world_size}) is not divisible by total_model_size ({total_model_size=})"
    )

    if args.attention_backend == AttnBackend.local:
        assert args.spec[0] == 'local' , '--attention-backend local is only supported with --spec local'

    # Pipeline model parallel size.
    args.transformer_pipeline_model_parallel_size = args.pipeline_model_parallel_size

    args.data_parallel_size = args.world_size // total_model_size

    if args.rank == 0:
        print('using world size: {}, data-parallel size: {}, '
              'context-parallel size: {}, '
              'hierarchical context-parallel sizes: {}, '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {}, '.format(
                  args.world_size, args.data_parallel_size,
                  args.context_parallel_size,
                  args.hierarchical_context_parallel_sizes,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)

    if args.hierarchical_context_parallel_sizes:
        from numpy import prod
        assert args.context_parallel_size == prod(args.hierarchical_context_parallel_sizes)
    if "a2a+p2p" in args.cp_comm_type:
        assert args.hierarchical_context_parallel_sizes is not None, \
        "--hierarchical-context-parallel-sizes must be set when a2a+p2p is used in cp comm"

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    if args.data_path is not None and args.split is None:
        legacy_default_split_value = '969, 30, 1'
        if args.rank == 0:
            print('WARNING: Please specify --split when using --data-path. Using legacy default value '
                  f'of "{legacy_default_split_value}"')
        args.split = legacy_default_split_value

    use_data_path = (args.data_path is not None) or (args.data_args_path is not None)
    if use_data_path:
        # Exactly one of the two has to be None if we use it.
        assert (args.data_path is None) or (args.data_args_path is None)
    use_per_split_data_path = any(
        elt is not None
        for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]) or \
            args.per_split_data_args_path is not None
    if use_per_split_data_path:
         # Exactly one of the two has to be None if we use it.
        assert any(elt is not None
                   for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]) is False \
                or args.per_split_data_args_path is None

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0

    # Uneven virtual pipeline parallelism
    assert args.num_layers_per_virtual_pipeline_stage is None or args.num_virtual_stages_per_pipeline_rank is None, \
        ('--num-layers-per-virtual-pipeline-stage and '
         '--num-virtual-stages-per-pipeline-rank cannot be set at the same time')

    if args.num_layers_per_virtual_pipeline_stage is not None or args.num_virtual_stages_per_pipeline_rank is not None:
        if args.overlap_p2p_comm:
            assert args.pipeline_model_parallel_size > 1, \
                'When interleaved schedule is used, pipeline-model-parallel size '\
                'should be greater than 1'
        else:
            assert args.pipeline_model_parallel_size > 2, \
                'When interleaved schedule is used and p2p communication overlap is disabled, '\
                'pipeline-model-parallel size should be greater than 2 to avoid having multiple '\
                'p2p sends and recvs between same 2 ranks per communication batch'

        if args.num_virtual_stages_per_pipeline_rank is None:
            assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, \
                ('please use --num-virtual-stages-per-pipeline-rank to specify virtual pipeline parallel '
                 'degree when enable uneven pipeline parallelism')
            if args.num_layers is not None:
                num_layers = args.num_layers
            else:
                num_layers = args.decoder_num_layers

            if args.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if args.account_for_loss_in_pipeline_split:
                num_layers += 1

            assert num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                'number of layers of the model must be divisible pipeline model parallel size'
            num_layers_per_pipeline_stage = num_layers // args.transformer_pipeline_model_parallel_size

            assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
                'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
            args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
                args.num_layers_per_virtual_pipeline_stage
        else:
            args.virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank
    else:
        args.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False
        args.align_param_gather = False
        # Only print warning if PP size > 1.
        if args.rank == 0 and args.pipeline_model_parallel_size > 1:
            print('WARNING: Setting args.overlap_p2p_comm and args.align_param_gather to False '
                  'since non-interleaved schedule does not support overlapping p2p communication '
                  'and aligned param AG')

        if args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None \
            and args.custom_pipeline_layers is None:
            # Divisibility check not applicable for T5 models which specify encoder_num_layers
            # and decoder_num_layers.
            if args.num_layers is not None:
                num_layers = args.num_layers

                if args.account_for_embedding_in_pipeline_split:
                    num_layers += 1

                if args.account_for_loss_in_pipeline_split:
                    num_layers += 1

                assert num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                    'Number of layers should be divisible by the pipeline-model-parallel size'
    if args.rank == 0:
        print(f"Number of virtual stages per pipeline stage: {args.virtual_pipeline_model_parallel_size}")

    if args.data_parallel_sharding_strategy == "optim_grads_params":
        args.overlap_param_gather = True
        args.overlap_grad_reduce = True

    if args.data_parallel_sharding_strategy == "optim_grads":
        args.overlap_grad_reduce = True

    if args.overlap_param_gather:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather only supported with distributed optimizer'
        assert args.overlap_grad_reduce, \
            'Must use --overlap-param-gather with --overlap-grad-reduce'
        assert not args.use_legacy_models, \
            '--overlap-param-gather only supported with MCore models'

    if args.use_torch_fsdp2:
        assert is_torch_min_version("2.4.0"), \
            'FSDP2 requires PyTorch >= 2.4.0 with FSDP 2 support.'
        assert args.pipeline_model_parallel_size == 1, \
            '--use-torch-fsdp2 is not supported with pipeline parallelism'
        assert args.expert_model_parallel_size == 1, \
            '--use-torch-fsdp2 is not supported with expert parallelism'
        assert not args.use_distributed_optimizer, \
            "--use-torch-fsdp2 is not supported with MCore's distributed optimizer"
        assert not args.gradient_accumulation_fusion, \
            '--use-torch-fsdp2 is not supported with gradient accumulation fusion'
        assert args.ckpt_format == 'torch_dist', \
            '--use-torch-fsdp2 requires --ckpt-format torch_dist'
        assert args.untie_embeddings_and_output_weights, \
            '--use-torch-fsdp2 requires --untie-embeddings-and-output-weights'
        assert not args.fp16, \
            '--use-torch-fsdp2 not supported with fp16 yet'
        assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1", \
            'FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one'

    if args.overlap_param_gather_with_optimizer_step:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather-with-optimizer-step only supported with distributed optimizer'
        assert args.overlap_param_gather, \
            'Must use --overlap-param-gather-with-optimizer-step with --overlap-param-gather'
        assert args.virtual_pipeline_model_parallel_size is not None, \
            '--overlap-param-gather-with-optimizer-step only supported with interleaved pipeline parallelism'
        assert not args.use_dist_ckpt, \
            '--overlap-param-gather-with-optimizer-step not supported with distributed checkpointing yet'

    dtype_map = {
        'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp8': torch.uint8,
    }
    map_dtype = lambda d: d if isinstance(d, torch.dtype) else dtype_map[d]

    args.main_grads_dtype = map_dtype(args.main_grads_dtype)
    args.main_params_dtype = map_dtype(args.main_params_dtype)
    args.exp_avg_dtype = map_dtype(args.exp_avg_dtype)
    args.exp_avg_sq_dtype = map_dtype(args.exp_avg_sq_dtype)

    if args.fp8_param_gather:
        assert args.use_distributed_optimizer or args.use_torch_fsdp2, \
            '--fp8-param-gather only supported with distributed optimizer or torch fsdp2'

    if args.use_megatron_fsdp:
        # NOTE: The flag `use_custom_fsdp` is deprecated and will be removed in future versions.
        #       Please use `use_megatron_fsdp` instead, as all functionality will be migrated there.
        #       Future updates will drop support for `use_custom_fsdp` to avoid confusion.
        args.use_custom_fsdp = True

        if args.data_parallel_sharding_strategy in ["optim_grads_params", "optim_grads"]:
            warn_rank_0(
                'Please make sure your TransformerEngine support FSDP + gradient accumulation fusion',
                args.rank,
            )

        if args.data_parallel_sharding_strategy == "optim_grads_params":
            assert args.check_weight_hash_across_dp_replicas_interval is None, \
                'check_weight_hash_across_dp_replicas_interval is not supported with optim_grads_params'

        assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1", \
            'FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one'

        assert args.ckpt_format == "fsdp_dtensor", \
            "Megatron FSDP only supports fsdp_dtensor checkpoint format"

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:
            args.check_for_nan_in_loss_and_grad = False
            if args.rank == 0:
                print('WARNING: Setting args.check_for_nan_in_loss_and_grad to False since '
                      'dynamic loss scaling is being used')
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if args.accumulate_allreduce_grads_in_fp32:
            assert args.main_grads_dtype == torch.float32, \
                "--main-grads-dtype can only be fp32 when --accumulate-allreduce-grads-in-fp32 is set"

        if args.grad_reduce_in_bf16:
            args.accumulate_allreduce_grads_in_fp32 = False
        elif not args.accumulate_allreduce_grads_in_fp32 and args.main_grads_dtype == torch.float32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # data
    assert args.num_dataset_builder_threads > 0

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.

    # Note: default false, but for sft, we should support variable sequence lengths
    # args.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    if args.num_layers is None:
        warnings.warn("WARNING: For some components, like image projector, num_layers is None, using default value 1" \
        "to pass the validation check.")
        args.num_layers = 1

    if args.hidden_size is None:
        warnings.warn("WARNING: For some components, like image projector, hidden_size is None, using default value "
                      "num_attention_heads to pass the validation check.")
        args.hidden_size = args.num_attention_heads

    # Checks.
    if args.ffn_hidden_size is None:
        if args.swiglu:
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size

    if args.kv_channels is None and args.num_attention_heads is not None:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None and args.context_parallel_size > 1:
        assert args.seq_length % (args.context_parallel_size * 2) == 0, \
            'seq-length should be a multiple of 2 * context-parallel-size ' \
            'if context-parallel-size > 1.'

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if args.seq_length is not None:
        assert args.max_position_embeddings >= args.seq_length, \
            f"max_position_embeddings ({args.max_position_embeddings}) must be greater than " \
            f"or equal to seq_length ({args.seq_length})."
    if args.decoder_seq_length is not None:
        assert args.max_position_embeddings >= args.decoder_seq_length
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    # Persistent fused layer norm.
    if not is_torch_min_version("1.11.0a0"):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # MoE overlap handling: non-foundation components (e.g., VIT) do not participate
    # in MoE all2all overlap scheduling, so disable overlap flags for them.
    # VIT recompute is configured independently of foundation's MoE a2a settings.
    # To customize VIT recompute, use YAML override:
    #   +model.image_encoder.recompute_granularity=selective
    #   +model.image_encoder.recompute_modules=[core_attn,layernorm]
    orig_overlap_moe = getattr(args, 'overlap_moe_expert_parallel_comm', False)
    if orig_overlap_moe:
        warnings.warn(f"Warning: Now for {name}, we do not support overlap_moe_expert_parallel_comm and "
                      "delay_wgrad_compute.")
        args.overlap_moe_expert_parallel_comm = False
        args.delay_wgrad_compute = False

    # Filter out a2a_overlap modules from recompute_modules for non-foundation components,
    # since they don't participate in MoE all2all overlap scheduling.
    if args.recompute_modules:
        non_a2a_modules = [m for m in args.recompute_modules if not m.startswith('a2a')]
        if len(non_a2a_modules) < len(args.recompute_modules):
            warnings.warn(f"WARNING: Now for {name} model, a2a_overlap modules are not supported, "
                          "ignoring them.")
        args.recompute_modules = non_a2a_modules

    # Filter out MLA-only recompute modules from non-MLA components.
    if args.recompute_modules and not getattr(args, 'multi_latent_attention', False):
        non_mla_modules = [m for m in args.recompute_modules if m != 'mla_up_proj']
        if len(non_mla_modules) < len(args.recompute_modules):
            warn_rank_0(
                f"WARNING: Now for {name} model, mla_up_proj recompute is not supported, ignoring it.",
                args.rank,
            )
        args.recompute_modules = non_mla_modules

    # When foundation uses selective recompute for MoE a2a overlap,
    # VIT inherits 'selective' but doesn't participate in MoE overlap.
    # Restore VIT to 'full' recompute by default — avoid requiring users to
    # mix a2a and non-a2a modules in CLI (which would pollute LLM's recompute config).
    # Users can independently configure VIT recompute via YAML override:
    #   +model.image_encoder.recompute_granularity=selective
    #   +model.image_encoder.recompute_modules=[core_attn,layernorm]
    if orig_overlap_moe and args.recompute_granularity == 'selective':
        warnings.warn(
            f"INFO: {name} does not participate in MoE all2all overlap scheduling. "
            f"Restoring recompute_granularity to 'full' for {name} to enable "
            f"full activation checkpointing. To customize {name} recompute independently, "
            f"use YAML override: +model.{name}.recompute_granularity=<granularity> "
            f"+model.{name}.recompute_modules=[<modules>]"
        )
        args.recompute_granularity = 'full'
        if args.recompute_method is None:
            args.recompute_method = 'uniform'
        if args.recompute_num_layers is None:
            args.recompute_num_layers = 1

    if args.recompute_granularity == 'selective':
        assert args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    if args.fine_grained_activation_offloading:
        warnings.warn(f"WARNING: Now for {name} model, fine_grained_activation_offloading is not supported.")
        args.fine_grained_activation_offloading = False

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        if args.sequence_parallel:
            warnings.warn("Disabling sequence parallelism because tensor model parallelism is disabled")
        args.sequence_parallel = False

    if args.tp_comm_overlap:
        assert args.sequence_parallel, \
            'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.tensor_model_parallel_size > 1 or args.context_parallel_size > 1 and get_device_arch_version() < 10:
        # CUDA_DEVICE_MAX_CONNECTIONS requirement no longer exists since the Blackwell architecture
        if args.use_torch_fsdp2 or getattr(args, "use_custom_fsdp", False):
            fsdp_impl = "Torch-FSDP2" if args.use_torch_fsdp2 else "Custom-FSDP"
            warnings.warn(
                f"Using tensor model parallelism or context parallelism with {fsdp_impl} together. "
                "Try not to using them together since they require different CUDA_MAX_CONNECTIONS "
                "settings for best performance. sequence parallelism requires setting the "
                f"environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 while {fsdp_impl} "
                "requires not setting CUDA_DEVICE_MAX_CONNECTIONS=1 for better parallelization.")
    if args.preprocess_data_on_cpu is True:
        print("Skipping CUDA_DEVICE_MAX_CONNECTIONS checks because use megatron preprocess data")
    else:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1" and get_device_arch_version() < 10:
            # CUDA_DEVICE_MAX_CONNECTIONS requirement no longer exists since the Blackwell architecture
            if args.sequence_parallel:
                warnings.warn(
		    "Using sequence parallelism requires setting the environment variable "
                    "CUDA_DEVICE_MAX_CONNECTIONS to 1")
            if args.async_tensor_model_parallel_allreduce:
                warnings.warn(
                    "Using async gradient all reduce requires setting the environment "
                    "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Keep the 'add bias' args in sync; add_qkv_bias is more targeted.
    if args.add_bias_linear:
        args.add_qkv_bias = True

    if args.decoupled_lr is not None or args.decoupled_min_lr is not None:
        assert not args.use_legacy_models, \
            '--decoupled-lr and --decoupled-min-lr is not supported in legacy models.'

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = 'rope'
    if args.rotary_interleaved and args.apply_rope_fusion:
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    if args.rotary_interleaved and args.use_legacy_models:
        raise RuntimeError('--rotary-interleaved is not supported in legacy models.')
    if args.position_embedding_type != 'rope':
        args.apply_rope_fusion = False

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type not in ['rope', 'alibi']:
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # Relative position embeddings arguments
    if args.position_embedding_type == 'relative':
        assert args.transformer_impl == "transformer_engine", \
            'Local transformer implementation currently does not support attention bias-based position embeddings.'

    # MoE Spec check
    if args.num_experts is not None:
        warnings.warn("Warning: For those non foundation model, num_experts must be None.")
        args.num_experts = None

    # Context parallel
    if args.context_parallel_size > 1:
        assert not args.use_legacy_models, "Context parallelism is not supported in legacy models."

    # Distributed checkpointing checks
    if args.use_dist_ckpt and args.use_legacy_models:
        raise RuntimeError('--use-dist-ckpt is not supported in legacy models.')

    # add loongforge for custom pipeline layers check
    if args.custom_pipeline_layers is not None:
        warnings.warn("Warning: For those non foundation model, custom_pipeline_layers must be None.")
        args.custom_pipeline_layers = None

    # # add loongforge for custom virtual layers in first pipeline stage check
    # if hasattr(args, 'custom_virtual_layers_first_pipeline') and \
    #         args.custom_virtual_layers_first_pipeline is not None:
    #     warnings.warn("Warning: For those non foundation model, custom_virtual_layers_first_pipeline must be None.")
    #     args.custom_virtual_layers_first_pipeline = None

    # add loongforge for custom virtual pipeline layers check
    if hasattr(args, 'custom_virtual_pipeline_layers') and args.custom_virtual_pipeline_layers is not None:
        warnings.warn("Warning: For those non foundation model, custom_virtual_pipeline_layers must be None.")
        args.custom_virtual_pipeline_layers = None

    # if args.custom_pipeline_recompute_layers is not None:
    #     warnings.warn("Warning: For those non foundation model, custom_pipeline_recompute_layers must be None.")
    #     args.custom_pipeline_recompute_layers = None

    # Data blend checks
    assert args.mock_data + \
           bool(args.data_path) + \
           any([args.train_data_path, args.valid_data_path, args.test_data_path]) <= 1, \
               "A single data source must be provided in training mode, else None"

    # Deterministic mode
    if args.deterministic_mode:
        assert not args.use_flash_attn, "Flash attention can not be used in deterministic mode."
        assert not args.cross_entropy_loss_fusion, "Cross Entropy Fusion is currently not deterministic."

        all_reduce_choices = ["Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"]
        assert os.getenv("NCCL_ALGO", -1) != -1 and os.getenv("NCCL_ALGO") in all_reduce_choices, \
            f"NCCL_ALGO must be one of {all_reduce_choices}."

        torch.use_deterministic_algorithms(True)

    # Update the printed args to reflect that `apply_query_key_layer_scaling` also controls `attention_softmax_in_fp32`
    if args.apply_query_key_layer_scaling:
        args.attention_softmax_in_fp32 = True

    if args.result_rejected_tracker_filename is not None:
        # Append to passed-in args.iterations_to_skip.
        iterations_to_skip_from_file = RerunStateMachine.get_skipped_iterations_from_tracker_file(
            args.result_rejected_tracker_filename
        )
        args.iterations_to_skip.extend(iterations_to_skip_from_file)

    # Make sure all functionality that requires Gloo process groups is disabled.
    if not args.enable_gloo_process_groups:
        if args.use_distributed_optimizer:
            # If using distributed optimizer, must use distributed checkpointing.
            # Legacy checkpointing uses Gloo process groups to collect full distributed
            # optimizer state in the CPU memory of DP rank 0.
            assert args.use_dist_ckpt

    # Checkpointing
    if args.ckpt_fully_parallel_save_deprecated and args.rank == 0:
        print('--ckpt-fully-parallel-save flag is deprecated and has no effect.'
              ' Use --no-ckpt-fully-parallel-save to disable parallel save.')
    if (
        args.use_dist_ckpt
        and not args.ckpt_fully_parallel_save
        and args.use_distributed_optimizer
        and args.rank == 0
    ):
        print('Warning: With non-parallel ckpt save and DistributedOptimizer,'
              ' it will be impossible to resume training with different parallelism.'
              ' Consider removing flag --no-ckpt-fully-parallel-save.')
    if args.use_dist_ckpt_deprecated and args.rank == 0:
        print('--use-dist-ckpt is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')
    if args.dist_ckpt_format_deprecated and args.rank == 0:
        print('--dist-ckpt-format is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')

    # Inference args
    if args.inference_batch_times_seqlen_threshold > -1:
        assert args.pipeline_model_parallel_size > 1, \
            "--inference-batch-times-seqlen-threshold requires setting --pipeline-model-parallel-size > 1."

    # Optimizer CPU offload check
    if args.optimizer_cpu_offload:
        assert args.use_precision_aware_optimizer, (
            "The optimizer cpu offload must be used in conjunction with `--use-precision-aware-optimizer`, "
            "as the hybrid device optimizer reuses the code path of this flag."
        )

    if args.non_persistent_ckpt_type == "local":
        assert args.non_persistent_local_ckpt_dir is not None, \
            "Tried to use local checkpointing without specifying --local-ckpt-dir!"
    if args.replication:
        assert args.replication_jump is not None, "--replication requires the value of --replication-jump!"
        assert args.non_persistent_ckpt_type == "local", \
            f"--replication requires args.non_persistent_ckpt_type == 'local', but got: {args.non_persistent_ckpt_type}"
    elif args.replication_jump:
        print("Warning: --replication-jump was specified despite not using replication. Ignoring.")
        args.replication_jump = None
