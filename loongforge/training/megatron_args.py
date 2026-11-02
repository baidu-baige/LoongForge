# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Training Arguments Module

This module defines command-line arguments for LoongForge training pipeline,
including model configuration, tokenizer settings, SFT data processing,
video processing, multimodal training, and parallel execution.
"""

import argparse

from loongforge.data import TrainingPhase
from loongforge.data.chat_templates.template import get_support_templates
from loongforge.training.diffusion.args import add_diffusion_args
from loongforge.training.llm.args import add_llm_args
from loongforge.training.vlm.args import add_vlm_args


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
        "--encoder-tensor-model-parallel-size",
        type=int,
        default=None,
        help="Encoder Tensor Model Parallel Size for Heterogeneous TP Training.",
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

    _extend_cuda_graph_scope_choices(parser)

    return parser


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
        default=None,
        help="Path to YAML configuration file containing model and training settings. "
             "Example: /path/to/config.yaml. When provided, other arguments may be "
             "overridden by config file values. Default: None",
    )
    group.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name identifier for the model architecture. Must match a registered model "
             "in loongforge/models/catalog.py. Example: 'llama2-7b', 'qwen2-7b'. Default: None",
    )
    group.add_argument(
        "--training-phase",
        type=str,
        default=TrainingPhase.PRETRAIN,
        choices=[TrainingPhase.PRETRAIN, TrainingPhase.SFT],
        help="Training phase: 'pretrain' for pre-training, 'sft' for supervised fine-tuning. "
             "Default: pretrain",
    )
    group.add_argument(
        "--save-hf",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Save an HF checkpoint at the end of training. Default: false",
    )
    group.add_argument(
        "--save-hf-path",
        type=str,
        default=None,
        help="Path to save the HF model checkpoint. If not specified, saves to "
             "<save>/release_hf_weights/",
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
             "Applied in addition to --freeze-parameters; a parameter is frozen when either matches. "
             "Default: None",
    )
    group.add_argument(
        "--allow-missing-adapter-checkpoint",
        action="store_true",
        help="Allow model loading to proceed even when the adapter checkpoint is missing. "
             "Useful when starting from a base model without adapter weights. "
             "Default: False (a missing adapter checkpoint raises an error)",
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
        help="Directory path to load the EMA checkpoint from. Defaults to ${args.load}/ema. "
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
    """Add arguments for supervised fine-tuning data and objective."""
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
             "padding. Sorting occurs after packing if --packing-sft-data is enabled.",
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


def _add_log_tensor_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add tensor logging and debugging arguments for the llm-inspector hook.

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
             "Only one stage can be selected. Default: forward",
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
             "When False, only prints tensor norms to the log file. Default: False",
    )
    group.add_argument(
        "--save-tensor-dir",
        type=str,
        default="",
        help="[llm-inspector] Directory path to save tensor files. Used when --save-tensor is "
             "enabled. Default: '' (current directory)",
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


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register all Megatron training arguments on *parser*."""
    parser.conflict_handler = "resolve"
    parser = _add_extra_tokenizer_args(parser)
    parser = _add_extra_training_args(parser)
    parser = _add_extra_parallel_args(parser)
    parser = _add_run_args(parser)
    parser = _add_freeze_and_init_args(parser)
    parser = _add_ema_args(parser)
    parser = _add_recipe_args(parser)
    parser = _add_sft_args(parser)
    parser = _add_loop_logging_args(parser)
    parser = _add_log_tensor_args(parser)
    parser = add_llm_args(parser)
    parser = add_vlm_args(parser)
    parser = add_diffusion_args(parser)
    parser = _add_extra_bridge_args(parser)
    return parser
