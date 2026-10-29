# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Megatron-side training arguments: run, recipe, and data semantics.

These arguments select the run (model, config, phase), describe the training
recipe (freeze, EMA, HF export path, batch and data pipeline), and control
loop logging and tensor debugging. They say nothing about how the backend
executes a step, so they live with the training loop and are registered on the
same parser as the engine arguments.

``parse_megatron_train_args`` is the single entry point used by the training
entrypoint, tools, and tests. Registering both providers on one parser keeps the CLI, the
parsed namespace, and the checkpointed args identical to the engine-only setup.
"""

import argparse
from pathlib import Path
from typing import Optional

from megatron.training.utils import print_rank_0

from loongforge.data import TrainingPhase
from loongforge.data.chat_templates.template import get_support_templates
from loongforge.models.catalog import VisionLanguageModelFamilies
from loongforge.training.diffusion.args import add_diffusion_args
from loongforge.training.llm.args import add_llm_args
from loongforge.training.vlm.args import add_vlm_args


# =============================================================================
# Argument groups
# =============================================================================

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
        help="Save an HF checkpoint at the end of training. "
             "The adapter merge size comes from the YAML peft_config. Default: false",
    )
    group.add_argument(
        "--save-hf-path",
        type=str,
        default=None,
        help="Path to save the HF model checkpoint. If not specified, saves to "
             "<save>/release_hf_weights/",
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
        description="Arguments that shape the training batches and image preprocessing",
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


def _add_log_tensor_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add tensor logging and debugging arguments for the llm-inspector hook.

    The training runner registers the hooks when --enable-log-tensor is set and
    llm-inspector is installed. If the package is missing, the arguments are
    ignored silently.
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

def _add_extra_training_args(parser: argparse.ArgumentParser):
    """Add arguments that configure how the engine executes a step.

    Training-phase, EMA, export, and data-recipe switches live in
    ``loongforge/training/megatron_args.py``. Pipeline sequence-length
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
    return parser


# =============================================================================
# Validation
# =============================================================================

def get_default_sft_dataset_config() -> Optional[str]:
    """Return the default SFT dataset config shipped with the repository."""
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "data"
    for name in ("sft_dataset_config.yaml", "sft_dataset_config.yml"):
        candidate = config_dir / name
        if candidate.exists():
            return str(candidate)
    return None


def _get_wan_packing_cp_alignment(cp_size):
    """Return WAN packing sequence alignment required by CP split."""
    return 2 * cp_size


def _align_wan_packing_seq_length(args):
    """Align WAN packing seq-length before Megatron validation."""
    if getattr(args, "model_name", None) != "wan2-2-i2v":
        return
    if not getattr(args, "packing_sft_data", False):
        return
    cp_size = getattr(args, "context_parallel_size", 1)
    seq_length = getattr(args, "seq_length", None)
    if cp_size <= 1 or seq_length is None:
        return
    alignment = _get_wan_packing_cp_alignment(cp_size)
    aligned_seq_length = ((seq_length + alignment - 1) // alignment) * alignment
    args.seq_length = aligned_seq_length
    if getattr(args, "encoder_seq_length", None) is not None:
        args.encoder_seq_length = aligned_seq_length
    args.max_position_embeddings = max(args.max_position_embeddings, aligned_seq_length)


def _validate_sft_args(args):
    """Validate SFT arguments."""
    if args.training_phase != TrainingPhase.SFT:
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


def _validate_vlm_data_args(args):
    """Validate vision-language data arguments."""
    if args.model_family not in VisionLanguageModelFamilies.names():
        return

    args.variable_seq_lengths = True

    if not (args.packing_pretrain_data or args.packing_sft_data):
        args.packing_buffer_size = None

    # [DEPRECATED] --max-packed-tokens is redundant with --seq-length. In
    # LoongForge seq_length IS the post-concatenation packed-sequence window: it
    # drives every downstream shape / position-embedding / parallel init, so the
    # packed length can never diverge from it. The flag defaults to None and is
    # always resolved to seq_length; only warn when the user passes a value that
    # differs. Guard on seq_length being set (it always is for VLM) to avoid
    # assigning None before Megatron finalizes seq_length.
    if args.seq_length is not None:
        if args.max_packed_tokens is not None and args.max_packed_tokens != args.seq_length:
            print_rank_0(
                "WARNING: --max-packed-tokens is deprecated and ignored; using "
                f"--seq-length ({args.seq_length}) instead "
                f"(you passed {args.max_packed_tokens}).",
                args.rank,
            )
        args.max_packed_tokens = args.seq_length


def validate_training_args(args):
    """Validate training-side Megatron arguments.

    Runs after the engine's model and tokenizer checks and before the engine's
    training and parallel checks and the upstream ``validate_args``. The SFT
    checks ran at this point before the move; the multimodal and wan checks now
    run earlier, but the engine checks they now precede do not read the fields
    they write, so the result does not change.
    """
    _validate_sft_args(args)
    _validate_vlm_data_args(args)
    _align_wan_packing_seq_length(args)


# =============================================================================
# Entry point
# =============================================================================

def parse_megatron_train_args(extra_args_providers=()):
    """Parse Megatron training arguments from the CLI and the model config.

    ``extra_args_providers`` lets tools register their own flags on the same
    parser. The bridge tool uses it; the training entrypoint passes nothing.
    """
    from loongforge.engines.megatron.parser import parse_train_args

    return parse_train_args(
        args_providers=[add_training_args, *extra_args_providers],
        extra_validator=validate_training_args,
    )
