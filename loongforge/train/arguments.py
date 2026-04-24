# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Training Arguments Module

This module defines command-line arguments for LoongForge training pipeline,
including model configuration, tokenizer settings, SFT data processing,
video processing, multimodal training, and parallel execution.
"""

import argparse

from loongforge.data import get_support_templates
from loongforge.models import get_support_model_archs
from loongforge.utils import constants


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
    parser = _add_extra_sft_args(parser)
    parser = _add_extra_video_args(parser)
    parser = _add_extra_training_args(parser)
    parser = _add_extra_multimodal_args(parser)
    parser = _add_extra_parallel_args(parser)
    # Debug and logging arguments
    parser = _add_log_tensor_args(parser)
    # Rice-VL specific arguments
    parser = _add_extra_training_rice_vl_args(parser)

    return parser


# =============================================================================
# Tensor Logging Arguments (llm-inspector)
# =============================================================================

def _add_log_tensor_args(parser):
    """Add arguments for tensor logging and debugging via llm-inspector.

    These arguments interface with the llm-inspector library for debugging
    model training. When enabled (via --enable-log-tensor), the system will
    register hooks to capture tensor statistics during training.


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
                    "Requires llm-inspector package to be installed."
    )

    group.add_argument(
        "--enable-log-tensor",
        action="store_true",
        help="[llm-inspector] Enable tensor logging for debugging. When enabled, tensor statistics "
             "will be traced during training using the llm-inspector library. "
             "Requires llm-inspector to be installed. Default: False"
    )
    
    group.add_argument(
        "--log-tensor-name-pattern",
        type=str,
        default=None,
        help="[llm-inspector] Regex pattern to filter module names for tensor logging. "
             "When None (default), logs all modules. "
             "Example: '.*attention.*' to log only attention modules. Default: None"
    )
    
    group.add_argument(
        "--log-tensor-stage",
        type=str,
        default="forward",
        choices=["init", "forward", "backward"],
        help="[llm-inspector] Training stage at which to log tensors. "
             "'init': model initialization, 'forward': forward pass, 'backward': backward pass. "
             "Multiple stages can be specified comma-separated, e.g., 'forward,backward'. "
             "Default: forward"
    )
    
    group.add_argument(
        "--log-tensor-iter-pattern",
        type=str,
        default=None,
        help="[llm-inspector] Comma-separated iteration indices at which to log tensors. "
             "Example: '8,15,20' logs tensors at iterations 8, 15, and 20. "
             "When None, logs at all iterations. Default: None"
    )
    
    group.add_argument(
        "--log-tensor-mbs-pattern",
        type=str,
        default=None,
        help="[llm-inspector] Comma-separated micro-batch indices at which to log tensors. "
             "Example: '0,2,4' logs tensors for micro-batches 0, 2, and 4. "
             "Default: None"
    )
    
    group.add_argument(
        "--log-tensor-layer-pattern",
        type=str,
        default=None,
        help="[llm-inspector] Comma-separated layer indices at which to log tensors. "
             "Example: '0,5,10' logs tensors for layers 0, 5, and 10. "
             "Default: None"
    )
    
    group.add_argument(
        "--log-tensor-rank",
        type=str,
        default="0",
        help="[llm-inspector] Comma-separated GPU ranks at which to log tensors. "
             "Example: '0,1,2,4' logs tensors on ranks 0, 1, 2, and 4. "
             "Default: 0"
    )

    # Tensor saving options
    group.add_argument(
        "--save-tensor",
        action="store_true",
        help="[llm-inspector] Save logged tensors to disk files for offline analysis. "
             "When False, only prints tensor norms to log file. Default: False"
    )
    
    group.add_argument(
        "--save-tensor-dir",
        type=str,
        default="",
        help="[llm-inspector] Directory path to save tensor files. Required when --save-tensor is enabled. "
             "Default: '' (current directory)"
    )

    group.add_argument(
        "--random-fallback-cpu",
        action="store_true",
        help="Generate random numbers on CPU then move to target device. "
             "Useful for hardware that has issues with on-device random number generation. "
             "Default: False"
    )

    return parser


# =============================================================================
# Rice-VL Training Arguments
# =============================================================================

def _add_extra_training_rice_vl_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments specific to Rice-VL model training.
    
    Rice-VL is a vision-language model that requires special handling
    for answer length during training.
    """
    group = parser.add_argument_group(
        title='Training Rice-VL',
        description='Arguments specific to Rice-VL vision-language model training'
    )

    group.add_argument(
        '--training-rice-vl-max-answer-length',
        type=int,
        default=4096,
        help="Maximum character length allowed for answers during Rice-VL training. "
             "Answers exceeding this length will be truncated. Default: 4096"
    )
    return parser


# =============================================================================
# Model Arguments
# =============================================================================

def _add_extra_model_args(parser: argparse.ArgumentParser):
    """Add arguments for model configuration and loading.
    
    These arguments control model architecture selection, parameter freezing,
    and checkpoint handling.
    """
    group = parser.add_argument_group(
        title="Model Configuration",
        description="Arguments for model architecture and parameter management"
    )

    group.add_argument(
        "--config-file",
        type=str,
        required=False,
        help="Path to YAML configuration file containing model and training settings. "
             "Example: /path/to/config.yaml. When provided, other arguments may be "
             "overridden by config file values. Default: None"
    )

    group.add_argument(
        "--model-name",
        type=str,
        default=None,
        required=False,
        help="Name identifier for the model architecture. Must match a registered model "
             "in the MODEL_CONFIG_REGISTRY. Example: 'llama2-7b', 'qwen2-7b'. Default: None"
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
        '--freeze-parameters',
        type=str,
        nargs="*",
        default=[],
        help="List of parameter name prefixes to freeze during training. "
             "Frozen parameters will not be updated by the optimizer. "
             "Example: --freeze-parameters 'embed' 'lm_head' freezes embedding and output layers. "
             "Default: [] (no freezing)"
    )

    group.add_argument(
        '--freeze-parameters-regex',
        type=str,
        default=None,
        help="Regular expression pattern to match parameter names to freeze. "
             "Example: '.*bias.*' freezes all bias parameters. "
             "Takes precedence over --freeze-parameters if both are specified. "
             "Default: None"
    )

    group.add_argument(
        "--allow-missing-adapter-checkpoint",
        action="store_true",
        help="Allow model loading to proceed even when adapter checkpoint is missing. "
             "Useful when starting training from a base model without adapter weights. "
             "Default: False (missing adapter checkpoint raises error)"
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

    group.add_argument(
        "--padded-vocab-size",
        type=int,
        default=None,
        help="Explicitly specify padded vocabulary size. Used for models with custom "
             "vocabulary padding requirements. Default: None"
    )

    group.add_argument(
        "--task-encoder",
        type=str,
        default=None,
        help="Task encoder class for multimodal data pipeline. Responsible for "
             "encoding task-specific inputs (images, video, text). "
             "Examples: 'VLMTaskEncoder', 'InternVLTaskEncoder'. Default: None"
    )
    return parser


# =============================================================================
# SFT (Supervised Fine-Tuning) Arguments
# =============================================================================

def _add_extra_sft_args(parser: argparse.ArgumentParser):
    """Add arguments for supervised fine-tuning data configuration.
    
    These arguments control dataset selection, preprocessing, packing,
    and training behavior for SFT tasks.
    """
    group = parser.add_argument_group(
        title="SFT Data Configuration",
        description="Arguments for supervised fine-tuning data processing"
    )
    
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
        "--sft-dataset-config",
        type=str,
        default=None,
        help="Path to YAML file containing dataset configurations. "
             "Defines dataset formats and processing rules. "
             "Default: configs/data/sft_dataset_config.yaml"
    )

    group.add_argument(
        "--sft-dataset",
        nargs="*",
        default=None,
        help="List of dataset names for combined train/valid/test splits. "
             "Names must be defined in --sft-dataset-config. "
             "Examples: 'dataset1' or 'dataset1 dataset2'. "
             "Mutually exclusive with --sft-train-dataset/--sft-valid-dataset/--sft-test-dataset. "
             "Default: None"
    )

    group.add_argument(
        "--sft-train-dataset",
        nargs="*",
        default=None,
        help="List of training dataset names. Used with --train-data-path. "
             "Follows same naming rules as --sft-dataset. Default: None"
    )

    group.add_argument(
        "--sft-valid-dataset",
        nargs="*",
        default=None,
        help="List of validation dataset names. Used with --valid-data-path. "
             "Follows same naming rules as --sft-dataset. Default: None"
    )

    group.add_argument(
        "--sft-test-dataset",
        nargs="*",
        default=None,
        help="List of test dataset names. Used with --test-data-path. "
             "Follows same naming rules as --sft-dataset. Default: None"
    )

    group.add_argument(
        "--sft-sort-batch",
        action="store_true",
        help="Sort dataset samples by length (smallest to largest) for more efficient "
             "padding. Sorting occurs after packing if --packing-sft-data is enabled. "
    )

    group.add_argument(
        "--sft-data-streaming",
        action="store_true",
        help="Enable streaming mode for large datasets that don't fit in memory. "
             "Data is loaded on-demand during training. Default: False"
    )

    group.add_argument(
        "--streaming-buffer-size",
        type=int,
        default=16384,
        help="Buffer size for random sampling in streaming mode. Larger buffers "
             "provide more randomization but use more memory. Default: 16384"
    )

    group.add_argument(
        "--sft-data-mix-strategy",
        type=str,
        choices=["concat", "interleave_under", "interleave_over"],
        default="concat",
        help=(
            "Strategy for mixing multiple SFT datasets: "
            "'concat': concatenate datasets sequentially; "
            "'interleave_under': interleave datasets and stop when the shortest dataset is exhausted; "
            "'interleave_over': interleave datasets until the longest dataset is exhausted."
        ),
    )

    group.add_argument(
        "--sft-num-preprocess-workers",
        type=int,
        default=None,
        help="Number of worker processes for data preprocessing. Only applies to "
             "non-streaming mode. More workers speed up preprocessing but use more memory. "
             "Default: None (auto-detect)"
    )

    group.add_argument(
        "--train-on-prompt",
        action="store_true",
        help="Include prompt tokens in loss computation. By default, loss is computed "
             "only on response tokens. Default: False"
    )

    group.add_argument(
        "--history-mask-loss",
        action="store_true",
        help="Compute loss only on the last turn response in multi-turn conversations, "
             "masking the loss for tokens from previous turns. Default: False"
    )

    group.add_argument(
        "--is-tokenized-data",
        action="store_true",
        help="Indicate that input data is already tokenized. Skips tokenization step "
             "in data processing pipeline. Default: False"
    )

    group.add_argument(
        "--packing-sft-data",
        action="store_true",
        help="Pack multiple short sequences into a single training sample to improve "
             "GPU utilization. Default: False"
    )

    group.add_argument(
        "--enable-discard-sample",
        action="store_true",
        help="Discard samples that exceed --seq-length instead of truncating. "
             "Useful for ensuring consistent sequence lengths. Default: False"
    )

    group.add_argument(
        "--packing-buffer-size",
        type=int,
        default=10000,
        help="Size of the sample buffer used for sequence packing. "
            "Samples are first loaded into this buffer and then packed into sequences "
            "by the packing algorithm. Effective only when --packing-sft-data is enabled."
        )

    group.add_argument(
        "--use-fixed-seq-lengths",
        action="store_true",
        help="Pad all sequences to exactly --seq-length. Currently only supported "
             "for language models. Default: False"
    )

    group.add_argument(
        "--sample-type",
        type=str,
        default=None,
        help="Default sample type used for fallback cooker routing when no cooker "
            "matches a sample. For example: 'multi_mix_vqa'. "
            "This allows the dataloader to apply the corresponding cooker "
            "to samples that do not specify a subflavor."
    )
    return parser


# =============================================================================
# Video Processing Arguments
# =============================================================================

def _add_extra_video_args(parser):
    """Add arguments for video and vision task configuration.
    
    These arguments control video latent processing, frame handling,
    and InternVL-specific vision settings.
    """
    group = parser.add_argument_group(
        title="Video & Vision Configuration",
        description="Arguments for video processing and vision model configuration"
    )

    # -------------------------------------------------------------------------
    # Latent Space Configuration
    # -------------------------------------------------------------------------
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
        "--caption-channels",
        type=int,
        default=None,
        help="Number of channels for caption/text embeddings in video models. Default: None"
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

    group.add_argument(
        "--max-text-length",
        type=int,
        default=None,
        help="Maximum text/caption token length. Default: None"
    )

    group.add_argument(
        "--max-video-length",
        type=int,
        default=32760,
        help="Maximum video token length. Default: 32760"
    )

    group.add_argument(
        "--max-image-length",
        type=int,
        default=None,
        help="Maximum image token length. Default: None"
    )

    group.add_argument(
        "--max-timestep-boundary",
        type=float,
        default=1.0,
        help="Maximum diffusion timestep boundary (0-1 range) for DiT models. "
             "Default: 1.0"
    )

    group.add_argument(
        "--min-timestep-boundary",
        type=float,
        default=0.0,
        help="Minimum diffusion timestep boundary (0-1 range) for DiT models. "
             "Default: 0.0"
    )

    group.add_argument(
        '--dataset-metadata-path',
        type=str,
        default=None,
        help="Path to dataset metadata file containing video/image information. "
             "Default: None"
    )

    # -------------------------------------------------------------------------
    # InternVL-specific Arguments
    # -------------------------------------------------------------------------
    group.add_argument(
        "--loss-reduction-all-gather",
        action="store_true",
        help="Gather losses from all GPUs during loss reduction. Default: False"
    )

    group.add_argument(
        "--conv-style",
        type=str,
        default="internvl2_5",
        help="Conversation/prompt style format. Defines how multi-turn conversations "
             "are structured. Default: internvl2_5"
    )

    group.add_argument(
        "--force-image-size",
        type=int,
        default=448,
        help="Force resize images to this size (pixels). Default: 448"
    )

    group.add_argument(
        "--num-images-expected",
        type=int,
        default=48,
        help="Maximum number of images allowed in a single packed sample. Default: 48"
    )

    group.add_argument(
        "--pad2square",
        action="store_true",
        help="Pad rectangular images to square shape with padding. Default: False"
    )

    group.add_argument(
        "--down-sample-ratio",
        type=float,
        default=0.5,
        help="Image down-sampling ratio for resolution reduction. Default: 0.5"
    )

    group.add_argument(
        "--max-buffer-size",
        type=int,
        default=20,
        help="Buffer size for packed dataset construction. Default: 20"
    )

    group.add_argument(
        "--max-packed-tokens",
        type=int,
        default=8192,
        help="Target token length per packed sample. Default: 8192"
    )

    group.add_argument(
        "--strict-mode",
        action="store_true",
        help="Pad images to exactly --num-images-expected when enabled. Default: False"
    )

    group.add_argument(
        "--replacement",
        action="store_true",
        help="Restart dataset iteration from beginning when exhausted. Default: False"
    )

    group.add_argument(
        "--loss-reduction",
        type=str,
        default="square",
        help="Loss reduction method: 'square', 'mean', or 'sum'. Default: square"
    )

    group.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Vision encoder patch size for image tokenization. Default: 14"
    )

    group.add_argument(
        "--group-by-length",
        action="store_true",
        help="Group samples by sequence length for more efficient batching. Default: False"
    )

    group.add_argument(
        "--min-num-frame",
        type=int,
        default=8,
        help="Minimum number of frames to sample from video data. Default: 8"
    )

    group.add_argument(
        "--max-num-frame",
        type=int,
        default=32,
        help="Maximum number of frames to sample from video data. Default: 32"
    )

    group.add_argument(
        "--dynamic-image-size",
        action="store_true",
        help="Enable dynamic high-resolution strategy for images. Default: False"
    )

    group.add_argument(
        "--min-dynamic-patch",
        type=int,
        default=1,
        help="Minimum number of dynamic patches per image. Default: 1"
    )

    group.add_argument(
        "--max-dynamic-patch",
        type=int,
        default=12,
        help="Maximum number of dynamic patches per image. Default: 12"
    )

    group.add_argument(
        "--use_thumbnail",
        action="store_true",
        help="Add thumbnail image alongside dynamic patches. Default: False"
    )

    group.add_argument(
        "--normalize_type",
        type=str,
        default="imagenet",
        help="Image normalization preset. Options: 'imagenet', 'clip', etc. Default: imagenet"
    )

    group.add_argument(
        "--use-packed-ds",
        action="store_true",
        help="[DEPRECATED] Enable packed dataset mode for efficient multi-image training. Default: False"
    )

    group.add_argument(
        "--save-dataset-state",
        action="store_true",
        help="[DEPRECATED] Save dataset state for resumable training. Default: False"
    )

    group.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker in dataloader. Default: 2"
    )

    return parser


# =============================================================================
# Training Arguments
# =============================================================================

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
        "--training-phase",
        type=str,
        default=constants.TrainingPhase.PRETRAIN,
        choices=[constants.TrainingPhase.PRETRAIN, constants.TrainingPhase.SFT],
        help="Training phase: 'pretrain' for pre-training, 'sft' for supervised fine-tuning. "
             "Default: pretrain"
    )
    

    group.add_argument(
        "--use-dsa-fused",
        action="store_true",
        help="Force use of Omni fused DSA implementation for DeepSeek models. "
             "Default: False"
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
        "--no-detail-log",
        action="store_false",
        dest="log_detail",
        help="Disable detailed timing logs during training. Default: enabled"
    )

    group.add_argument(
        "--detail-log-interval",
        type=int,
        default=20,
        help="Interval (iterations) between detailed timing log reports. "
             "Only effective when timing-log-level is 0. Default: 20"
    )

    group.add_argument(
        "--variable-seq-lengths",
        action="store_true",
        help="[DEPRECATED] This flag is ignored. Variable sequence length support "
             "is now automatic. Default: False"
    )

    # EMA (Exponential Moving Average) arguments
    group.add_argument(
        "--enable-ema",
        action="store_true",
        help="Enable Exponential Moving Average (EMA) of model parameters. "
             "EMA provides smoothed parameter estimates for more stable inference. "
             "Default: False"
    )

    group.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="Decay rate for EMA parameter updates. Higher values = slower update. "
             "Default: 0.9999"
    )

    group.add_argument(
        "--save-ema",
        type=str,
        default=None,
        help="Directory path to save EMA checkpoints. Defaults to ${args.save}/ema. "
             "Default: None"
    )

    group.add_argument(
        "--load-ema",
        type=str,
        default=None,
        help="Directory path to load EMA checkpoint from. Defaults to ${args.load}/ema. "
             "Default: None"
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
        "--log-memory-stats",
        action="store_true",
        default=False,
        help="Log GPU memory statistics (allocated/peak) during training. "
             "Default: False"
    )

    group.add_argument(
        "--save-hf",
        type=str,
        default='false',
        choices=['true', 'false'],
        help="Save HF checkpoint at the end of training. Choices: [true, false]. Default: true."
    )

    group.add_argument(
        "--save-hf-path",
        type=str,
        default=None,
        help="Path to save the HF model checkpoint. If not specified, will save to <save>/release_hf_weights/"
    )

    group.add_argument('--lora-alpha', type=int, help="Lora alpha for LoRA fine tuning.")
    group.add_argument('--lora-dim', type=int, help="Lora dim for LoRA fine tuning.")

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
        "--force-all-weight-decay",
        action="store_false",
        default=None,
        help="Force all parameters into weight-decay group, overriding default "
             "exclusions for bias and LayerNorm. Default: None (use model defaults)"
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
# Multimodal Arguments
# =============================================================================

def _add_extra_multimodal_args(parser):
    """Add arguments for multimodal model configuration.
    
    These arguments control vision-language model settings, image/video
    processing parameters, and data packing strategies.
    """
    group = parser.add_argument_group(
        title="Multimodal Configuration",
        description="Arguments for vision-language and multimodal models"
    )
    
    group.add_argument(
        "--language-model-type",
        type=str,
        default=None,
        choices=get_support_model_archs(constants.LanguageModelFamilies.names()),
        help="Language model backbone architecture for multimodal models. "
             "Must be from supported language model families. Default: None"
    )

    group.add_argument(
        "--trainable-modules",
        default=["all"],
        nargs="*",
        help="List of modules to train (unfrozen). Options: 'all', 'language_model', "
             "'adapter', 'vision_model', 'language_expert_linear', 'vision_expert_linear'. "
             "Default: ['all']"
    )

    group.add_argument(
        "--dataloader-save",
        type=str,
        default=None,
        help="Path to save Energon dataloader state for resumable training. Default: None"
    )

    group.add_argument(
        "--packing-pretrain-data",
        action="store_true",
        help="Pack multiple pretraining samples into single sequence. Default: False"
    )

    group.add_argument(
        "--add-question-in-pretrain",
        action="store_true",
        help="Include question text in pretrain VQA samples. Default: False"
    )

    # Qwen2-VL specific image processing arguments
    group.add_argument(
        "--image-resolution",
        type=int,
        default=None,
        help="Target resolution (height/width) for image inputs. Default: None"
    )

    group.add_argument(
        "--min-pixels",
        type=int,
        default=4 * 28 * 28,
        help="Minimum pixel count for image resizing. Images smaller than this "
             "will be upscaled. Default: 3136 (4 * 28 * 28)"
    )

    group.add_argument(
        "--max-pixels",
        type=int,
        default=16384 * 28 * 28,
        help="Maximum pixel count for image resizing. Images larger than this "
             "will be downscaled. Default: 1286144 (16384 * 28 * 28)"
    )

    group.add_argument(
        "--frame-min-pixels",
        type=int,
        default=128 * 28 * 28,
        help="Minimum pixel count per video frame. Default: 100352 (128 * 28 * 28)"
    )

    group.add_argument(
        "--frame-max-pixels",
        type=int,
        default=768 * 28 * 28,
        help="Maximum pixel count per video frame. Default: 602112 (768 * 28 * 28)"
    )

    group.add_argument(
        "--video-max-pixels",
        type=int,
        default=65536 * 28 * 28,
        help="Maximum total pixel count for video. Default: 51380224 (65536 * 28 * 28)"
    )

    # Frame sampling arguments
    group.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to sample from video. Default: 2.0"
    )

    group.add_argument(
        "--fps-min-frames",
        type=int,
        default=4,
        help="Minimum number of frames to sample from video. Default: 4"
    )

    group.add_argument(
        "--fps-max-frames",
        type=int,
        default=768,
        help="Maximum number of frames to sample from video. Default: 768"
    )

    group.add_argument(
        '--energon-pack-algo',
        type=str,
        default="balanced",
        choices=["balanced", "sequential", "sequential_max_images"],
        help="Sample packing algorithm for Energon dataloader: "
             "'balanced': Greedy knapsack approach, balances computational load across GPUs, "
             "'sequential': Fills buffers in order, minimizes sequence disruption, "
             "'sequential_max_images': Sequential with priority on maximizing images per buffer. "
             "Default: balanced"
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
             "Set to None to disable the constraint. Default: 1.2"
    )

    group.add_argument(
        '--dp-balance-max-len-ratio-vit',
        type=float,
        default=None,
        help="Maximum sequence length ratio for ViT-level DP load balancing. "
             "Limits each DP rank's maximum sequence length to (avg_len * ratio). "
             "Set to None to disable the constraint (default for ViT mode). Default: None"
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
