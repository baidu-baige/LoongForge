# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLM-only Megatron arguments."""

import argparse

from packaging.version import Version as PkgVersion

from loongforge.engines.megatron import print_rank_0
from loongforge.engines.megatron.utils.runtime import get_transformers_version
from loongforge.models import catalog as constants
from loongforge.models.factory import get_support_model_archs


def add_vlm_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register VLM-only arguments on *parser*."""
    group = parser.add_argument_group(
        title="VLM Data Configuration",
        description="Arguments for vision-language, InternVL, and video data",
    )
    group.add_argument(
        "--sample-type",
        type=str,
        default=None,
        help="Default sample type used for fallback cooker routing when no cooker "
             "matches a sample. For example: 'multi_mix_vqa'. "
             "This allows the dataloader to apply the corresponding cooker "
             "to samples that do not specify a subflavor.",
    )
    group.add_argument(
        "--training-rice-vl-max-answer-length",
        type=int,
        default=4096,
        help="Maximum character length allowed for answers during Rice-VL training. "
             "Answers exceeding this length will be truncated. Default: 4096",
    )
    group.add_argument(
        "--hf-processor-path",
        type=str,
        default=None,
        help="Optional HuggingFace processor path when it differs from the tokenizer path. "
             "Defaults to --hf-tokenizer-path.",
    )
    group.add_argument(
        "--task-encoder",
        type=str,
        default=None,
        help="Registry name for the multimodal task encoder. Responsible for "
             "encoding task-specific inputs (images, video, text). "
             "Examples: 'VLMTaskEncoder', 'KimiTaskEncoder'. Default: None",
    )
    group.add_argument(
        "--data-shuffle-buffer-size",
        type=int,
        default=0,
        help="Size of the Energon sample shuffle buffer applied before cooking and "
             "encoding. For an offline-packed dataset, each sample is one complete "
             "packed sequence, so this randomizes pack order without changing pack "
             "contents. Use this to reduce length correlation when shards are ordered "
             "by sample length. Larger buffers increase cross-sample mixing and worker "
             "memory use. 0 or 1 disables the sample-level buffer (default).",
    )
    group.add_argument(
        "--data-max-samples-per-sequence",
        type=int,
        default=0,
        help="Target size of each shard sequence used by Energon for parallel "
             "iteration. Large worker shard ranges are divided into sequences of roughly "
             "this many samples, creating more sequences that Energon can interleave. "
             "Use this when --data-shuffle-buffer-size cannot cover long ordered runs. "
             "0 leaves Energon's limit unset (default).",
    )
    group.add_argument(
        "--packing-pretrain-data",
        action="store_true",
        help="Pack multiple pretraining samples into single sequence. Default: False",
    )
    group.add_argument(
        "--add-question-in-pretrain",
        action="store_true",
        help="Include question text in pretrain VQA samples. Default: False",
    )
    group.add_argument(
        "--image-resolution",
        type=int,
        default=None,
        help="Target resolution (height/width) for image inputs. Default: None",
    )
    group.add_argument(
        "--min-pixels",
        type=int,
        default=4 * 28 * 28,
        help="Minimum pixel count for image resizing. Images smaller than this "
             "will be upscaled. Default: 3136 (4 * 28 * 28)",
    )
    group.add_argument(
        "--max-pixels",
        type=int,
        default=16384 * 28 * 28,
        help="Maximum pixel count for image resizing. Images larger than this "
             "will be downscaled. Default: 1286144 (16384 * 28 * 28)",
    )
    group.add_argument(
        "--frame-min-pixels",
        type=int,
        default=128 * 28 * 28,
        help="Minimum pixel count per video frame. Default: 100352 (128 * 28 * 28)",
    )
    group.add_argument(
        "--frame-max-pixels",
        type=int,
        default=768 * 28 * 28,
        help="Maximum pixel count per video frame. Default: 602112 (768 * 28 * 28)",
    )
    group.add_argument(
        "--video-max-pixels",
        type=int,
        default=65536 * 28 * 28,
        help="Maximum total pixel count for video. Default: 51380224 (65536 * 28 * 28)",
    )
    group.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to sample from video. Default: 2.0",
    )
    group.add_argument(
        "--fps-min-frames",
        type=int,
        default=4,
        help="Minimum number of frames to sample from video. Default: 4",
    )
    group.add_argument(
        "--fps-max-frames",
        type=int,
        default=768,
        help="Maximum number of frames to sample from video. Default: 768",
    )
    group.add_argument(
        "--energon-pack-algo",
        type=str,
        default="balanced",
        choices=["balanced", "sequential", "sequential_max_images"],
        help="Sample packing algorithm for Energon dataloader: "
             "'balanced': Greedy knapsack approach, balances computational load across GPUs, "
             "'sequential': Fills buffers in order, minimizes sequence disruption, "
             "'sequential_max_images': Sequential with priority on maximizing images per buffer. "
             "Default: balanced",
    )
    group.add_argument(
        "--conv-style",
        type=str,
        default="internvl2_5",
        help="Conversation/prompt style format. Defines how multi-turn conversations "
             "are structured. Default: internvl2_5",
    )
    group.add_argument(
        "--num-images-expected",
        type=int,
        default=48,
        help="Maximum number of images allowed in a single packed sample. Default: 48",
    )
    group.add_argument(
        "--pad2square",
        action="store_true",
        help="Pad rectangular images to square shape with padding. Default: False",
    )
    group.add_argument(
        "--down-sample-ratio",
        type=float,
        default=0.5,
        help="Image down-sampling ratio for resolution reduction. Default: 0.5",
    )
    group.add_argument(
        "--max-buffer-size",
        type=int,
        default=20,
        help="Buffer size for packed dataset construction. Default: 20",
    )
    group.add_argument(
        "--max-packed-tokens",
        type=int,
        default=None,
        help="[DEPRECATED] Ignored; defaults to and is forced equal to "
             "--seq-length. In LoongForge seq_length is the packed-sequence length.",
    )
    group.add_argument(
        "--strict-mode",
        action="store_true",
        help="Pad images to exactly --num-images-expected when enabled. Default: False",
    )
    group.add_argument(
        "--loss-reduction",
        type=str,
        default="square",
        help="Loss reduction method: 'square', 'mean', or 'sum'. Default: square",
    )
    group.add_argument(
        "--group-by-length",
        action="store_true",
        help="Group samples by sequence length for more efficient batching. Default: False",
    )
    group.add_argument(
        "--min-num-frame",
        type=int,
        default=8,
        help="Minimum number of frames to sample from video data. Default: 8",
    )
    group.add_argument(
        "--max-num-frame",
        type=int,
        default=32,
        help="Maximum number of frames to sample from video data. Default: 32",
    )
    group.add_argument(
        "--dynamic-image-size",
        action="store_true",
        help="Enable dynamic high-resolution strategy for images. Default: False",
    )
    group.add_argument(
        "--min-dynamic-patch",
        type=int,
        default=1,
        help="Minimum number of dynamic patches per image. Default: 1",
    )
    group.add_argument(
        "--max-dynamic-patch",
        type=int,
        default=12,
        help="Maximum number of dynamic patches per image. Default: 12",
    )
    group.add_argument(
        "--use_thumbnail",
        action="store_true",
        help="Add thumbnail image alongside dynamic patches. Default: False",
    )
    group.add_argument(
        "--normalize_type",
        type=str,
        default="imagenet",
        help="Image normalization preset. Options: 'imagenet', 'clip', etc. Default: imagenet",
    )
    group.add_argument(
        "--force-image-size",
        type=int,
        default=448,
        help="Force resize images to this size (pixels). Default: 448",
    )
    group.add_argument(
        "--replacement",
        action="store_true",
        help="Restart dataset iteration from beginning when exhausted. Default: False"
    )

    group.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Vision encoder patch size for image tokenization. Default: 14"
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

    group.add_argument(
        "--language-model-type",
        type=str,
        default=None,
        choices=get_support_model_archs(constants.LanguageModelFamilies.names()),
        help="Language model backbone architecture for multimodal models. "
             "Must be from supported language model families. Default: None"
    )

    group = parser.add_argument_group(
        title="VLM Parallel Configuration",
        description="Arguments for encoder parallelism and data load balancing",
    )
    group.add_argument(
        "--encoder-tensor-model-parallel-size",
        type=int,
        default=None,
        help="Encoder Tensor Model Parallel Size for Heterogeneous TP Training.",
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


def _validate_vlm_model_args(args):
    """Check the Qwen3-VL dependency before tokenizer validation."""
    if args.model_family == constants.VisionLanguageModelFamilies.QWEN3_VL:
        required = PkgVersion("4.57.1")
        current = get_transformers_version()
        assert current >= required, (
            f"transformers>={required} required for qwen3-vl, found {current}"
        )


def _validate_extra_multimodal_args(args):
    """Validate multimodal arguments"""
    if args.model_family not in constants.VisionLanguageModelFamilies.names():
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
