# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLM-only Megatron arguments."""

import argparse


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
        "--ernie-chat-template",
        type=str,
        default="ernie_vl",
        choices=["ernie_vl", "ernie_vl_thinking"],
        help="Chat template used by ErnieTaskEncoder. Default: ernie_vl",
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
        "--use-thumbnail",
        "--use_thumbnail",
        dest="use_thumbnail",
        action="store_true",
        help="Add thumbnail image alongside dynamic patches. Default: False",
    )
    group.add_argument(
        "--normalize-type",
        "--normalize_type",
        dest="normalize_type",
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
    return parser
