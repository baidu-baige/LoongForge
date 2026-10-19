# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Megatron-side training arguments: task and data semantics.

These arguments describe the training recipe and the data pipeline. They say
nothing about how the backend executes a step, so they live with the training
loop and are registered on the same parser as the engine arguments.

``parse_megatron_train_args`` is the single entry point used by the runner,
tools, and tests. Registering both providers on one parser keeps the CLI, the
parsed namespace, and the checkpointed args identical to the engine-only setup.
"""

import argparse
from pathlib import Path
from typing import Optional

from megatron.training.utils import print_rank_0

from loongforge import constants


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
    group.add_argument(
        "--sample-type",
        type=str,
        default=None,
        help="Default sample type used for fallback cooker routing when no cooker "
             "matches a sample. For example: 'multi_mix_vqa'. "
             "This allows the dataloader to apply the corresponding cooker "
             "to samples that do not specify a subflavor.",
    )
    return parser


def _add_vlm_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments for vision-language and InternVL data processing."""
    group = parser.add_argument_group(
        title="Multimodal Data Configuration",
        description="Arguments for vision-language data and InternVL preprocessing",
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

    # InternVL preprocessing
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
        "--patch-size",
        type=int,
        default=14,
        help="Vision encoder patch size for image tokenization. Default: 14",
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

    # Diffusion timestep sampling
    group.add_argument(
        "--max-timestep-boundary",
        type=float,
        default=1.0,
        help="Maximum diffusion timestep boundary (0-1 range) for DiT models. Default: 1.0",
    )
    group.add_argument(
        "--min-timestep-boundary",
        type=float,
        default=0.0,
        help="Minimum diffusion timestep boundary (0-1 range) for DiT models. Default: 0.0",
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


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register all training-side Megatron arguments on *parser*."""
    parser.conflict_handler = "resolve"
    parser = _add_sft_args(parser)
    parser = _add_vlm_data_args(parser)
    parser = _add_loop_logging_args(parser)
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
    if args.training_phase != constants.TrainingPhase.SFT:
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

def parse_megatron_train_args():
    """Parse Megatron training arguments from the CLI and the model config."""
    from loongforge.engines.megatron.parser import parse_train_args

    return parse_train_args(
        extra_args_provider=add_training_args,
        extra_validator=validate_training_args,
    )
