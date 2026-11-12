# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion-only Megatron arguments."""

import argparse

from loongforge.models import catalog


def add_diffusion_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register diffusion-only arguments on *parser*."""
    group = parser.add_argument_group(
        title="Diffusion Configuration",
        description="Arguments for diffusion timestep sampling and latent shapes",
    )
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
    group.add_argument(
        "--caption-channels",
        type=int,
        default=None,
        help="Number of channels for caption/text embeddings in video models. Default: None"
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
        '--dataset-metadata-path',
        type=str,
        default=None,
        help="Path to dataset metadata file containing video/image information. "
             "Default: None"
    )

    return parser


def _validata_extra_custom_args(args):
    """Validate multimodal arguments"""
    if args.model_family not in catalog.DiffusionModelFamilies.names():
        return

    # make text length divisible by cp size
    if args.max_text_length is not None and args.context_parallel_size > 1:
        while (args.max_text_length % args.context_parallel_size) != 0:
            args.max_text_length += 1


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
