# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion-only Megatron arguments."""

import argparse


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
        nargs=3,
        type=int,
        default=(1, 1, 1),
        metavar=("T", "H", "W"),
        help="Patch size (time, height, width) for latent space tokenization, given as three "
             "space-separated integers. Default: (1, 1, 1)"
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
