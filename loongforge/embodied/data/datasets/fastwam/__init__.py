# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM multi-frame-observation dataset strategy: behaviour hook for the
generic lerobot datasets."""

from loongforge.embodied.data.datasets.fastwam.fastwam_dataset import (
    build_fastwam_lerobot_dataset,
    fastwam_delta_timestamps,
)

__all__ = [
    "build_fastwam_lerobot_dataset",
    "fastwam_delta_timestamps",
]
