# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VPP2 data strategy for the pinned LIBERO LeRobot v2.1 assets."""

from __future__ import annotations

from pathlib import Path

from loongforge.embodied.data.datasets.lerobot_dataset import MultiLeRobotV2Dataset


def build_vpp2_lerobot_dataset(model_cfg, data_cfg, training_args):
    """Build the four-suite VPP2 dataset with its 17-frame observation stack."""
    dataset_path = training_args.dataset_path
    if not dataset_path:
        raise ValueError("Must specify --dataset-path")
    roots = [Path(item.strip()) for item in str(dataset_path).split(",") if item.strip()]
    if len(roots) != 4:
        raise ValueError(f"VPP2 expects four LIBERO dataset roots, got {len(roots)}")
    return MultiLeRobotV2Dataset(
        roots=roots,
        action_horizon=model_cfg.action_horizon,
        video_backend=training_args.video_backend,
        observation_delta_indices=data_cfg.observation_delta_indices,
    )
