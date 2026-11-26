# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge dataset builders for VPP2 LIBERO and RoboDojo recipes."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

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


def build_robodojo_dataset(model_cfg, data_cfg, training_args):
    """Build the native EE16 pipeline described by the RoboDojo data config."""
    return instantiate(OmegaConf.create(data_cfg.pipeline))
