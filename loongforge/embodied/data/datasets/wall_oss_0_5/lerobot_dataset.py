# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot dataset builder for Wall-OSS-0.5."""

from __future__ import annotations

import logging

import torch
from torch.utils.data import random_split

from loongforge.embodied.data.datasets.lerobot_dataset import LeRobotV3Dataset

logger = logging.getLogger(__name__)


def _wall_oss_split_len(dataset):
    """Wall oss split len."""
    return len(dataset._wall_oss_split_indices)


def _wall_oss_split_index(dataset, idx):
    """Wall oss split index."""
    return dataset._wall_oss_split_indices[int(idx)]


def build_wall_oss_0_5_lerobot_dataset(model_cfg, data_cfg, training_args):
    """Build the small Wall-OSS-0.5 LeRobot dataset used by smoke/parity runs."""
    repo_id = training_args.dataset_path
    if not repo_id:
        raise ValueError("Wall-OSS-0.5 LeRobot dataset requires --dataset-path.")

    episodes = list(data_cfg.episodes) if data_cfg.episodes is not None else None
    if episodes is not None:
        split_idx = int(len(episodes) * data_cfg.train_test_split)
        if split_idx < 1:
            raise ValueError(
                f"train_test_split={data_cfg.train_test_split} applied to "
                f"{len(episodes)} episode(s) yields 0 train episodes."
            )
        train_episodes = episodes[:split_idx]
        test_episodes = episodes[split_idx:]
    else:
        train_episodes = None
        test_episodes = []

    raw_dataset = LeRobotV3Dataset(
        repo_id=repo_id,
        action_horizon=model_cfg.action_horizon,
        episodes=train_episodes,
        video_backend=data_cfg.video_backend or training_args.video_backend,
        delta_timestamps_fn=lambda ds, info, fps: {
            "action": [t / fps for t in range(model_cfg.action_horizon)]
        },
    )
    if test_episodes:
        logger.info("Wall-OSS-0.5 selected test episodes: %s", test_episodes)

    train_split, val_split = random_split(
        raw_dataset,
        [data_cfg.train_test_split, 1.0 - data_cfg.train_test_split],
        torch.Generator().manual_seed(training_args.seed),
    )
    del val_split
    base_len = len(raw_dataset)
    split_indices = list(train_split.indices)

    raw_dataset._wall_oss_dataset_name = str(repo_id)
    raw_dataset._wall_oss_train_episodes = train_episodes
    raw_dataset._wall_oss_base_len = base_len
    raw_dataset._wall_oss_split_indices = split_indices
    raw_dataset._length_fn = _wall_oss_split_len
    raw_dataset._index_map_fn = _wall_oss_split_index
    logger.info(
        "Wall-OSS-0.5 train split: episodes=%s, base_len=%s, train_len=%s",
        train_episodes,
        raw_dataset._wall_oss_base_len,
        len(raw_dataset),
    )
    return raw_dataset
