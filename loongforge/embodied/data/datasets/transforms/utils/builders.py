# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for building configurable per-sample transforms."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from loongforge.embodied.data.datasets.transforms.utils.action_transform import ActionTransform
from loongforge.embodied.data.datasets.transforms.utils.image_transform import ImageTransform


def convert_stats(stats_raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
    """Convert dataset stats (torch.Tensor/list) to numpy for Normalizer."""
    if stats_raw is None:
        return None

    stats = {}
    for key, value in stats_raw.items():
        if isinstance(value, torch.Tensor):
            stats[key] = value.cpu().numpy()
        elif isinstance(value, np.ndarray):
            stats[key] = value
        elif isinstance(value, (list, tuple)):
            stats[key] = np.array(value)
        else:
            stats[key] = value
    return stats


def build_image_transform(
    data_cfg: Any,
    image_keys: list[str],
    image_size: int | tuple[int, int],
) -> ImageTransform | None:
    """Build the standard ImageTransform from config, or None when disabled."""
    if not image_keys or not data_cfg.use_image_transform:
        return None

    return ImageTransform(
        apply_to=image_keys,
        image_size=image_size,
        resize_strategy=data_cfg.image_resize_strategy,
        normalize_mode=data_cfg.image_normalize_mode,
    )


def build_action_transform(
    data_cfg: Any,
    dataset_stats: Optional[Dict[str, Any]],
    action_horizon: int | None,
    max_action_dim: int | None,
    normalization_mode: str,
) -> ActionTransform | None:
    """Build the standard ActionTransform from config, or None when disabled."""
    if not data_cfg.use_action_transform:
        return None

    action_stats = (
        convert_stats(dataset_stats.get("action"))
        if dataset_stats and data_cfg.action_use_statistics
        else None
    )

    return ActionTransform(
        apply_to=data_cfg.action_apply_to,
        action_horizon=data_cfg.action_transform_horizon or action_horizon,
        max_action_dim=data_cfg.action_transform_max_action_dim or max_action_dim,
        normalization_mode=normalization_mode,
        statistics=action_stats,
        padding_strategy=data_cfg.action_padding_strategy,
    )
