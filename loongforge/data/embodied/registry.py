# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dataset formats and build strategies for embodied data."""

from __future__ import annotations

import importlib
from typing import Dict

# dataset_strategy name -> "module:callable" string.  Resolved lazily so that
# lerobot / model-specific deps only load when actually selected.  The
# ``"default"`` strategy is the stock lerobot build; model-specific strategies
# plug custom multi-frame geometry into the same generic dataset classes.
DATASET_STRATEGIES: Dict[str, str] = {
    "default": "datasets.lerobot_dataset:build_default_lerobot_dataset",
    "vlda": "vlda:build_vlda_dataset",
    "fastwam": "fastwam.fastwam_dataset:build_fastwam_lerobot_dataset",
    "lingbot_va": "lingbot_va.latent_lerobot_dataset:build_lingbot_dataset",
    "lingbot_vla_v2": "lingbot_vla_v2.lingbot_vla_v2_dataset:build_lingbot_vla_v2_dataset",
    "gr00t_n1_7": "gr00t_n1_7.gr00t_n1_7_dataset:build_groot_n1_7_lerobot_dataset",
    "cosmos3_droid": "cosmos3.droid_lerobot_dataset:build_droid_dataset",
    "dreamzero": "dreamzero.builder:build_dreamzero_dataset",
    "wall_oss_0_5": "wall_oss_0_5.lerobot_dataset:build_wall_oss_0_5_lerobot_dataset",
}

# --dataset-format -> "module:callable" (relative to ``loongforge.data.embodied``).
# "lerobot_datasets" further dispatches on --dataset-strategy through DATASET_STRATEGIES.
DATASET_FORMATS: Dict[str, str] = {
    "lerobot_datasets": "registry:build_dataset_by_strategy",
    "hdf5_datasets": "datasets.hdf5_dataset:build_hdf5_dataset",
    "dummy_datasets": "datasets.dummy_dataset:build_dummy_dataset",
}


def build_dataset_by_strategy(model_cfg, data_cfg, training_args):
    """Build a lerobot dataset via the strategy named by ``training_args.dataset_strategy``.

    Falls back to ``"default"`` when the field is absent, empty, or names a
    dataset that has no dedicated strategy.
    """
    name = training_args.dataset_strategy or "default"
    entry = DATASET_STRATEGIES.get(name, DATASET_STRATEGIES["default"])
    module_rel, func_name = entry.split(":")
    mod = importlib.import_module(f"loongforge.data.embodied.{module_rel}")
    return getattr(mod, func_name)(model_cfg, data_cfg, training_args)


def build_dataset(model_cfg, data_cfg, training_args):
    """Build the dataset named by ``training_args.dataset_format``.

    Lerobot datasets dispatch further on ``training_args.dataset_strategy``
    through ``DATASET_STRATEGIES``.
    """
    name = training_args.dataset_format
    if name not in DATASET_FORMATS:
        raise ValueError(
            f"Unknown dataset_format: '{name}'. Supported: {sorted(DATASET_FORMATS)}"
        )
    module_rel, func_name = DATASET_FORMATS[name].split(":")
    mod = importlib.import_module(f"loongforge.data.embodied.{module_rel}")
    return getattr(mod, func_name)(model_cfg, data_cfg, training_args)
