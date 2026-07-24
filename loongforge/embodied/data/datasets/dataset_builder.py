# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dataset build strategies for the generic lerobot datasets.

A *dataset strategy* is a builder ``fn(model_cfg, data_cfg, training_args) ->
Dataset`` selected by ``training_args.dataset_strategy``. This unifies dataset
construction under the single ``--dataset-format lerobot_datasets`` entry: the
``"default"`` strategy reproduces the stock lerobot build, while model-specific
strategies (e.g. ``"motus"`` / ``"fastwam"``) plug their own multi-frame geometry
into the same generic dataset classes via behaviour hooks — no ``dataset_format``
branch per model is required.
"""

from __future__ import annotations

from typing import Callable


def _default_strategy_builder() -> Callable:
    from loongforge.embodied.data.datasets.lerobot_dataset import (
        build_default_lerobot_dataset,
    )

    return build_default_lerobot_dataset


def _fastwam_strategy_builder() -> Callable:
    from loongforge.embodied.data.datasets.fastwam import (
        build_fastwam_lerobot_dataset,
    )

    return build_fastwam_lerobot_dataset


def _lingbot_va_strategy_builder() -> Callable:
    from loongforge.embodied.data.datasets.lingbot_va import build_lingbot_dataset

    return build_lingbot_dataset


def _groot_n1_7_strategy_builder() -> Callable:
    from loongforge.embodied.data.datasets.groot_n1_7 import (
        build_groot_n1_7_lerobot_dataset,
    )

    return build_groot_n1_7_lerobot_dataset


def _cosmos3_droid_strategy_builder() -> Callable:
    from loongforge.embodied.data.datasets.cosmos3 import (
        build_droid_dataset,
    )

    return build_droid_dataset


def _dreamzero_strategy_builder() -> Callable:
    from loongforge.embodied.data.datasets.dreamzero.builder import (
        build_dreamzero_dataset,
    )

    return build_dreamzero_dataset


# name -> lazy loader (imports deferred so lerobot / motus deps only load when used)
_DATASET_STRATEGY_LOADERS: dict[str, Callable[[], Callable]] = {
    "default": _default_strategy_builder,
    "fastwam": _fastwam_strategy_builder,
    "lingbot_va": _lingbot_va_strategy_builder,
    "groot_n1_7": _groot_n1_7_strategy_builder,
    "cosmos3_droid": _cosmos3_droid_strategy_builder,
    "dreamzero": _dreamzero_strategy_builder,
}


def build_dataset_by_strategy(model_cfg, data_cfg, training_args):
    """Build a lerobot dataset via the strategy named by ``training_args.dataset_strategy``.

    Falls back to ``"default"`` when the field is absent, empty, or names a
    dataset that has no dedicated strategy.
    """
    name = training_args.dataset_strategy or "default"
    loader = _DATASET_STRATEGY_LOADERS.get(name, _default_strategy_builder)
    return loader()(model_cfg, data_cfg, training_args)
