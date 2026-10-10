# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM multi-frame-observation dataset strategy: behaviour hook for the
generic lerobot datasets.

FastWAM plugs its geometry into :class:`LeRobotV3Dataset` through the
``delta_timestamps_fn`` hook instead of branching inside the dataset class. The
hook adds, for every video key, a multi-frame observation stack sampled at the
``observation_delta_indices`` offsets (in addition to the standard pi05 action
chunk). The base dataset stays entirely model-agnostic.

``build_fastwam_lerobot_dataset`` wires the hook onto a single
``LeRobotV3Dataset``, deriving ``repo_id`` from ``--dataset-path`` and reading
``observation_delta_indices`` / ``action_horizon`` from the FastWAM configs.
"""

from __future__ import annotations

from typing import Any, Dict, List

from loongforge.data.embodied.datasets.lerobot_dataset import (
    LeRobotV3Dataset,
    build_default_lerobot_dataset,
)


def _build_fastwam_delta_timestamps(
    action_horizon: int,
    fps: int,
    observation_delta_indices: List[int],
    image_keys: List[str],
) -> Dict[str, list]:
    """Build delta_timestamps with a pi05 action chunk plus multi-frame observations.

    ``action`` gets the standard ``[i / fps for i in range(action_horizon)]``
    chunk; every video key additionally gets multi-frame observation timestamps
    at the ``observation_delta_indices`` offsets.
    """
    timestamps: Dict[str, list] = {"action": [i / fps for i in range(action_horizon)]}
    for key in image_keys:
        timestamps[key] = [i / fps for i in observation_delta_indices]
    return timestamps


def fastwam_delta_timestamps(dataset: LeRobotV3Dataset, info: Dict[str, Any], fps: int) -> Dict[str, list]:
    """``delta_timestamps_fn`` hook: add a multi-frame observation stack per video key.

    Runs before ``LeRobotDataset.__init__``. Discovers the video keys from
    ``info``, reads ``action_horizon`` off the dataset and
    ``observation_delta_indices`` from the strategy kwargs stashed on it.
    """
    action_horizon = int(dataset._action_horizon)
    observation_delta_indices = list(dataset._strategy_kwargs["observation_delta_indices"])

    image_keys = [
        k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"
    ]

    return _build_fastwam_delta_timestamps(
        action_horizon=action_horizon,
        fps=fps,
        observation_delta_indices=observation_delta_indices,
        image_keys=image_keys,
    )


def build_fastwam_lerobot_dataset(model_cfg, data_cfg, training_args):
    """Build the FastWAM multi-frame-observation dataset via the behaviour hook.

    Calls :func:`build_default_lerobot_dataset` with
    :func:`fastwam_delta_timestamps` as the ``delta_timestamps_fn`` hook.
    Sampling geometry (``observation_delta_indices``) comes from the FastWAM
    ``data_cfg``; ``action_horizon`` comes from the ``model_cfg``.
    """
    return build_default_lerobot_dataset(
        model_cfg,
        data_cfg,
        training_args,
        observation_delta_indices=data_cfg.observation_delta_indices,
        delta_timestamps_fn=fastwam_delta_timestamps,
    )
