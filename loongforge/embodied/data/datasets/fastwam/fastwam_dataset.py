# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM multi-frame-observation dataset strategy: behaviour hook for the
generic lerobot datasets.

Like Motus, FastWAM plugs its geometry into :class:`LeRobotV3Dataset` through the
``delta_timestamps_fn`` hook instead of branching inside the dataset class. The
hook adds, for every video key, a multi-frame observation stack sampled at the
``observation_delta_indices`` offsets (in addition to the standard pi05 action
chunk). The base dataset stays entirely model-agnostic.

``build_fastwam_lerobot_dataset`` wires the hook onto a single
``LeRobotV3Dataset``, deriving ``repo_id`` from ``--dataset-path`` and reading
``observation_delta_indices`` / ``action_horizon`` from the FastWAM configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from loongforge.embodied.data.datasets.lerobot_dataset import (
    LeRobotV3Dataset,
    _build_lerobot_dataset,
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

    Derives ``repo_id`` from ``--dataset-path`` and dispatches through the
    version-aware factory (``_build_lerobot_dataset``), wiring
    :func:`fastwam_delta_timestamps` as the ``delta_timestamps_fn`` hook.
    Sampling geometry (``observation_delta_indices``) comes from the FastWAM
    ``data_cfg``; ``action_horizon`` comes from the ``model_cfg``.
    """
    dataset_path = training_args.dataset_path
    if not dataset_path:
        raise ValueError("Must specify --dataset-path")

    dataset_path = Path(dataset_path)
    repo_id = dataset_path.name

    return _build_lerobot_dataset(
        repo_id=repo_id,
        root=str(dataset_path),
        action_horizon=model_cfg.action_horizon,
        streaming=training_args.streaming,
        episodes=None,
        video_backend=training_args.video_backend,
        tolerance_s=1e-4,
        lerobotdataset_version=training_args.lerobotdataset_version,
        observation_delta_indices=data_cfg.observation_delta_indices,
        delta_timestamps_fn=fastwam_delta_timestamps,
    )
