# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain-0 dataset loader for ``--dataset-format giga_brain_datasets``.

Wraps the pip-installed ``lerobot`` package's ``LeRobotDataset`` (v3.0-capable,
lerobot==0.5.0 — see environment note below) directly, reproducing the
subset of ``giga_datasets.datasets.lerobot_dataset.LeRobotDataset`` behavior
that GigaBrain-0's transform pipeline depends on:

  * ``delta_info=dict(action=action_chunk)`` -> ``delta_timestamps`` built the
    same way giga_datasets does: ``{key: [i / fps for i in range(chunk)]}``.
  * ``data_dict['meta']`` -> an object exposing ``.info`` (the parsed
    ``meta/info.json`` dict), since ``giga_brain_0_transforms.py`` reads
    ``data_dict['meta'].info['robot_type']``.

This module does NOT vendor ``giga_datasets`` itself (per user requirement:
model/dataset code is vendored into LoongForge, but the lerobot dependency
uses the current environment's installed package rather than an isolated
venv). ``lerobot==0.5.0`` is a required environment dependency for this
dataset format — v3.0 datasets are unreadable with the 0.3.2 series that
older LoongForge/giga-brain-0 environments may have installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class _MetaView:
    """Minimal stand-in for ``giga_datasets``' per-sample ``meta`` object.

    ``GigaBrain0Transform.__call__`` only ever reads ``meta.info[...]``
    (see ``giga_brain_0_transforms.py``), so this only needs to expose
    ``.info`` — the parsed ``meta/info.json`` dict, same as
    ``lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata.info``.
    """

    __slots__ = ("info",)

    def __init__(self, info: dict):
        self.info = info


class GigaBrainLeRobotDataset(Dataset):
    """Map-style dataset wrapping ``lerobot.datasets.LeRobotDataset``.

    Attaches a ``meta`` key to every sample (mirroring
    ``giga_datasets.LeRobotDataset(meta_name='meta')``) so
    ``GigaBrain0Transform`` can resolve ``robot_type`` -> embodiment id
    without any other code change.
    """

    def __init__(self, data_path: str, action_chunk: int, meta_name: str = "meta"):
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )

        repo_id = Path(data_path).name
        meta = LeRobotDatasetMetadata(repo_id, root=data_path)
        delta_timestamps = {"action": [i / meta.fps for i in range(action_chunk)]}

        self._dataset = LeRobotDataset(
            repo_id,
            root=data_path,
            delta_timestamps=delta_timestamps,
        )
        self._meta_name = meta_name
        self._meta_view = _MetaView(dict(meta.info))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data_dict = dict(self._dataset[index])
        data_dict[self._meta_name] = self._meta_view
        return data_dict


def build_giga_brain_dataset(model_cfg, data_cfg, training_args) -> GigaBrainLeRobotDataset:
    """Build a :class:`GigaBrainLeRobotDataset` for ``--dataset-format giga_brain_datasets``.

    ``training_args.dataset_path`` (``--dataset-path``) takes precedence over
    ``data_cfg.data_path`` (YAML ``data:`` section), matching the convention
    used by ``lerobot_dataset.py::build_default_lerobot_dataset``.
    ``action_chunk`` is read from the model config's ``n_action_steps``
    (== reference ``action_chunk`` in ``configs/giga_brain_0_*_finetune*.py``).
    """
    data_path = training_args.dataset_path or data_cfg.data_path
    if not data_path:
        raise ValueError(
            "giga_brain_datasets: no dataset path given. Set --dataset-path or "
            "data.data_path in the YAML config."
        )
    action_chunk = model_cfg.n_action_steps
    logger.info(
        "Building GigaBrain LeRobotDataset: data_path=%s action_chunk=%d",
        data_path, action_chunk,
    )
    return GigaBrainLeRobotDataset(data_path=data_path, action_chunk=action_chunk)
