# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""giga_brain07 dataset strategy -- route-B build on the shared LeRobot reader.

Route B: giga's dataset *reading* reuses the shared
``LeRobotV3Dataset`` (which wraps the official lerobot v3 reader) instead of the
vendored ``giga_datasets.LeRobotDataset``. Parameters that differ from the stock
default are passed explicitly, so divergences are expressed as config rather
than a parallel reader:

  - ``action_chunk``  -> ``action_horizon`` (delta_timestamps action chunk)
  - ``video_backend`` -> forwarded as-is (giga uses ``torchcodec``)
  - ``meta_name``     -> attach dataset meta under this key per sample
                         (giga's transform consumes it for stats / fps /
                         robot_type), via the shared reader's ``meta_name`` param.

Registered as the ``"giga_brain07"`` strategy in ``registry`` and
re-exported from the package root (``giga_brain07``) so the custom trainer and
the shared dispatcher both resolve ``build_giga_brain07_dataset`` the same way
``lingbot_vla_v2`` exposes ``build_lingbot_vla_v2_dataset``.

Genuinely giga-specific reader hardening (bounded-memory video decode,
HF-parquet cache locking/retry, thin per-sample metadata, repack / selective
columns, dynamic/boundary samplers) is intentionally not part of this shared
base; if a large-scale run needs it, re-add it as separately-gated parameters on
the shared reader rather than as a fork.
"""

from __future__ import annotations

from pathlib import Path


def _resolve_data_root(data_cfg, training_args) -> str:
    """Prefer the CLI/launcher-provided ``--dataset-path`` (absolute machine
    path), falling back to the dataclass default ``data_cfg.data_path``.
    """
    root = getattr(training_args, "dataset_path", None) or data_cfg.data_path
    if not root:
        raise ValueError(
            "giga_brain07 strategy requires a dataset path "
            "(training_args.dataset_path or data_cfg.data_path)."
        )
    return str(root)


def build_giga_brain07_dataset(model_cfg, data_cfg, training_args):
    """Build giga_brain07's training dataset on top of the shared v3 reader.

    Returns a shared ``LeRobotV3Dataset`` configured for giga. The caller
    (trainer) attaches the giga transform via ``dataset._transform = transform``
    exactly as the generic ``dataloader._apply_transform`` does for every other
    model.
    """
    from loongforge.data.embodied.datasets.lerobot_dataset import LeRobotV3Dataset

    root = _resolve_data_root(data_cfg, training_args)
    repo_id = Path(root).name

    action_horizon = int(
        getattr(data_cfg, "action_chunk", getattr(model_cfg, "action_horizon", 50))
    )
    video_backend = getattr(data_cfg, "video_backend", "torchcodec")
    meta_name = getattr(data_cfg, "meta_name", None)

    dataset = LeRobotV3Dataset(
        repo_id=repo_id,
        root=root,
        action_horizon=action_horizon,
        episodes=None,
        video_backend=video_backend,
        meta_name=meta_name,
    )
    return dataset
