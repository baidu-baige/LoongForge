# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.7 LeRobot dataset strategy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from loongforge.embodied.data.datasets.lerobot_dataset import _build_lerobot_dataset


def build_groot_n1_7_lerobot_dataset(model_cfg, data_cfg, training_args):
    """Build and prepare a LeRobot dataset for GR00T-N1.7 sample semantics."""
    dataset_path = training_args.dataset_path
    if not dataset_path:
        raise ValueError("Must specify --dataset-path")

    dataset_path = Path(dataset_path)
    repo_id = dataset_path.name

    dataset = _build_lerobot_dataset(
        repo_id=repo_id,
        root=str(dataset_path),
        action_horizon=model_cfg.action_horizon,
        streaming=training_args.streaming,
        episodes=None,
        video_backend=training_args.video_backend,
        tolerance_s=1e-4,
        lerobotdataset_version=training_args.lerobotdataset_version,
    )
    return prepare_groot_n1d7_lerobot_dataset(
        dataset,
        model_cfg,
        data_cfg=data_cfg,
        training_args=training_args,
    )


def prepare_groot_n1d7_lerobot_dataset(
    dataset: Any,
    model_cfg: Any,
    data_cfg: Any = None,
    training_args: Any = None,
) -> Any:
    """Adapt a generic LeRobot dataset to GR00T-N1.7 sample semantics."""
    root = _dataset_root(dataset)
    if root is not None:
        dataset.dataset_path = Path(root)
        dataset.modality = _read_json_if_exists(Path(root) / "meta" / "modality.json")
        _load_groot_n1d7_stats_json(dataset, Path(root))

    if _is_prepared(dataset):
        return dataset

    data_cfg = _default_data_cfg() if data_cfg is None else data_cfg
    dataset._groot_n1d7_data_cfg = data_cfg
    seed = _resolve_data_seed(training_args)
    dataset._groot_n1d7_seed = seed
    action_horizon = _resolve_data_action_horizon(model_cfg, data_cfg)
    if not _has_episode_index(dataset):
        raise TypeError(
            "GR00T-N1.7 requires a map-style LeRobot dataset with episode metadata; "
            f"got {type(dataset).__name__}"
        )

    if data_cfg.groot_preprocess_mode == "sharded":
        full_step_index, shard_meta = _build_isaac_sharded_step_index(
            dataset,
            data_cfg,
            action_horizon,
            training_args=training_args,
        )
        dataset._step_index = full_step_index
        dataset._groot_n1d7_prepared = True
        dataset.already_transformed = False
        return _GrootN1d7IsaacIterableDataset(
            dataset,
            sharded_episodes=shard_meta["sharded_episodes"],
            shard_lengths=shard_meta["shard_lengths"],
            seed=seed,
            num_shards_per_epoch=int(data_cfg.num_shards_per_epoch),
        )

    dataset._step_index = _build_effective_step_index(dataset, action_horizon)
    dataset._groot_n1d7_prepared = True
    dataset.already_transformed = False
    return dataset


def _default_data_cfg():
    from loongforge.embodied.data.datasets.groot_n1_7.transforms.data_configuration_groot_n1_7 import (
        GrootN1d7DataConfig,
    )

    return GrootN1d7DataConfig()


def _dataset_root(dataset: Any) -> Any:
    return dataset.root


def _is_prepared(dataset: Any) -> bool:
    try:
        return bool(dataset._groot_n1d7_prepared)
    except AttributeError:
        return False


def _has_episode_index(dataset: Any) -> bool:
    try:
        return dataset._episodes is not None and dataset._step_index is not None
    except AttributeError:
        return False


def _resolve_data_seed(training_args: Any) -> int:
    return int(training_args.seed)


def _load_groot_n1d7_stats_json(dataset: Any, root: Path) -> None:
    """Load GR00T-N1.7 stats.json while preserving Isaac-style JSON values.

    Isaac's loader keeps the raw JSON/list structure and lets NumPy perform the
    dtype conversion inside the normalization processor. Do the same here so the
    GR00T-N1.7 normalization path matches baseline semantics more closely.
    """
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        return
    raw = _read_json_if_exists(stats_path)
    dataset._stats = {
        key: {stat_key: stat_value for stat_key, stat_value in value.items() if not isinstance(stat_value, str)}
        for key, value in raw.items()
        if isinstance(value, dict) and not str(key).startswith("__")
    }


def _resolve_data_action_horizon(model_cfg: Any, data_cfg: Any) -> int:
    if data_cfg.preprocess_action_horizon is not None:
        return int(data_cfg.preprocess_action_horizon)
    checkpoint_horizon = _resolve_checkpoint_action_horizon(model_cfg, data_cfg)
    if checkpoint_horizon is not None:
        return checkpoint_horizon
    return int(model_cfg.action_horizon)


def _build_effective_step_index(dataset: Any, action_horizon: int) -> list[tuple[int, int]]:
    effective_index: list[tuple[int, int]] = []
    for episode in dataset._episodes:
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        effective_episode_length = max(0, episode_length - action_horizon + 1)
        effective_index.extend((episode_index, step) for step in range(effective_episode_length))
    return effective_index


def _build_isaac_sharded_step_index(
    dataset: Any,
    data_cfg: Any,
    action_horizon: int,
    training_args: Any = None,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    seed = _resolve_data_seed(training_args)
    shard_size = int(data_cfg.shard_size)
    episode_sampling_rate = float(data_cfg.episode_sampling_rate)

    full_step_index = _build_effective_step_index(dataset, action_horizon)
    pair_to_dataset_index = {
        (int(episode_index), int(step_index)): dataset_index
        for dataset_index, (episode_index, step_index) in enumerate(full_step_index)
    }

    shard_rng = np.random.default_rng(seed)
    shuffled_episode_positions = shard_rng.permutation(len(dataset._episodes))
    num_splits = int(1 / episode_sampling_rate)

    episode_splits = []
    total_steps = 0
    for episode_position in shuffled_episode_positions:
        episode = dataset._episodes[int(episode_position)]
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        step_indices = np.arange(0, max(0, episode_length - action_horizon + 1))
        shard_rng.shuffle(step_indices)
        total_steps += len(step_indices)
        for split_idx in range(num_splits):
            split_step_indices = step_indices[split_idx::num_splits]
            if len(split_step_indices) > 0:
                episode_splits.append((episode_index, split_step_indices))

    if total_steps <= 0 or not episode_splits:
        raise ValueError(
            "No valid GR00T-N1.7 sharded steps; episode lengths may be shorter "
            f"than action horizon {action_horizon}"
        )

    num_shards = min(int(np.ceil(total_steps / shard_size)), len(episode_splits))
    sharded_episodes: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(num_shards)]
    shard_lengths = np.zeros(num_shards, dtype=int)
    for split_idx, (episode_index, split_step_indices) in enumerate(episode_splits):
        if split_idx < num_shards:
            shard_index = split_idx
        else:
            shard_index = int(np.argmin(shard_lengths))
        sharded_episodes[shard_index].append((episode_index, split_step_indices))
        shard_lengths[shard_index] += len(split_step_indices)

    world_size, worker_count, batch_size = _resolve_isaac_dataloader_shape(training_args)
    dataset._groot_n1d7_sharded_episodes = sharded_episodes
    dataset._groot_n1d7_shard_lengths = shard_lengths.tolist()
    dataset._groot_n1d7_sharded_order = True
    dataset._groot_n1d7_sharded_order_shape = {
        "world_size": world_size,
        "worker_count": worker_count,
        "batch_size": batch_size,
    }
    dataset._groot_n1d7_pair_to_dataset_index = pair_to_dataset_index
    return full_step_index, {
        "sharded_episodes": sharded_episodes,
        "shard_lengths": shard_lengths.tolist(),
    }


class _GrootN1d7IsaacIterableDataset(IterableDataset):
    """Iterable dataset that mirrors Isaac's shard-cache preprocessing order."""

    use_torch_dataloader = True

    def __init__(
        self,
        dataset: Any,
        *,
        sharded_episodes: list[list[tuple[int, np.ndarray]]],
        shard_lengths: list[int],
        seed: int,
        num_shards_per_epoch: int,
    ) -> None:
        self._dataset = dataset
        self._transform = None
        self._sharded_episodes = sharded_episodes
        self._shard_lengths = np.asarray(shard_lengths, dtype=int)
        self._seed = int(seed)
        self._num_shards_per_epoch = int(num_shards_per_epoch)
        self._pair_to_dataset_index = dict(dataset._groot_n1d7_pair_to_dataset_index)
        self._groot_n1d7_sharded_order = True

        if self._dataset._transform is not None:
            self._dataset._transform = None

    def __iter__(self):
        rank, world_size = _resolve_dist_rank_world()
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)

        worker_slot = rank * num_workers + worker_id
        global_worker_count = world_size * num_workers
        epoch = 0
        while True:
            shard_schedule = _build_isaac_shard_schedule(
                num_shards=len(self._sharded_episodes),
                num_shards_per_epoch=self._num_shards_per_epoch,
                seed=self._seed + epoch,
            )
            worker_schedule = [
                shard_index
                for schedule_index, shard_index in enumerate(shard_schedule)
                if schedule_index % global_worker_count == worker_slot
            ]
            rng = np.random.default_rng(self._seed + epoch)
            for shard_index in worker_schedule:
                shard = self._get_transformed_shard(int(shard_index))
                indices_in_shard = np.arange(len(shard))
                rng.shuffle(indices_in_shard)
                for index in indices_in_shard:
                    yield shard[int(index)]
            epoch += 1

    def _get_transformed_shard(self, shard_index: int) -> list[dict[str, Any]]:
        datapoints: list[dict[str, Any]] = []
        for episode_index, split_step_indices in self._sharded_episodes[shard_index]:
            for step_index in split_step_indices:
                dataset_index = self._pair_to_dataset_index[(int(episode_index), int(step_index))]
                datapoint = self._dataset[dataset_index]
                if self._transform is not None:
                    datapoint = self._transform(datapoint)
                datapoints.append(datapoint)
        return datapoints

    def __getattr__(self, name: str):
        return object.__getattribute__(self._dataset, name)


def make_groot_n1d7_isaac_iterable_dataset(dataset: Any, transform: Any) -> IterableDataset:
    """Wrap a prepared LeRobot dataset with Isaac's iterable shard semantics."""
    sharded_episodes = dataset._groot_n1d7_sharded_episodes
    shard_lengths = dataset._groot_n1d7_shard_lengths
    if not sharded_episodes or not shard_lengths:
        raise ValueError("GR00T-N1.7 sharded iterable requested before shard metadata was built")
    data_cfg = dataset._groot_n1d7_data_cfg
    seed = dataset._groot_n1d7_seed
    num_shards_per_epoch = int(data_cfg.num_shards_per_epoch)
    return _GrootN1d7IsaacIterableDataset(
        dataset,
        sharded_episodes=sharded_episodes,
        shard_lengths=shard_lengths,
        seed=seed,
        num_shards_per_epoch=num_shards_per_epoch,
    )


def _resolve_dist_rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _resolve_isaac_dataloader_shape(training_args: Any) -> tuple[int, int, int]:
    world_size = _resolve_dist_rank_world()[1]
    num_workers = int(training_args.num_workers)
    batch_size = int(training_args.per_device_batch_size)
    return max(1, world_size), max(1, num_workers), max(1, batch_size)


def _build_isaac_shard_schedule(
    *,
    num_shards: int,
    num_shards_per_epoch: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    dataset_sampling_schedule = rng.choice(1, size=num_shards_per_epoch, p=[1.0])
    shard_ids = list(range(num_shards))
    rng.shuffle(shard_ids)
    shards_to_sample = [shard_ids]
    shard_schedule: list[int] = []
    for dataset_index in dataset_sampling_schedule:
        dataset_index = int(dataset_index)
        if not shards_to_sample[dataset_index]:
            shard_ids = list(range(num_shards))
            rng.shuffle(shard_ids)
            shards_to_sample[dataset_index] = shard_ids
        shard_schedule.append(shards_to_sample[dataset_index].pop(0))
    return shard_schedule


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open(encoding="utf-8") as file_obj:
            return json.load(file_obj)
    return {}


def _resolve_checkpoint_path(model_cfg: Any) -> Path | None:
    path_value = model_cfg.base_model_path
    if not path_value:
        path_value = model_cfg.pretrained_checkpoint
    if path_value is None:
        return None
    path = Path(str(path_value))
    return path if path.exists() else None


def _resolve_checkpoint_action_horizon(model_cfg: Any, data_cfg: Any) -> int | None:
    checkpoint_path = _resolve_checkpoint_path(model_cfg)
    if checkpoint_path is None:
        return None
    processor_config = _read_json_if_exists(checkpoint_path / "processor_config.json")
    processor_kwargs = processor_config.get("processor_kwargs", {})
    if not isinstance(processor_kwargs, dict):
        return None
    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None
    embodiment_tag = data_cfg.embodiment_tag
    embodiment_config = modality_configs.get(embodiment_tag, {})
    if not isinstance(embodiment_config, dict):
        return None
    action_config = embodiment_config.get("action", {})
    if not isinstance(action_config, dict):
        return None
    delta_indices = action_config.get("delta_indices", [])
    if not isinstance(delta_indices, list) or not delta_indices:
        return None
    return len(delta_indices)
