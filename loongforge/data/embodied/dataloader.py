# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DataLoader factory for embodied VLA training."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from loongforge.data.embodied.registry import (
    DataRuntime,
    build_collator,
    build_dataset,
    build_transforms_from_args,
)
from loongforge.data.embodied.sampler_builder import build_sampler

logger = logging.getLogger(__name__)


class _SeedWorkerInit:
    """Picklable worker initializer for spawn/forkserver DataLoader workers."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def __call__(self, worker_id: int) -> None:
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)


class _TransformedMapDataset(Dataset):
    """Wrap a map-style dataset and apply a per-sample transform."""

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        data = self._dataset[idx]
        return self._transform(data)

    def __getattr__(self, name):
        try:
            dataset = self.__dict__["_dataset"]
        except KeyError:
            raise AttributeError(name)
        return getattr(dataset, name)


class _TransformedIterableDataset(IterableDataset):
    """Wrap an iterable dataset and apply a per-sample transform."""

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __iter__(self):
        for data in self._dataset:
            yield self._transform(data)

    def __getattr__(self, name):
        try:
            dataset = self.__dict__["_dataset"]
        except KeyError:
            raise AttributeError(name)
        return getattr(dataset, name)


def build_dataloader(model_cfg, data_cfg, training_args, runtime: DataRuntime) -> StatefulDataLoader:
    """Build DataLoader with model-specific collator as collate_fn.

    ``runtime`` carries the execution facts the training method resolved. The
    loader reads data-recipe values from ``training_args`` but must not query
    backend distributed state or call global readers.

    The returned DataLoader yields PreparedBatch objects on CPU. Call
    ``batch.to(device)`` before passing a batch to ``model.forward``.
    """
    batch_size = training_args.per_device_batch_size
    num_workers = training_args.num_workers
    mp_context = training_args.dataloader_multiprocessing_context or ("spawn" if num_workers > 0 else None)

    dataset = build_dataset(model_cfg, data_cfg, training_args, runtime)
    dataset_stats = _get_dataset_stats(dataset)

    transform = build_transforms_from_args(
        model_cfg, data_cfg, training_args, dataset, dataset_stats, runtime
    )
    if transform is not None:
        dataset = _apply_transform(dataset, transform)

    collator = _build_collator(model_cfg, data_cfg, training_args, dataset_stats, dataset, runtime)

    return _build_stateful_dataloader(
        dataset=dataset,
        collator=collator,
        model_type=model_cfg.model_type or "dummy",
        batch_size=batch_size,
        num_workers=num_workers,
        mp_context=mp_context,
        training_args=training_args,
        runtime=runtime,
    )


def _get_dataset_stats(dataset):
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "stats"):
        return dataset.meta.stats
    return {}


def _apply_transform(dataset, transform):
    if hasattr(dataset, "_transform"):
        dataset._transform = transform
        return dataset
    if isinstance(dataset, IterableDataset):
        return _TransformedIterableDataset(dataset, transform)
    return _TransformedMapDataset(dataset, transform)


def _build_stateful_dataloader(
    *,
    dataset,
    collator,
    model_type: str,
    batch_size: int,
    num_workers: int,
    mp_context,
    training_args,
    runtime: DataRuntime,
) -> StatefulDataLoader:
    """_build_stateful_dataloader."""
    seed = training_args.seed
    seed_workers = training_args.dataloader_seed_workers
    drop_last = training_args.batch_drop_last
    prefetch_factor = (
        training_args.dataloader_prefetch_factor if num_workers > 0 else None
    )
    if isinstance(dataset, IterableDataset):
        return StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0 and mp_context == "spawn",
        )

    generator = torch.Generator().manual_seed(seed) if seed_workers else None
    sampler = build_sampler(
        model_type,
        dataset=dataset,
        training_args=training_args,
        runtime=runtime,
        batch_size=batch_size,
        seed=seed,
        shuffle=training_args.sampler_shuffle,
    )
    shuffle = sampler is None
    dl = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_SeedWorkerInit(seed) if seed_workers and num_workers > 0 else None,
        generator=generator,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0 and mp_context == "spawn",
    )
    dl.steps_per_epoch = len(dl)
    return dl


def _build_collator(model_cfg, data_cfg, training_args, dataset_stats, dataset, runtime):
    """Build batch-level collator from the collator registry."""
    model_type = model_cfg.model_type or "dummy"
    collator = build_collator(
        model_type,
        model_cfg,
        data_cfg,
        training_args=training_args,
        dataset_stats=dataset_stats,
        dataset=dataset,
        runtime=runtime,
    )

    logger.info(f"Using collator: {model_type}")
    return collator
