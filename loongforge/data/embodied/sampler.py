# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DataLoader sampler selection, with a default distributed sampler.

A *sampler builder* is a callable ``fn(SamplerBuilderContext) -> Sampler | None``
selected by ``model_cfg.model_type``. Models that need custom index ordering
(e.g. multi-frame grouping, curriculum sampling) register their own builder via
``@register_sampler_builder("<model_type>")`` (``embodied/registry.py``); models without a registered
builder fall back to :func:`default_sampler_builder`.

Returning ``None`` means "no sampler" — the DataLoader then falls back to plain
shuffling (map-style) or natural iteration (iterable-style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset, Sampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


@dataclass(frozen=True)
class SamplerBuilderContext:
    """Shared inputs available to sampler builders."""

    dataset: Any
    training_args: Any
    ctx: Any  # DistributedContext
    batch_size: int
    seed: int
    shuffle: bool


SamplerBuilder = Callable[[SamplerBuilderContext], Optional[Sampler]]


class _StatefulIndexIterator(Iterator[int]):
    """Iterator state used by torchdata's StatefulDataLoader."""

    _YIELDED = "yielded"

    def __init__(self, sampler) -> None:
        self._sampler = sampler
        self._yielded = 0

    def __iter__(self) -> "_StatefulIndexIterator":
        return self

    def __next__(self) -> int:
        if self._yielded >= len(self._sampler._indices):
            raise StopIteration
        value = self._sampler._indices[self._yielded]
        self._yielded += 1
        return value

    def state_dict(self) -> dict[str, int]:
        """Return the iterator state for checkpointing."""
        return {self._YIELDED: self._yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the iterator position from a checkpoint."""
        yielded = int(state_dict.get(self._YIELDED, 0))
        if yielded < 0:
            raise ValueError(f"Cannot restore sampler iterator with negative yielded={yielded}")
        if yielded > len(self._sampler._indices):
            raise ValueError(
                f"Cannot restore sampler iterator yielded={yielded}; "
                f"current sampler only has {len(self._sampler._indices)} indices"
            )
        self._yielded = yielded


class GlobalBatchShardSampler(Sampler[int]):
    """Shard a global index order by whole local batches across DP ranks."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank} for world_size {num_replicas}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._indices = self._build_indices()

    def _build_indices(self) -> list[int]:
        length = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(length, generator=generator).tolist()
        else:
            indices = list(range(length))

        if self.drop_last:
            usable = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:usable]

        out: list[int] = []
        global_batch = self.batch_size * self.num_replicas
        local_start = self.rank * self.batch_size
        for start in range(local_start, len(indices), global_batch):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                out.extend(batch)
        return out

    def __iter__(self) -> Iterator[int]:
        return _StatefulIndexIterator(self)

    def __len__(self) -> int:
        return len(self._indices)

    def set_epoch(self, epoch: int) -> None:
        """Rebuild this rank's indices for a new epoch."""
        self.epoch = int(epoch)
        self._indices = self._build_indices()

    def state_dict(self) -> dict[str, Any]:
        """Return sampler state for checkpointing."""
        return {"version": 1, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore sampler state from a checkpoint."""
        self.set_epoch(int(state_dict.get("epoch", 0)))


def filter_schedule_for_worker(
    schedule: list[int],
    *,
    rank: int,
    world_size: int,
    worker_id: int = 0,
    num_workers: int = 1,
) -> list[int]:
    """Round-robin ``schedule`` over ``(rank, worker)`` slots; keep this slot's entries."""
    modulus = world_size * num_workers
    offset = rank * num_workers + worker_id
    return [s for pos, s in enumerate(schedule) if pos % modulus == offset]


def default_sampler_builder(context: SamplerBuilderContext) -> Optional[Sampler]:
    """Default distributed sampler selection (block vs stateful-distributed).

    Returns ``None`` for iterable datasets and for the single-process /
    non-distributed case, letting the DataLoader handle plain shuffling.
    """
    dataset = context.dataset
    if isinstance(dataset, IterableDataset):
        return None

    ctx = context.ctx
    if not (ctx.is_distributed and ctx.world_size > 1):
        return None

    drop_last = context.training_args.batch_drop_last
    if context.training_args.distributed_sampler_mode == "block":
        return GlobalBatchShardSampler(
            dataset,
            batch_size=context.batch_size,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=context.shuffle,
            seed=context.seed,
            drop_last=drop_last,
        )

    return StatefulDistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=context.shuffle,
        seed=context.seed,
        drop_last=drop_last,
    )


def build_sampler(
    model_type: str,
    *,
    dataset,
    training_args,
    ctx,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> Optional[Sampler]:
    """Build the sampler for ``model_type`` via its registered builder."""
    from loongforge.data.embodied.registry import get_sampler_builder

    builder = get_sampler_builder(model_type) or default_sampler_builder
    context = SamplerBuilderContext(
        dataset=dataset,
        training_args=training_args,
        ctx=ctx,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
    )
    return builder(context)
