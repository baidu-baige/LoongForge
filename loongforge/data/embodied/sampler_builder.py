# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Registry for DataLoader samplers, with a default distributed sampler.

A *sampler builder* is a callable ``fn(SamplerBuilderContext) -> Sampler | None``
selected by ``model_cfg.model_type``. Models that need custom index ordering
(e.g. multi-frame grouping, curriculum sampling) register their own builder via
``@register_sampler_builder("<model_type>")``; models without a registered
builder fall back to :func:`default_sampler_builder`.

Returning ``None`` means "no sampler" — the DataLoader then falls back to plain
shuffling (map-style) or natural iteration (iterable-style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
from torch.utils.data import IterableDataset, Sampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from loongforge.data.embodied.registry import get_sampler_builder

if TYPE_CHECKING:
    from loongforge.data.embodied.registry import DataRuntime


@dataclass(frozen=True)
class SamplerBuilderContext:
    """Shared inputs available to sampler builders."""

    dataset: Any
    training_args: Any
    runtime: "DataRuntime"
    batch_size: int
    seed: int
    shuffle: bool


SamplerBuilder = Callable[[SamplerBuilderContext], Optional[Sampler]]

class _BlockShardSampler(Sampler[int]):
    """Distribute contiguous batches across data-parallel ranks."""

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        length = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(length, generator=generator).tolist()
        else:
            indices = list(range(length))

        if self.drop_last:
            usable = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:usable]

        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
            if len(indices[start : start + self.batch_size]) == self.batch_size or not self.drop_last
        ]
        for batch_idx, batch in enumerate(batches):
            if batch_idx % self.num_replicas == self.rank:
                yield from batch

    def __len__(self) -> int:
        full_batches, remainder = divmod(len(self.dataset), self.batch_size)
        total_batches = full_batches if self.drop_last or remainder == 0 else full_batches + 1
        local_batches = (total_batches + self.num_replicas - 1 - self.rank) // self.num_replicas
        if not local_batches:
            return 0
        if self.drop_last or remainder == 0 or (local_batches - 1) * self.num_replicas + self.rank < full_batches:
            return local_batches * self.batch_size
        return (local_batches - 1) * self.batch_size + remainder

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch index for deterministic shuffling."""
        self.epoch = epoch

    def state_dict(self) -> dict:
        """Return checkpointable sampler state."""
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load checkpointed sampler state."""
        self.epoch = int(state_dict.get("epoch", 0))


def default_sampler_builder(context: SamplerBuilderContext) -> Optional[Sampler]:
    """Default distributed sampler selection (block vs stateful-distributed).

    Returns ``None`` for iterable datasets and for the single-process /
    non-distributed case, letting the DataLoader handle plain shuffling.
    """
    dataset = context.dataset
    if isinstance(dataset, IterableDataset):
        return None

    runtime = context.runtime
    if not (runtime.is_distributed and runtime.world_size > 1):
        return None

    drop_last = context.training_args.batch_drop_last
    if context.training_args.distributed_sampler_mode == "block":
        return _BlockShardSampler(
            dataset,
            batch_size=context.batch_size,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=context.shuffle,
            seed=context.seed,
            drop_last=drop_last,
        )

    return StatefulDistributedSampler(
        dataset,
        num_replicas=runtime.world_size,
        rank=runtime.rank,
        shuffle=context.shuffle,
        seed=context.seed,
        drop_last=drop_last,
    )


def build_sampler(
    model_type: str,
    *,
    dataset,
    training_args,
    runtime,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> Optional[Sampler]:
    """Build the sampler for ``model_type`` via its registered builder."""
    builder = get_sampler_builder(model_type) or default_sampler_builder
    context = SamplerBuilderContext(
        dataset=dataset,
        training_args=training_args,
        runtime=runtime,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
    )
    return builder(context)
