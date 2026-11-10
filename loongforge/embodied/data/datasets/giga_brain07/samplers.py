# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""giga_brain07 sampler, registered for the shared FinetuneTrainer dataloader.

Reproduces the giga-brain-0 index stream across data-parallel ranks:

* per epoch, ``np.random.seed(seed + epoch)`` and one ``np.random.permutation``
  per pass over the dataset, padded with further passes up to a whole multiple
  of the *global* batch (per_device_bs x world_size x grad_accum), exactly as
  giga's ``GigaBrainSampler``;
* the padded stream is cut into per-device batches and batch ``i`` goes to rank
  ``i % world_size``. That is how accelerate's ``BatchSamplerShard``
  (``split_batches=False``) hands batches out in the original giga trainer, and
  it is the same round-robin ``_BlockShardSampler`` does, hence the subclass.

``set_epoch`` / ``state_dict`` / ``load_state_dict`` are inherited from
``_BlockShardSampler``; the trainer advances the epoch at each StopIteration, so
consecutive finite epochs concatenate into giga's infinite stream.

The global ``np.random`` seeding is intentional: giga reseeds the process-wide
numpy RNG at each epoch, which also drives any main-process (num_workers=0)
transform randomness. Keeping it preserves loss parity with the baseline.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from loongforge.embodied.data.datasets.sampler_builder import (
    SamplerBuilderContext,
    _BlockShardSampler,
    register_sampler_builder,
)


class GigaBrain07Sampler(_BlockShardSampler):
    """giga per-epoch shuffle + padding, sharded by per-device batch round-robin."""

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        self.data_size = len(dataset)
        self.global_batch_size = int(batch_size) * int(num_replicas) * int(gradient_accumulation_steps)
        self.total_size = int(math.ceil(self.data_size / self.global_batch_size)) * self.global_batch_size

    def _epoch_indices(self) -> list[int]:
        """One giga epoch: reseed, permute, pad to ``total_size`` with extra passes."""
        np.random.seed(self.seed + self.epoch)
        chunks = []
        num = 0
        while num < self.total_size:
            indices = np.arange(self.data_size)
            if self.shuffle:
                indices = np.random.permutation(indices)
            take = min(len(indices), self.total_size - num)
            chunks.append(indices[:take])
            num += take
        return np.concatenate(chunks).tolist()

    def __iter__(self) -> Iterator[int]:
        indices = self._epoch_indices()
        bs = self.batch_size
        for batch_idx, start in enumerate(range(0, self.total_size, bs)):
            if batch_idx % self.num_replicas == self.rank:
                yield from indices[start : start + bs]

    def __len__(self) -> int:
        # total_size is a multiple of world_size * batch_size, so every rank
        # gets the same number of full batches.
        return self.total_size // self.num_replicas


@register_sampler_builder("giga_brain07")
def build_giga_brain07_sampler(context: SamplerBuilderContext) -> GigaBrain07Sampler:
    """Always use the giga sampler, single-process included (the default
    builder returns None there, which would fall back to torch shuffling)."""
    ctx = context.ctx
    distributed = bool(ctx.is_distributed and ctx.world_size > 1)
    return GigaBrain07Sampler(
        context.dataset,
        batch_size=int(context.batch_size),
        num_replicas=int(ctx.world_size) if distributed else 1,
        rank=int(ctx.rank) if distributed else 0,
        shuffle=bool(context.shuffle),
        seed=int(context.seed),
        gradient_accumulation_steps=int(context.training_args.gradient_accumulation_steps),
    )


__all__ = ["GigaBrain07Sampler", "build_giga_brain07_sampler"]
