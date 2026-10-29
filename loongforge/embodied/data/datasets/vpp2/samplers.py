# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VPP2 sampler compatibility helpers.

The canonical sampler is selected by the framework.  This module intentionally
keeps the strategy-specific geometry separate so the sampler can be extended
without changing the dataset output contract.
"""

from __future__ import annotations

from typing import Iterable

import torch

from loongforge.embodied.data.datasets.sampler_builder import (
    SamplerBuilderContext,
    _BlockShardSampler,
    register_sampler_builder,
)


def sequential_indices(length: int) -> Iterable[int]:
    """Yield the deterministic single-process order used by parity probes."""
    return range(int(length))


class VPP2BlockShardSampler(_BlockShardSampler):
    """Match Accelerate's whole-batch sharding with ``even_batches=True``.

    Pad the final global batch from the beginning of the epoch so every rank
    executes the same number of full batches (and DeepSpeed collectives).
    With ``drop_last``, discard the incomplete global batch instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._yielded = 0
        self._resume_offset = 0

    def __len__(self) -> int:
        # Accelerate skips BatchSamplerShard for a single process, keeping
        # the plain DataLoader's partial final batch instead of padding it.
        if self.num_replicas == 1 and not self.drop_last:
            return len(self.dataset)
        global_batch = self.batch_size * self.num_replicas
        batches, remainder = divmod(len(self.dataset), global_batch)
        if remainder and not self.drop_last:
            batches += 1
        return batches * self.batch_size

    def __iter__(self):
        length = len(self.dataset)
        if not length:
            return
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(length, generator=generator).tolist()
        else:
            indices = list(range(length))
        total_size = len(self) * self.num_replicas
        if total_size > length:
            padding = total_size - length
            indices += (indices * ((padding + length - 1) // length))[:padding]
        else:
            indices = indices[:total_size]
        global_batch = self.batch_size * self.num_replicas
        offset = self._resume_offset
        self._resume_offset = 0
        self._yielded = offset
        local_position = 0
        for start in range(self.rank * self.batch_size, total_size, global_batch):
            for index in indices[start : start + self.batch_size]:
                if local_position >= offset:
                    self._yielded += 1
                    yield index
                local_position += 1

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self._yielded = 0
        self._resume_offset = 0

    def state_dict(self) -> dict:
        # StatefulDataLoader trusts a stateful sampler to restore its cursor;
        # saving only the epoch would silently replay the consumed batches.
        return {"epoch": self.epoch, "yielded": self._yielded}

    def load_state_dict(self, state_dict: dict) -> None:
        if "yielded" not in state_dict:
            raise ValueError(
                "VPP2 sampler checkpoint has no sample cursor; epoch-only state "
                "cannot resume the consumed batches exactly. Start a fresh run."
            )
        super().load_state_dict(state_dict)
        self._yielded = int(state_dict["yielded"])
        self._resume_offset = self._yielded


@register_sampler_builder("vpp2")
def build_vpp2_sampler(context: SamplerBuilderContext):
    """Match BASE's global randperm followed by contiguous batch assignment.

    BASE constructs one full permutation and accelerate assigns complete batches
    round-robin to ranks.  StatefulDistributedSampler would instead shard
    individual samples before DataLoader batching, so VPP2 uses the framework's
    block sampler with the same global tail padding explicitly.
    """
    # A single rank also needs a dedicated generator. Returning None lets
    # RandomSampler consume global RNG and ignores an explicit shuffle=False.
    return VPP2BlockShardSampler(
        context.dataset,
        batch_size=context.batch_size,
        num_replicas=context.ctx.world_size,
        rank=context.ctx.rank,
        shuffle=context.shuffle,
        seed=context.seed,
        drop_last=context.training_args.batch_drop_last,
    )
