# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 sampler: cosmos-framework's episode-shuffle stream as an index sampler.

Port of ``ActionIterableShuffleDataset.__iter__``: each
epoch permutes the per-episode blocks with ``randperm(len(blocks), seed + epoch)``,
assigns blocks disjointly to shards ``order[shard::total_shards]`` and walks the
windows of each block sequentially. A shard is one (rank, cosmos worker) pair;
with ``shards_per_rank > 1`` the rank's stream round-robins its shards in chunks of
``worker_batch`` samples, like a DataLoader with that many workers.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler

from loongforge.embodied.data.datasets.sampler_builder import (
    SamplerBuilderContext,
    default_sampler_builder,
    register_sampler_builder,
)


def _unwrap(dataset):
    """Return the innermost dataset exposing ``get_shuffle_blocks``."""
    seen = set()
    while id(dataset) not in seen:
        seen.add(id(dataset))
        if callable(getattr(type(dataset), "get_shuffle_blocks", None)):
            return dataset
        inner = getattr(dataset, "_dataset", None) or getattr(dataset, "dataset", None)
        if inner is None:
            break
        dataset = inner
    return None


class Cosmos3EpisodeStreamSampler(Sampler[int]):
    """Rank-local index stream matching cosmos-framework's episode-shuffle order."""

    def __init__(
        self,
        blocks: List[tuple],
        *,
        num_replicas: int,
        rank: int,
        seed: int = 42,
        shards_per_rank: int = 1,
        worker_batch: int = 16,
    ) -> None:
        self.blocks = blocks
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shards_per_rank = max(1, int(shards_per_rank))
        self.worker_batch = max(1, int(worker_batch))
        self.epoch = 0

    def _shard_order(self) -> List[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.blocks), generator=g).tolist()

    def _shard_indices(self, order: List[int], shard: int, total_shards: int) -> Iterator[int]:
        for b in order[shard::total_shards]:
            start, length = self.blocks[b]
            yield from range(start, start + length)

    def __iter__(self) -> Iterator[int]:
        order = self._shard_order()
        total_shards = self.num_replicas * self.shards_per_rank
        streams = [
            self._shard_indices(order, self.rank * self.shards_per_rank + w, total_shards)
            for w in range(self.shards_per_rank)
        ]
        if len(streams) == 1:
            yield from streams[0]
            return
        active = list(streams)
        while active:
            still_active = []
            for stream in active:
                chunk = [idx for _, idx in zip(range(self.worker_batch), stream)]
                yield from chunk
                if len(chunk) == self.worker_batch:
                    still_active.append(stream)
            active = still_active

    def __len__(self) -> int:
        order = self._shard_order()
        total_shards = self.num_replicas * self.shards_per_rank
        shards = range(self.rank * self.shards_per_rank, (self.rank + 1) * self.shards_per_rank)
        return sum(self.blocks[b][1] for s in shards for b in order[s::total_shards])

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch index (block permutation seed is ``seed + epoch``)."""
        self.epoch = int(epoch)

    def state_dict(self) -> dict:
        """Return checkpointable sampler state."""
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load checkpointed sampler state."""
        self.epoch = int(state_dict.get("epoch", 0))


@register_sampler_builder("cosmos3")
def cosmos3_sampler_builder(context: SamplerBuilderContext) -> Optional[Sampler]:
    """Use the episode stream when the DROID dataset enables it, else the default sampler."""
    inner = _unwrap(context.dataset)
    cfg = getattr(inner, "episode_stream_cfg", None) if inner is not None else None
    if not cfg or not cfg.get("enabled", False):
        return default_sampler_builder(context)
    ctx = context.ctx
    distributed = ctx.is_distributed and ctx.world_size > 1
    return Cosmos3EpisodeStreamSampler(
        inner.get_shuffle_blocks(),
        num_replicas=ctx.world_size if distributed else 1,
        rank=ctx.rank if distributed else 0,
        seed=cfg["seed"],
        shards_per_rank=cfg["shards_per_rank"],
        worker_batch=cfg["worker_batch"],
    )
