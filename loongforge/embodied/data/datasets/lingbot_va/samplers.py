# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA distributed samplers and sampler registry hook."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterator, List

import torch

from loongforge.embodied.model.lingbot_va.features import feature_enabled
from torch.utils.data import Sampler

from loongforge.embodied.data.datasets.sampler_builder import (
    SamplerBuilderContext,
    default_sampler_builder,
    register_sampler_builder,
)


class _LingBotBalancedDistributedSampler(Sampler[int]):
    """Deterministically rebalance variable-length LingBot samples across DP ranks."""

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        seed: int,
        balance_group_size: int,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.balance_group_size = max(1, int(balance_group_size))
        if not hasattr(dataset, "estimate_sample_cost"):
            raise ValueError(
                "LINGBOT_BALANCED_SAMPLER requires dataset.estimate_sample_cost(index)"
            )
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.block_size = self.num_replicas * self.balance_group_size
        self.total_size = int(
            math.ceil(len(self.dataset) / self.block_size) * self.block_size
        )
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if len(indices) < self.total_size:
            repeats = math.ceil((self.total_size - len(indices)) / len(indices))
            indices.extend((indices * repeats)[: self.total_size - len(indices)])

        rank_indices: List[int] = []
        for start in range(0, self.total_size, self.block_size):
            block = indices[start : start + self.block_size]
            items = [
                (float(self.dataset.estimate_sample_cost(index)), pos, index)
                for pos, index in enumerate(block)
            ]
            buckets = self._assign_rank_balanced(items)
            rank_indices.extend(index for _, index, _ in buckets[self.rank])

        self._maybe_export_order(rank_indices)
        return iter(rank_indices)

    def _maybe_export_order(self, rank_indices):
        export_dir = os.environ.get("LINGBOT_SAMPLE_ORDER_EXPORT_DIR", "")
        if not export_dir:
            return
        try:
            path = Path(export_dir)
            path.mkdir(parents=True, exist_ok=True)
            out_path = path / f"rank{self.rank:05d}_epoch{self.epoch:05d}.json"
            payload = {
                "rank": self.rank,
                "num_replicas": self.num_replicas,
                "epoch": self.epoch,
                "seed": self.seed,
                "shuffle": self.shuffle,
                "mode": "rank",
                "balance_group_size": self.balance_group_size,
                "dataset_len": len(self.dataset),
                "total_size": self.total_size,
                "num_samples": self.num_samples,
                "indices": [int(index) for index in rank_indices],
            }
            tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
            tmp_path.write_text(json.dumps(payload, separators=(",", ":")))
            os.replace(tmp_path, out_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to export LingBot sample order to {export_dir}: {exc}"
            ) from exc

    def _assign_rank_balanced(self, items):
        buckets = [[] for _ in range(self.num_replicas)]
        loads = [0.0] * self.num_replicas
        counts = [0] * self.num_replicas
        for cost, pos, index in sorted(items, key=lambda item: item[0], reverse=True):
            candidates = [
                rank
                for rank in range(self.num_replicas)
                if counts[rank] < self.balance_group_size
            ]
            dst = min(candidates, key=lambda rank: (loads[rank], counts[rank], rank))
            buckets[dst].append((pos, index, cost))
            loads[dst] += cost
            counts[dst] += 1
        for bucket in buckets:
            bucket.sort(key=lambda item: item[0])
        return buckets

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed deterministic sampler shuffling."""
        self.epoch = int(epoch)

    def state_dict(self) -> dict:
        """Return the sampler state needed for checkpointing."""
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore sampler state from a checkpoint dictionary."""
        self.epoch = int(state_dict.get("epoch", 0))


@register_sampler_builder("lingbot_va")
def build_lingbot_va_distributed_sampler(context: SamplerBuilderContext):
    """Return the accepted rank-balanced sampler when enabled."""
    dataset = context.dataset
    training_args = context.training_args
    ctx = context.ctx
    if not feature_enabled("LINGBOT_BALANCED_SAMPLER"):
        # Preserve the public distributed sampler semantics when the LingBot
        # rank-balancing optimization is disabled.  Returning None would make
        # each rank create an independent random sampler and diverge from the
        # community baseline's DistributedSampler partitioning.
        return default_sampler_builder(context)

    balance_group_size = max(1, int(training_args.gradient_accumulation_steps))
    return _LingBotBalancedDistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        seed=context.seed,
        balance_group_size=balance_group_size,
        shuffle=context.shuffle,
    )


__all__ = [
    "_LingBotBalancedDistributedSampler",
    "build_lingbot_va_distributed_sampler",
]
