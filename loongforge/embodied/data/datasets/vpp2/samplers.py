# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VPP2 sampler compatibility helpers.

The canonical sampler is selected by the framework.  This module intentionally
keeps the strategy-specific geometry separate so the sampler can be extended
without changing the dataset output contract.
"""

from __future__ import annotations

from typing import Iterable

from loongforge.embodied.data.datasets.sampler_builder import (
    SamplerBuilderContext,
    _BlockShardSampler,
    register_sampler_builder,
)


def sequential_indices(length: int) -> Iterable[int]:
    """Yield the deterministic single-process order used by parity probes."""
    return range(int(length))


@register_sampler_builder("vpp2")
def build_vpp2_sampler(context: SamplerBuilderContext):
    """Match BASE's global randperm followed by contiguous batch assignment.

    BASE constructs one full permutation and accelerate assigns complete batches
    round-robin to ranks.  StatefulDistributedSampler would instead shard
    individual samples before DataLoader batching, so VPP2 uses the framework's
    block sampler explicitly.
    """
    if not context.ctx.is_distributed or context.ctx.world_size <= 1:
        return None
    return _BlockShardSampler(
        context.dataset,
        batch_size=context.batch_size,
        num_replicas=context.ctx.world_size,
        rank=context.ctx.rank,
        shuffle=context.shuffle,
        seed=context.seed,
        drop_last=context.training_args.batch_drop_last,
    )
