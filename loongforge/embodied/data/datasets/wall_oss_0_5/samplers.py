# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Wall-OSS-0.5 sampler parity with the original Wall-X trainer."""

from __future__ import annotations

from torch.utils.data import DistributedSampler

from loongforge.embodied.data.datasets.sampler_builder import (
    SamplerBuilderContext,
    register_sampler_builder,
)


@register_sampler_builder("wall_oss_0_5")
def wall_oss_0_5_sampler_builder(context: SamplerBuilderContext):
    """Wall oss 0 5 sampler builder."""
    return DistributedSampler(
        context.dataset,
        num_replicas=context.ctx.world_size,
        rank=context.ctx.rank,
        shuffle=context.shuffle,
        seed=context.seed,
        drop_last=True,
    )
