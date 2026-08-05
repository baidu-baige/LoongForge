# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Motus distributed sampler builder.

Registered under ``@register_sampler_builder("motus")`` (mirrors the DreamZero
registration in ``datasets/dreamzero/__init__.py``) so the Motus map-style
dataset gets a sampler whose per-rank index stream is byte-identical to the base
Motus training run.

Base (``bak_dpc/Motus/train/train.py::create_dataloaders``) builds:

    torch.utils.data.DistributedSampler(dataset, num_replicas=ws, rank=r,
                                        shuffle=not PARITY)          # drop_last defaults to False
    DataLoader(..., drop_last=True)

i.e. the *sampler* keeps ``drop_last=False`` (it PADS the tail so every rank
sees ``ceil(N/ws)`` indices) and only the *DataLoader* drops the final partial
micro-batch. The framework default builder instead threads
``training_args.batch_drop_last`` (=True) into the sampler, which makes the
sampler TRUNCATE the tail — a different tail geometry from base. This builder
pins ``drop_last=False`` at the sampler to match base exactly.

``StatefulDistributedSampler.__iter__`` delegates to
``torch.utils.data.DistributedSampler.__iter__`` (plus a resume ``islice``), so
for the same ``shuffle``/``seed`` the strided per-rank order — rank ``r`` ==
``indices[r::ws]`` — is identical to base's ``DistributedSampler``. Under
``PARITY_DATA_SEED`` the dataloader passes ``shuffle=False``, giving the
deterministic ``rank0=[0,8,16,24,...]`` partition the parity harness relies on.
"""

from __future__ import annotations

from typing import Optional

from torch.utils.data import IterableDataset, Sampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from loongforge.embodied.data.datasets.sampler_builder import (
    SamplerBuilderContext,
    register_sampler_builder,
)


@register_sampler_builder("motus")
def build_motus_sampler(context: SamplerBuilderContext) -> Optional[Sampler]:
    """Build the Motus DP sampler, matching base's ``DistributedSampler``.

    Returns ``None`` for iterable datasets and the single-process /
    non-distributed case (letting the DataLoader handle plain ordering), exactly
    like ``default_sampler_builder``.
    """
    dataset = context.dataset
    if isinstance(dataset, IterableDataset):
        return None

    ctx = context.ctx
    if not (ctx.is_distributed and ctx.world_size > 1):
        return None

    # drop_last=False: pad the tail like base's DistributedSampler; the DataLoader
    # (drop_last=training_args.batch_drop_last) drops the final partial batch,
    # matching base's DataLoader(drop_last=True).
    return StatefulDistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=context.shuffle,
        seed=context.seed,
        drop_last=False,
    )
