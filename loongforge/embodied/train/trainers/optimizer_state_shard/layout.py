# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Layout plumbing shared by the gradient and parameter collective paths.

Bucketing, the flat-buffer pool and the cross-rank plan probe are needed by both
``GradientReducer`` and ``ParameterSynchronizer``. Keeping one copy here is what
makes the overlapped and serial paths provably share a layout instead of drifting
apart.
"""

from __future__ import annotations

import zlib

import torch
import torch.distributed as dist

WHOLE_TENSOR = "whole_tensor"


def split_by_dtype(items, dtype_of):
    """Group items by wire dtype so a bucket never mixes precisions."""
    groups: dict[torch.dtype, list] = {}
    for item in items:
        groups.setdefault(dtype_of(item), []).append(item)
    return [groups[key] for key in sorted(groups, key=str)]


def bucket_by_bytes(items, limit_bytes, size_of):
    """Split an ordered sequence into buckets of at most ``limit_bytes``.

    ``size_of`` reports the fp32 master footprint, not the wire footprint: the
    measured bucket layout the overlap schedule was tuned against uses that
    accounting, so a bf16 bucket really is half its nominal size.
    """
    buckets = []
    current = []
    current_bytes = 0
    for item in items:
        size = size_of(item)
        if current and current_bytes + size > limit_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += size
        if current_bytes >= limit_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
    if current:
        buckets.append(current)
    return buckets


class BufferPool:
    """Reuse flat collective buffers, keyed by exact ``(numel, dtype, device)``."""

    def __init__(self):
        """Start with an empty pool."""
        self._pool: dict[tuple, list[torch.Tensor]] = {}

    def acquire(self, numel, dtype, device):
        """Return a buffer of exactly ``numel`` elements, reusing one if possible."""
        pool = self._pool.setdefault((numel, dtype, str(device)), [])
        if pool:
            return pool.pop()
        return torch.empty(numel, dtype=dtype, device=device)

    def release(self, buffer):
        """Return a buffer to the pool; ``None`` is accepted and ignored."""
        if buffer is None:
            return
        key = (buffer.numel(), buffer.dtype, str(buffer.device))
        self._pool.setdefault(key, []).append(buffer)

    def clear(self):
        """Drop every pooled buffer.

        Required whenever the plan is rebucketed: the key is the exact element
        count, so buffers from the previous generation would stay resident for
        the rest of training without ever being handed out again.
        """
        self._pool.clear()

    def __len__(self):
        """Return how many buffers are currently pooled."""
        return sum(len(buffers) for buffers in self._pool.values())


def assert_plan_matches_ranks(plan, group, world_size, device, message):
    """Fail fast, instead of deadlocking, when ranks disagree on a plan.

    The probe sends ``(signature, -signature)`` through a MAX all-reduce: the two
    halves only stay symmetric if every rank contributed the same signature.
    """
    if world_size <= 1:
        return
    signature = zlib.crc32(repr(plan).encode("utf-8"))
    probe = torch.tensor([signature, -signature], device=device, dtype=torch.int64)
    dist.all_reduce(probe, op=dist.ReduceOp.MAX, group=group)
    if int(probe[0]) != -int(probe[1]):
        raise RuntimeError(message)


__all__ = [
    "WHOLE_TENSOR",
    "BufferPool",
    "assert_plan_matches_ranks",
    "bucket_by_bytes",
    "split_by_dtype",
]
