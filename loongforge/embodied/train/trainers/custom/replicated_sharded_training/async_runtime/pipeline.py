# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Bounded look-ahead over an async executor.

The pattern this generalises: start work for step N+1 during step N's optimizer
window, then consume it in submission order. That needs three things a bare
executor does not give you -- a hard bound on how far ahead the producer may run,
the data-stream position as of *before* the look-ahead read, and a defined answer
for what happens to unconsumed work at a checkpoint.

    AsyncTargetProvider   the model-side adapter: submit / resolve / close
    BoundedAsyncPipeline  the depth bound, the ordering, and the position record

CUDA events, threads and live handles are never serialised. Unconsumed work is
dropped on resume and recomputed from the recorded position, which is the only
strategy that cannot double-count or skip an item.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol


class AsyncTargetProvider(Protocol):
    """Model-side adapter: build the inputs, interpret the outputs."""

    def submit(self, batch): ...
    def resolve(self, handle): ...
    def close(self) -> None: ...


class BoundedAsyncPipeline:
    """Ordered, depth-bounded look-ahead queue over an ``AsyncTargetProvider``."""

    def __init__(self, provider: AsyncTargetProvider, max_depth: int = 1):
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        self._provider = provider
        self._max_depth = int(max_depth)
        self._queue: deque = deque()
        # Producer position as of before the queued items were read, i.e. the
        # position a checkpoint must resume from while anything is still queued.
        self._position = None

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def position(self):
        """Recorded producer position, or ``None`` when nothing is queued."""
        return self._position if self._queue else None

    def pending(self) -> int:
        """Items submitted but not consumed yet."""
        return len(self._queue)

    def full(self) -> bool:
        """True when no further submission is allowed without consuming first."""
        return len(self._queue) >= self._max_depth

    def mark_position(self, position) -> None:
        """Record where the producer stood before the look-ahead read began.

        Called once per look-ahead round, before the first ``submit`` of that
        round; re-marking while items are still queued would lose the older,
        still-authoritative position, so it is refused.
        """
        if self._queue:
            raise RuntimeError(
                "cannot re-mark the pipeline position while items are still queued"
            )
        self._position = position

    def submit(self, item):
        """Start the provider's work for ``item`` and queue it in fetch order."""
        if self.full():
            raise RuntimeError(
                f"async pipeline is at max_depth={self._max_depth}; consume before "
                f"submitting more"
            )
        handle = self._provider.submit(item)
        self._queue.append((item, handle))
        return handle

    def consume(self):
        """Pop the oldest ``(item, handle)`` pair, or ``None`` when empty.

        The handle is returned unresolved: the consumer decides when to block on
        it, which is the whole point of submitting early.
        """
        if not self._queue:
            return None
        return self._queue.popleft()

    def close(self) -> None:
        """Drop queued work and close the provider; safe to call more than once."""
        self._queue.clear()
        self._position = None
        if self._provider is not None:
            self._provider.close()


__all__ = ["AsyncTargetProvider", "BoundedAsyncPipeline"]
