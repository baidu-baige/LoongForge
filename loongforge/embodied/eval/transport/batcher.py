# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Server-side request batcher for eval inference (M3).

The batcher sits between the WebSocket handler and the policy. Requests that
need inference are queued and grouped for up to ``max_wait_ms`` or until
``max_batch_size`` is reached; cache hits (legacy server-side chunk replay) are
served immediately and never enter the queue.

Client-owned chunks (``return_action_chunk=True``) are never cached server-side,
so their queue contains real inference requests only.

The model-level batch contract (per-row seeds / batched input encoding) is
tracked by T-06. Until it lands, ``GenericPredictActionPolicy`` processes a
batch sequentially; enabling the batcher is therefore correctness-preserving
but does not yet reduce GPU time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AsyncBatcher:
    """Coalesce concurrent inference requests into batches."""

    def __init__(
        self,
        policy: Any,
        max_batch_size: int = 8,
        max_wait_ms: float = 5.0,
        latency_recorder: Optional[Any] = None,
        stats_interval_s: float = 30.0,
    ) -> None:
        self._policy = policy
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self._latency = latency_recorder
        # Periodic heartbeat so a SIGTERM'd server still leaves batch stats in
        # the log (the finally block never runs on SIGTERM).
        self._stats_interval_s = max(0.0, float(stats_interval_s))
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._closed = False
        self.total_requests = 0
        self.total_batches = 0
        self._batch_sizes: List[int] = []

    async def start(self) -> None:
        """Start the background batch loop."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="eval-batcher")
        if self._stats_interval_s > 0 and self._stats_task is None:
            self._stats_task = asyncio.create_task(
                self._log_stats_loop(), name="eval-batcher-stats"
            )
        if not getattr(self._policy, "supports_batching", False):
            logger.warning(
                "Server batching is enabled but the model has no batch contract "
                "(T-06); requests will be processed sequentially per batch."
            )

    async def stop(self) -> None:
        """Stop the batch loop."""
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._stats_task is not None:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass
            self._stats_task = None

    @property
    def stats(self) -> Dict[str, Any]:
        """Return cumulative batching counters."""
        return {
            "requests": self.total_requests,
            "batches": self.total_batches,
            "mean_batch_size": (
                sum(self._batch_sizes) / len(self._batch_sizes) if self._batch_sizes else 0.0
            ),
            "max_batch_size": max(self._batch_sizes) if self._batch_sizes else 0,
        }

    async def _log_stats_loop(self) -> None:
        """Emit a heartbeat with cumulative batch statistics."""
        while not self._closed:
            await asyncio.sleep(self._stats_interval_s)
            if self._closed:
                return
            stats = self.stats
            logger.info(
                "Batcher stats: requests=%d batches=%d mean_batch=%.2f max_batch=%d pending=%d",
                stats["requests"],
                stats["batches"],
                stats["mean_batch_size"],
                stats["max_batch_size"],
                self._queue.qsize(),
            )

    async def submit(self, payload: Dict[str, Any]) -> Any:
        """Submit one inference payload and await its result."""
        loop = asyncio.get_running_loop()
        # Cache hits do not need inference; keep them out of the queue.
        if hasattr(self._policy, "has_cached_action") and self._policy.has_cached_action(payload):
            return await loop.run_in_executor(None, lambda: self._policy.predict_action(**payload))

        future: asyncio.Future = loop.create_future()
        self.total_requests += 1
        await self._queue.put((payload, future))
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            payload, future = await self._queue.get()
            batch = [(payload, future)]
            deadline = loop.time() + self._max_wait_s
            while len(batch) < self._max_batch_size:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=timeout))
                except asyncio.TimeoutError:
                    break

            payloads = [item[0] for item in batch]
            start = time.perf_counter()
            try:
                results = await loop.run_in_executor(
                    None, self._policy.predict_action_batch, payloads
                )
            except Exception as exc:  # Propagate to every waiting request.
                for _, pending in batch:
                    if not pending.done():
                        pending.set_exception(exc)
                continue

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.total_batches += 1
            self._batch_sizes.append(len(batch))
            if self._latency is not None:
                self._latency.record(elapsed_ms)
            for (_, pending), result in zip(batch, results):
                if not pending.done():
                    pending.set_result(result)
