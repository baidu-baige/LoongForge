# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in per-request latency recording for the eval policy server.

Used by the M0 calibration sweep to export a service-time distribution without
changing request handling. Enabled via ``LOONGFORGE_EVAL_TIMING``; when disabled
recording is a no-op.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List


def _env_flag(name: str) -> bool:
    """Return True when an environment flag is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def percentile(sorted_values: List[float], q: float) -> float:
    """Linear-interpolation percentile over pre-sorted values; ``q`` in [0, 1]."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize(values: List[float]) -> Dict[str, Any]:
    """Return count/mean/p50/p95/p99/max for a list of millisecond latencies."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": float(ordered[-1]),
    }


class LatencyRecorder:
    """Collect per-request service times and export a summary (opt-in)."""

    def __init__(self, enabled: bool = False, path: str = "", flush_every: int = 0) -> None:
        """Run __init__."""
        self.enabled = enabled
        self.path = path
        self.flush_every = flush_every
        self._values: List[float] = []
        self._lock = threading.RLock()
        if self.enabled and self.path:
            # Best-effort final write on a clean interpreter exit. A SIGTERM does
            # not run atexit handlers, so calibration also sets FLUSH_EVERY=1.
            atexit.register(self.flush)

    @classmethod
    def from_env(cls) -> "LatencyRecorder":
        """Build a recorder from the ``LOONGFORGE_EVAL_TIMING*`` environment variables."""
        return cls(
            enabled=_env_flag("LOONGFORGE_EVAL_TIMING"),
            path=os.environ.get("LOONGFORGE_EVAL_TIMING_PATH", ""),
            flush_every=int(os.environ.get("LOONGFORGE_EVAL_TIMING_FLUSH_EVERY", "0") or 0),
        )

    def record(self, ms: float) -> None:
        """Append one service time in milliseconds; no-op when disabled."""
        if not self.enabled:
            return
        with self._lock:
            self._values.append(float(ms))
            if self.flush_every and len(self._values) % self.flush_every == 0:
                self.flush()

    def summary(self) -> Dict[str, Any]:
        """Return the current distribution summary."""
        return summarize(self._values)

    def flush(self) -> None:
        """Atomically write the current summary to ``path`` (no-op if unset)."""
        if not self.enabled or not self.path:
            return
        out = Path(self.path)
        with self._lock:
            out.parent.mkdir(parents=True, exist_ok=True)
            # Unique temp name: concurrent requests (one server, many sims) may
            # flush at the same time and would otherwise race on the same file.
            tmp = out.with_name(f"{out.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
            os.replace(tmp, out)
