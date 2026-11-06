# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in NVTX region markers for fine-grained profiling.

A single process-wide switch gates all NVTX ranges used to generate the
per-stage / per-operator breakdown (vision_tower / embed_prefix / embed_suffix /
paligemma_expert per-layer attention(QKV/SDPA/O)+MLP(gate/up/down)+RMSNorm+rotary
/ diffusion_loss / backward / optimizer_step). When disabled these are true
no-ops, so normal training pays zero overhead and has zero behavior change.

Enable via EITHER:
  * the unified config switch ``model.enable_nvtx_profiling: true``
    (GigaBrain07Policy.__init__ calls ``set_enabled`` at build time); or
  * the environment variable ``LOONGFORGE_NVTX=1`` (force-on override).

In addition to ``torch.cuda.nvtx`` ranges (visible in Nsight Systems), each
range — when enabled — also opens a ``torch.autograd.profiler.record_function``
so the SAME hierarchical labels show up inside ``torch.profiler`` traces. This
lets the op-level report attribute every aten op (forward AND its linked
backward) to the stage that issued it, exactly mirroring the nsys NVTX tree.
record_function does not alter any tensor value; it only annotates the trace,
so numerics/loss are identical whether the switch is on or off.

This module is a dependency-light leaf (os/contextlib/torch only) so both the
model package and the generic trainer can import the SAME switch without a
layering cycle.
"""

import contextlib
import os

import torch

_ENV_DEFAULT = os.environ.get("LOONGFORGE_NVTX", "0") == "1"
_ENABLED = _ENV_DEFAULT

# Manual record_function stack for the push/pop API (nvtx_push / nvtx_pop).
_RF_STACK: list = []


def set_enabled(flag: bool) -> None:
    """Set the process-wide NVTX switch (env LOONGFORGE_NVTX=1 force-on wins)."""
    global _ENABLED
    _ENABLED = bool(flag) or _ENV_DEFAULT


def is_enabled() -> bool:
    return _ENABLED


def _active() -> bool:
    return _ENABLED and torch.cuda.is_available()


@contextlib.contextmanager
def nvtx_range(msg: str):
    """Push/pop an NVTX + record_function range (no-op unless enabled)."""
    if _active():
        torch.cuda.nvtx.range_push(msg)
        rf = torch.autograd.profiler.record_function(msg)
        rf.__enter__()
        try:
            yield
        finally:
            rf.__exit__(None, None, None)
            torch.cuda.nvtx.range_pop()
    else:
        yield


def nvtx_push(msg: str) -> None:
    """Open an NVTX + record_function range (no-op unless enabled).

    Pair with ``nvtx_pop``. record_function objects are tracked on an internal
    stack so ``nvtx_pop`` closes them in LIFO order.
    """
    if _active():
        torch.cuda.nvtx.range_push(msg)
        rf = torch.autograd.profiler.record_function(msg)
        rf.__enter__()
        _RF_STACK.append(rf)


def nvtx_pop() -> None:
    """Close the most recent NVTX + record_function range (no-op unless enabled)."""
    if _active():
        if _RF_STACK:
            rf = _RF_STACK.pop()
            rf.__exit__(None, None, None)
        torch.cuda.nvtx.range_pop()
