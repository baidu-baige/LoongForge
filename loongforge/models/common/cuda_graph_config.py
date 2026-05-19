# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CUDA Graph mode query interface (model-agnostic).

Provides a unified interface for querying which CUDA graph mode is active.
Reads training args ``--cuda-graph-impl`` and ``--cuda-graph-scope`` and
exposes typed predicates that any model layer can call to fork its behavior
between eager and graph-captured paths.

Configuration is read from training args:
    --cuda-graph-impl local              : Required for graph to activate.
    --cuda-graph-scope full_iteration    : Single-graph capture of the entire
                                           forward+backward iteration.
    --cuda-graph-scope per_microbatch    : Per-microbatch sub-graph capture
                                           with eager RNG between sub-graphs
                                           (bit-exact loss alignment).
"""
from __future__ import annotations


def _get_args():
    """Import and return training args."""
    from loongforge.utils import get_args
    return get_args()


def _is_local_impl() -> bool:
    """Check whether cuda_graph_impl is 'local'."""
    args = _get_args()
    if args is None:
        return False
    return getattr(args, "cuda_graph_impl", "none") == "local"


def is_full_iteration_graph() -> bool:
    """Check whether full-iteration CUDA graph is active.

    When --cuda-graph-impl=local and --cuda-graph-scope=full_iteration,
    the entire forward+backward is captured as a single CUDA graph.
    """
    args = _get_args()
    if args is None:
        return False
    scope = getattr(args, "cuda_graph_scope", "full")
    return _is_local_impl() and scope == "full_iteration"


def is_per_microbatch_graph() -> bool:
    """Check whether per-microbatch CUDA graph mode is active.

    When --cuda-graph-impl=local and --cuda-graph-scope=per_microbatch, each
    microbatch is captured as its own sub-graph and any non-deterministic RNG
    operations are externalized to run eagerly between sub-graph replays. This
    achieves bit-exact loss alignment with pure eager at near full-iteration
    performance.
    """
    args = _get_args()
    if args is None:
        return False
    scope = getattr(args, "cuda_graph_scope", "full")
    return _is_local_impl() and scope == "per_microbatch"


def is_any_graph_mode() -> bool:
    """Check whether any CUDA graph mode is active."""
    return is_full_iteration_graph() or is_per_microbatch_graph()
