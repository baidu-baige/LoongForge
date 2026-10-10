# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Full-iteration CUDA graph wrapper registry.
# Models register their wrapper class via register_full_iteration_graph_wrapper()
# during trainer construction; mcore_runner.train() looks up the wrapper here.
# ---------------------------------------------------------------------------
_FULL_ITER_GRAPH_REGISTRY: dict[str, type] = {}


def register_full_iteration_graph_wrapper(model_name: str, wrapper_cls: type):
    """Register a full-iteration CUDA graph wrapper for a given model."""
    _FULL_ITER_GRAPH_REGISTRY[model_name] = wrapper_cls


def get_full_iteration_graph_wrapper(model_name: str):
    """Look up a registered full-iteration graph wrapper, or None."""
    return _FULL_ITER_GRAPH_REGISTRY.get(model_name)


# ---------------------------------------------------------------------------
# Per-microbatch CUDA graph wrapper registry.
# Models register their wrapper class via register_per_microbatch_graph_wrapper()
# during trainer construction; mcore_runner.train() looks up the wrapper here.
# ---------------------------------------------------------------------------
_PER_MICROBATCH_GRAPH_REGISTRY: dict[str, type] = {}


def register_per_microbatch_graph_wrapper(model_name: str, wrapper_cls: type):
    """Register a per-microbatch CUDA graph wrapper for a given model."""
    _PER_MICROBATCH_GRAPH_REGISTRY[model_name] = wrapper_cls


def get_per_microbatch_graph_wrapper(model_name: str):
    """Look up a registered per-microbatch CUDA graph wrapper, or None."""
    return _PER_MICROBATCH_GRAPH_REGISTRY.get(model_name)
