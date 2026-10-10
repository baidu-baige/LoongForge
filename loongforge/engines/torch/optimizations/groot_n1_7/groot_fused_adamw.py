# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for the optional GR00T-N1.7 AOT AdamW ops.

The AOT package is only needed by the GR00T ``TEFusedAdamW`` update.  Keep its
import out of module initialization so importing the trainer still works with
the generic optimizer path (including when CUDA graphs are disabled).
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _load_ops():
    """Load and cache the optional AOT operators on their first use."""
    try:
        from groot_n1_7_op.groot_fused_adamw import (
            capturable_grad_scaled_step,
            capturable_step,
            eager_step,
        )
    except ImportError as exc:
        raise ImportError(
            "The GR00T-N1.7 AOT operators are unavailable. Install "
            "DeepTraining/cuda_source/groot_n1_7_op in the runtime Python environment."
        ) from exc
    return capturable_grad_scaled_step, capturable_step, eager_step


def capturable_grad_scaled_step(*args, **kwargs):
    """Run the AOT capturable AdamW update with a device-side grad scale."""
    return _load_ops()[0](*args, **kwargs)


def capturable_step(*args, **kwargs):
    """Run the AOT capturable AdamW update."""
    return _load_ops()[1](*args, **kwargs)


def eager_step(*args, **kwargs):
    """Run the AOT eager AdamW update."""
    return _load_ops()[2](*args, **kwargs)


__all__ = [
    "capturable_grad_scaled_step",
    "capturable_step",
    "eager_step",
]
