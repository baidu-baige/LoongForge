# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Side-stream CUDA execution, shared by the custom trainers.

The executor runs a side-effect-free CUDA function on its own stream and worker
thread; the pipeline bounds how many such tasks may run ahead of the step and
records the training position they were launched from. Neither knows what the
submitted work computes -- the replicated-sharded trainer uses it for the teacher
look-ahead, and nothing here depends on that subsystem.
"""

from .executor import (
    ERROR_POLICIES,
    AsyncTaskError,
    CudaStreamExecutor,
    CudaTaskHandle,
)
from .pipeline import AsyncTargetProvider, BoundedAsyncPipeline

__all__ = [
    "ERROR_POLICIES",
    "AsyncTargetProvider",
    "AsyncTaskError",
    "BoundedAsyncPipeline",
    "CudaStreamExecutor",
    "CudaTaskHandle",
]
