# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Side-stream CUDA execution, shared by the custom trainers.

The executor runs a side-effect-free CUDA function on its own stream and worker
thread. It does not know what the submitted work computes -- the custom_dp
trainer uses it for the teacher look-ahead, and nothing here depends on that
subsystem.
"""

from .executor import (
    ERROR_POLICIES,
    AsyncTaskError,
    CudaStreamExecutor,
    CudaTaskHandle,
)

__all__ = [
    "ERROR_POLICIES",
    "AsyncTaskError",
    "CudaStreamExecutor",
    "CudaTaskHandle",
]
