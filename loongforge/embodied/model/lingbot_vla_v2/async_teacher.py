# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Bind the async CUDA executor to LingBot's frozen teachers.

The executor lives in ``replicated_sharded_training.async_runtime`` and knows
nothing about teachers. The teachers are a pure function of the sampled images
and only feed the auxiliary heads, so they run on a side stream, and
``fallback_to_sync`` is a legal error policy.
"""

from __future__ import annotations

from loongforge.embodied.train.trainers.custom.replicated_sharded_training.async_runtime import (
    CudaStreamExecutor,
)


class AsyncTeacherRunner(CudaStreamExecutor):
    """Executor with ``fallback_to_sync`` pinned.

    torch.fx patches ``nn.Module.__call__`` process-wide while the main thread
    traces, so it can catch the teacher mid-forward. The teacher is frozen and
    side-effect-free, so recomputing it on the training thread is safe.
    """

    def __init__(self, device=None, timeout=300):
        super().__init__(
            device=device,
            timeout=timeout,
            error_policy="fallback_to_sync",
            name="lingbot-teacher",
        )


__all__ = ["AsyncTeacherRunner"]
