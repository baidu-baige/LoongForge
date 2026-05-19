# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared mixin for GRooT CUDA graph wrappers.

Plugs GRooT-specific behavior into the model-agnostic
``BaseCudaGraphWrapper`` template via the documented hook API.
"""
from __future__ import annotations

from contextlib import contextmanager


class GrootGraphMixin:
    """Suppress GRooT's per-step loss allreduce while a CUDA graph is being
    captured or replayed.

    Loss allreduce performs NCCL ops, which are not capturable. The flag
    ``loongforge.train.embodied.sft_groot._SKIP_LOSS_ALLREDUCE`` is read by
    the loss reduction path; we set it inside the ctxmgr and restore the
    prior value (in ``finally``, so an in-graph error cannot leave the flag
    permanently corrupted)."""

    @contextmanager
    def _suppress_loss_allreduce(self):
        import loongforge.train.embodied.sft_groot as _sft_groot_mod

        _saved = getattr(_sft_groot_mod, "_SKIP_LOSS_ALLREDUCE", False)
        _sft_groot_mod._SKIP_LOSS_ALLREDUCE = True
        try:
            yield
        finally:
            _sft_groot_mod._SKIP_LOSS_ALLREDUCE = _saved
