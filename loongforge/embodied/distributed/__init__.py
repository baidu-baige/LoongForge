# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Core training infrastructure — pure PyTorch native, no third-party training libs.

The contracts the replicated-sharded strategy shares with model code live in the
``replicated_sharded_utils`` subpackage; nothing here depends on them.
"""

from .context import DistributedContext
from .parallel import wrap_model
from .utils import is_rank_zero, unwrap_model

__all__ = [
    "DistributedContext",
    "is_rank_zero",
    "wrap_model",
    "unwrap_model",
]
