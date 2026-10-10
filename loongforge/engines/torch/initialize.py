# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Seed, precision, deterministic mode, and backend precision initialization.

All functions use lazy torch imports so the module can be imported without
torch installed (e.g. for compileall or registry resolution).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, by_rank: bool = False) -> None:
    """Set random seed across all sources."""
    import random
    import numpy as np
    import torch as _torch

    if by_rank and _torch.distributed.is_initialized():
        seed += _torch.distributed.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    logger.info(f"Using random seed {seed}.")


def set_precision(allow_tf32: bool) -> None:
    """Set TF32 precision policy."""
    import torch as _torch
    _torch.backends.cudnn.allow_tf32 = allow_tf32
    _torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def set_deterministic() -> None:
    """Enable deterministic algorithms for reproducibility."""
    import torch as _torch

    if "PYTHONHASHSEED" not in os.environ:
        logger.warning(
            "PYTHONHASHSEED is not set; --deterministic-mode is best-effort without it. "
            "For full reproducibility, prepend `PYTHONHASHSEED=42` (or any fixed value) "
            "to your launch command \u2014 Python's hash seed is fixed at interpreter startup "
            "and cannot be set retroactively."
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
    _torch.use_deterministic_algorithms(True)


def resolve_dtype(dtype_str: str) -> "torch.dtype":
    """Convert string dtype to torch.dtype."""
    import torch as _torch
    mapping = {
        "bfloat16": _torch.bfloat16,
        "float16": _torch.float16,
        "float32": _torch.float32,
    }
    return mapping[dtype_str]


def cfg_bool(cfg: Any, key: str, default: bool = False) -> bool:
    """Read a boolean flag from a config namespace, accepting string truthy values."""
    value = getattr(cfg, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def set_backend_precision(model_cfg: Any) -> None:
    """Apply optional CUDA backend precision policy from model config."""
    import torch as _torch

    if not cfg_bool(model_cfg, "disable_reduced_precision_reduction", False):
        return
    matmul = _torch.backends.cuda.matmul
    if hasattr(matmul, "allow_bf16_reduced_precision_reduction"):
        matmul.allow_bf16_reduced_precision_reduction = False
    if hasattr(matmul, "allow_fp16_reduced_precision_reduction"):
        matmul.allow_fp16_reduced_precision_reduction = False
