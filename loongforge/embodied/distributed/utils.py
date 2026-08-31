# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Distributed model inspection and argument helpers."""

import inspect
import os
from fnmatch import fnmatchcase

import torch
import torch.nn as nn


def is_rank_zero() -> bool:
    """Rank 0 check covering single-process, torchrun, and dist-initialized cases."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # Fallback to env (torchrun sets RANK before init); single process defaults to 0.
    return int(os.environ.get("RANK", "0")) == 0


def filter_kwargs(callable_obj, kwargs: dict) -> dict:
    """Drop kwargs unsupported by a callable's signature."""
    supported_params = set(inspect.signature(callable_obj).parameters)
    return {key: value for key, value in kwargs.items() if key in supported_params}


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip wrappers with ``.module`` such as DDP; FSDP2 fully_shard is in-place."""
    if hasattr(model, "module"):
        return model.module
    return model


def is_module_path_matching_pattern(pattern: str, module_path: str) -> bool:
    """Return True if ``module_path`` matches ``pattern`` segment-by-segment.

    Both strings are split on ``'.'`` into segments. A match requires:

    1. **Equal depth**: the number of segments must be identical, so a
       wildcard can never span a dot boundary.  For example, ``"encoder.*"``
       matches ``"encoder.layer"`` but NOT ``"encoder.layer.0"``.

    2. **Per-segment fnmatch**: each segment of ``pattern`` is tested against
       the corresponding segment of ``module_path`` using :func:`fnmatchcase`,
       which supports ``*`` (any substring) and ``?`` (any single character)
       within a single segment only.

    Example::
        is_module_path_matching_pattern("encoder.*", "encoder.layer")   # True
        is_module_path_matching_pattern("encoder.*", "encoder.layer.0") # False  (depth mismatch)
        is_module_path_matching_pattern("layers.*.attn", "layers.3.attn") # True
    """
    pattern_segments = pattern.split(".")
    module_path_segments = module_path.split(".")
    return len(pattern_segments) == len(module_path_segments) and all(
        fnmatchcase(module_path_segment, pattern_segment)
        for pattern_segment, module_path_segment in zip(
            pattern_segments,
            module_path_segments,
        )
    )


def unwrap_checkpoint_module(module: nn.Module) -> nn.Module:
    """Return the original module stored inside a checkpoint wrapper, if any.

    ``checkpoint_wrapper`` saves the wrapped module under the attribute
    ``_checkpoint_wrapped_module``.  If the module was not wrapped, it is
    returned unchanged.
    """
    return getattr(module, "_checkpoint_wrapped_module", module)


def is_mixed_param_dtype(module: nn.Module, trainable_only: bool = False) -> bool:
    """Return whether the model contains parameters of more than one dtype.

    Iterates over all modules and inspects only their **direct** parameters
    (``recurse=False``), returning True as soon as a second distinct dtype
    is observed.

    Args:
        model: Model to inspect.
        trainable_only: If True, only consider parameters with ``requires_grad=True``.

    Returns:
        True if two or more distinct parameter dtypes are found, else False.
    """
    seen_dtype: torch.dtype | None = None
    for submodule in module.modules():
        for param in submodule.parameters(recurse=False):
            if trainable_only and not param.requires_grad:
                continue
            if seen_dtype is None:
                seen_dtype = param.dtype
            elif param.dtype != seen_dtype:
                return True
    return False
