# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Distributed model inspection and argument helpers."""

import inspect
import os

import torch
import torch.nn as nn


def is_rank_zero() -> bool:
    """Rank 0 check covering single-process, torchrun, and dist-initialized cases."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # Fallback to env (torchrun sets RANK before init); single process defaults to 0.
    return int(os.environ.get("RANK", "0")) == 0



def parse_optional_int_list(value) -> list[int] | None:
    """Parse an optional CLI/list value into ``list[int]``.

    Accepts ``None``, an existing list/tuple, or a comma-separated string such
    as ``"25,50,100"``. Empty strings or strings with only separators return
    ``None`` so optional kwargs can stay unset.
    """
    if value is None:
        return None

    # Some callers may already normalize CLI input into a sequence.
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]

    # Keep string parsing permissive for values like "25, 50,100".
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return [int(item) for item in items] if items else None


def filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    """Drop kwargs unsupported by a callable's signature."""
    supported_params = set(inspect.signature(callable_obj).parameters)
    return {key: value for key, value in kwargs.items() if key in supported_params}


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip wrappers with ``.module`` such as DDP; FSDP2 fully_shard is in-place."""
    if hasattr(model, "module"):
        return model.module
    return model


def is_container_module(module: nn.Module) -> bool:
    """Return True for modules that are traversal containers without forward."""
    return isinstance(module, (nn.ModuleList, nn.ModuleDict))


def module_params(
    module: nn.Module,
    recurse: bool = True,
    excluded_param_ids: set[int] | None = None,
) -> list[nn.Parameter]:
    """Return unique parameters, optionally excluding ids already managed."""
    params = []
    seen = set()
    excluded_param_ids = excluded_param_ids or set()
    for param in module.parameters(recurse=recurse):
        param_id = id(param)
        if param_id in seen:
            continue
        if param_id in excluded_param_ids:
            continue
        params.append(param)
        seen.add(param_id)
    return params


def module_param_dtypes(
    module: nn.Module,
    excluded_param_ids: set[int] | None = None,
) -> set[torch.dtype]:
    """Return dtypes for remaining unique parameters in a module tree."""
    return {param.dtype for param in module_params(module, excluded_param_ids=excluded_param_ids)}


def module_param_numel(
    module: nn.Module,
    excluded_param_ids: set[int] | None = None,
) -> int:
    """Return total numel for unique parameters in a module tree."""
    return sum(param.numel() for param in module_params(module, excluded_param_ids=excluded_param_ids))


def get_module_names_by_dtype(model: nn.Module, trainable_only: bool = False) -> dict:
    """Group submodule names by the dtype of their direct parameters.

    Modules without direct parameters are omitted. If a module owns direct
    parameters with multiple dtypes, its name appears under each dtype.

    Args:
        model: Model to inspect.
        trainable_only: If True, only count parameters with requires_grad=True.
    """
    modules_by_dtype = {}
    for module_name, module in model.named_modules():
        display_name = module_name or "<root>"
        direct_dtypes = set()
        for param in module.parameters(recurse=False):
            if trainable_only and not param.requires_grad:
                continue
            direct_dtypes.add(param.dtype)
        for dtype in direct_dtypes:
            modules_by_dtype.setdefault(dtype, []).append(display_name)
    return modules_by_dtype
