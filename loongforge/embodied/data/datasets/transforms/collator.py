# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Base collator framework for DataLoader collate functions.

Classes:
    - PreparedBatch: Base dataclass for model-ready batch tensors
    - BasePreprocessor: Abstract base for collate functions
    - register_preprocessor: Decorator to register model-specific collators
"""

import importlib
import pkgutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Type

import torch


_PREPROCESSOR_REGISTRY: Dict[str, Type["BasePreprocessor"]] = {}
_DISCOVERED_PREPROCESSORS = False


def register_preprocessor(name: str):
    """Decorator to register a preprocessor class for a model."""
    def decorator(cls):
        _PREPROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator


@dataclass
class PreparedBatch:
    """Base class for preprocessed batch data.

    All tensor fields are on CPU after collation.
    Call .to(device) to move everything to GPU before forward().

    Tensor fields may be nested inside ``list`` / ``tuple`` / ``dict``
    containers; both :meth:`to` and :meth:`pin_memory` recurse into these
    containers so subclasses with nested structures (e.g. multi-view image
    dicts) do not need to override the base methods.
    """
    def to(self, device: torch.device) -> "PreparedBatch":
        """Move all tensor fields to the given device. Returns self."""
        for f in fields(self):
            setattr(self, f.name, _move_to_device(getattr(self, f.name), device))
        return self

    def pin_memory(self) -> "PreparedBatch":
        """Pin tensor fields so host-to-device copies can be non-blocking."""
        for f in fields(self):
            setattr(self, f.name, _pin_memory(getattr(self, f.name)))
        return self


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move CPU tensors in a nested structure to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    return value


def _pin_memory(value: Any) -> Any:
    """Recursively pin CPU tensors in a nested structure.

    Non-CPU tensors are returned as-is; ``RuntimeError`` from
    :meth:`torch.Tensor.pin_memory` (e.g. CUDA context unavailable, pinned
    pool exhausted) is caught and the original tensor is returned so a
    failure to pin degrades to a slower H2D copy rather than aborting.
    """
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            return value
        try:
            return value.pin_memory()
        except RuntimeError:
            return value
    if isinstance(value, list):
        return [_pin_memory(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_pin_memory(v) for v in value)
    if isinstance(value, dict):
        return {k: _pin_memory(v) for k, v in value.items()}
    return value


class BasePreprocessor:
    """Abstract base for model-specific DataLoader collate functions."""

    @classmethod
    def from_config(
        cls,
        model_cfg,
        data_cfg,
        training_args=None,
        dataset_stats=None,
        dataset=None,
    ) -> "BasePreprocessor":
        """Construct preprocessor from typed configs."""
        raise NotImplementedError(
            f"{cls.__name__} must implement from_config(model_cfg, data_cfg, ...) classmethod"
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> PreparedBatch:
        """Transform a list of dataset samples into a PreparedBatch."""
        raise NotImplementedError


def get_preprocessor(name: str) -> Type[BasePreprocessor]:
    """Look up a registered preprocessor class by name."""
    discover_preprocessors()
    if name not in _PREPROCESSOR_REGISTRY:
        raise ValueError(
            f"Unknown preprocessor '{name}'. "
            f"Available: {list(_PREPROCESSOR_REGISTRY.keys())}"
        )
    return _PREPROCESSOR_REGISTRY[name]


def build_preprocessor(
    name: str,
    model_cfg,
    data_cfg,
    training_args=None,
    dataset_stats=None,
    dataset=None,
) -> BasePreprocessor:
    """Instantiate a registered preprocessor via its from_config classmethod."""
    cls = get_preprocessor(name)
    return cls.from_config(
        model_cfg,
        data_cfg,
        training_args=training_args,
        dataset_stats=dataset_stats,
        dataset=dataset,
    )


def discover_preprocessors() -> None:
    """Import model transform subpackages so collator decorators can register.

    Model-specific collators now live under ``datasets/<model>/transforms/``;
    walk the ``datasets`` package and import each ``<model>.transforms`` package.
    """
    global _DISCOVERED_PREPROCESSORS
    if _DISCOVERED_PREPROCESSORS:
        return

    _DISCOVERED_PREPROCESSORS = True
    datasets_dir = Path(__file__).resolve().parent.parent
    datasets_pkg = "loongforge.embodied.data.datasets"
    for module_info in pkgutil.iter_modules([str(datasets_dir)]):
        if not module_info.ispkg or module_info.name.startswith("_"):
            continue
        if (datasets_dir / module_info.name / "transforms").is_dir():
            importlib.import_module(f"{datasets_pkg}.{module_info.name}.transforms")


@register_preprocessor("dummy")
class DummyPreprocessor(BasePreprocessor):
    """Pass-through preprocessor that returns examples as-is in a PreparedBatch."""

    @classmethod
    def from_config(
        cls,
        model_cfg,
        data_cfg,
        training_args=None,
        dataset_stats=None,
        dataset=None,
    ) -> "DummyPreprocessor":
        return cls()

    def __call__(self, examples: List[Dict[str, Any]]) -> PreparedBatch:
        return PreparedBatch()
