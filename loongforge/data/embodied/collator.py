# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Base collator framework for DataLoader collate functions.

Classes:
    - PreparedBatch: Base dataclass for model-ready batch tensors
    - BaseCollator: Abstract base for collate functions
    - DummyCollator: Pass-through collator for ``model_type="dummy"``

Registration and lookup by model type live in ``embodied/registry.py``.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, List

import torch

from loongforge.data.embodied.registry import register_collator


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


class BaseCollator:
    """Abstract base for model-specific DataLoader collate functions."""

    @classmethod
    def from_config(
        cls,
        model_cfg,
        data_cfg,
        training_args=None,
        dataset_stats=None,
        dataset=None,
    ) -> "BaseCollator":
        """Construct collator from typed configs."""
        raise NotImplementedError(
            f"{cls.__name__} must implement from_config(model_cfg, data_cfg, ...) classmethod"
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> PreparedBatch:
        """Transform a list of dataset samples into a PreparedBatch."""
        raise NotImplementedError


@register_collator("dummy")
class DummyCollator(BaseCollator):
    """Pass-through collator that returns examples as-is in a PreparedBatch."""

    @classmethod
    def from_config(
        cls,
        model_cfg,
        data_cfg,
        training_args=None,
        dataset_stats=None,
        dataset=None,
    ) -> "DummyCollator":
        return cls()

    def __call__(self, examples: List[Dict[str, Any]]) -> PreparedBatch:
        return PreparedBatch()
