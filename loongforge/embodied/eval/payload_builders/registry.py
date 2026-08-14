# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""PayloadBuilder registry.

Mirrors ``factories/registry.py`` — decorator-based registration plus a hard
coded auto-import list. Orchestrator asserts ``set(MODEL_FACTORY_REGISTRY) ==
set(PAYLOAD_BUILDER_REGISTRY)`` at startup so a Factory without a matching
PayloadBuilder (or vice-versa) fails fast.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Mapping, Optional, Type

from loongforge.embodied.eval.payload_builders.base import PayloadBuilder

PAYLOAD_BUILDER_REGISTRY: Dict[str, Type[PayloadBuilder]] = {}


def register_payload_builder(model_type: str):
    """Decorator that registers a PayloadBuilder subclass under ``model_type``.

    Usage::

        @register_payload_builder("pi05")
        class Pi05PayloadBuilder(PayloadBuilder):
            ...
    """

    def decorator(cls: Type[PayloadBuilder]) -> Type[PayloadBuilder]:
        """Register ``cls`` and return it unchanged."""
        PAYLOAD_BUILDER_REGISTRY[model_type] = cls
        return cls

    return decorator


def _auto_import_payload_builder_modules() -> None:
    """Import known PayloadBuilder modules so decorators fire on lookup."""
    _MODULES = [
        "loongforge.embodied.eval.payload_builders.pi05",
        "loongforge.embodied.eval.payload_builders.xvla",
        "loongforge.embodied.eval.payload_builders.groot_n1_6",
        "loongforge.embodied.eval.payload_builders.lingbot_va",
    ]
    for mod in _MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as e:  # pragma: no cover - defensive
            logging.warning("Failed to import payload builder module %s: %s", mod, e)


def build_payload_builder(
    model_type: str,
    yaml_model: Optional[Mapping[str, Any]] = None,
    yaml_server: Optional[Mapping[str, Any]] = None,
    yaml_benchmark: Optional[Mapping[str, Any]] = None,
) -> PayloadBuilder:
    """Instantiate the PayloadBuilder registered under ``model_type``."""
    _auto_import_payload_builder_modules()
    if model_type not in PAYLOAD_BUILDER_REGISTRY:
        raise SystemExit(
            f"Unsupported model_type for PayloadBuilder: {model_type!r}. "
            f"Registered: {sorted(PAYLOAD_BUILDER_REGISTRY.keys())}"
        )
    cls = PAYLOAD_BUILDER_REGISTRY[model_type]
    return cls(yaml_model=yaml_model, yaml_server=yaml_server, yaml_benchmark=yaml_benchmark)


def registered_model_types() -> set:
    """Return the set of currently registered PayloadBuilder model types."""
    _auto_import_payload_builder_modules()
    return set(PAYLOAD_BUILDER_REGISTRY.keys())
