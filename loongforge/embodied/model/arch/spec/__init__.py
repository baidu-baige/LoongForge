# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Config layer: ``Spec`` / ``build`` / provider registry / ``WorldModelSpec``.

``core`` is the root of the dependency graph (it imports nothing from ``arch``),
so it is imported eagerly here. ``world`` goes the other way: it depends on every
module in L0-L4, and those modules in turn import ``Spec`` from this package.
``world`` is therefore exported lazily through PEP 562 ``__getattr__``:
``arch.spec.WorldModelSpec`` still resolves, but ``import arch.spec.core`` does
not drag ``world`` in and create an import cycle.
This is the only place in the tree that uses this trick; do not copy it into
other ``__init__`` files.
"""

from __future__ import annotations

from typing import Any

from .core import (
    Spec, build, get_provider, provider_names, register_provider,
)

__all__ = [
    "Spec", "build", "register_provider", "get_provider", "provider_names",
    "WorldModelSpec", "build_world_model_spec",
]

_LAZY = {"WorldModelSpec", "build_world_model_spec"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import world

        return getattr(world, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
