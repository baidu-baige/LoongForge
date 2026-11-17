# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""配置层：``Spec`` / ``build`` / provider 注册表 / ``WorldModelSpec``。

``core`` 是依赖图的根（无依赖），所以它在这里**直接导入**。
``world`` 反过来依赖 L0–L4 的全部模块，而那些模块又要从本包取 ``Spec`` ——
因此 ``world`` 走 PEP 562 的惰性导出：``arch.spec.WorldModelSpec`` 照常可用，
但 ``import arch.spec.core`` 不会把 ``world`` 连带拉起来。
这是整棵树无环的唯一一处技巧，别在别的 ``__init__`` 里复制它。
"""

from __future__ import annotations

from typing import Any

from .core import (
    Spec, build, get_provider, provider_names, register_provider,
)

__all__ = [
    "Spec", "build", "register_provider", "get_provider", "provider_names",
    "WorldModelSpec", "build_world_model",
]

_LAZY = {"WorldModelSpec", "build_world_model"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import world

        return getattr(world, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
