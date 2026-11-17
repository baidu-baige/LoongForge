# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""配置层核心：``Spec`` / ``build`` / provider 注册表。

签名以 ``docs/design/mot_spec_config.zh-CN.md`` §3.1 / §3.8 为准。

``Spec`` 借用 Megatron 的 ``ModuleSpec`` 形态：一个 impl + 一包参数 + 一包子 spec，
``build`` 递归构造。与 Megatron 的差别只有一处但很关键：
**键冲突抛错，不静默覆盖**（§3.8 的三处刻意严格之一）。

本模块**不导入 arch 下的任何其他模块** —— 它是依赖图的根，
保持无依赖是整棵树无环的前提。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Spec", "build", "register_provider", "get_provider",
           "provider_names"]


@dataclass(frozen=True)
class Spec:
    """「用什么实现、带什么参数」的声明。构造期的东西，不含张量。

    三层语义，缺一不可：

    1. ``impl`` 是**可调用对象**（类或工厂函数），不是字符串。
       字符串会把「这个实现存不存在」推迟到运行期；可调用对象让它
       在 import 期就报 NameError。
    2. ``params`` 是 ``impl`` 的关键字参数。``build`` 的 ``**overrides``
       与它合并，**键冲突抛错**。
    3. ``sub`` 是子 spec，键即 ``impl`` 的参数名，值被递归 build 之后
       作为该参数传入。
    """

    impl: Callable[..., Any]
    params: Mapping[str, Any] = field(default_factory=dict)
    sub: Mapping[str, "Spec"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.impl):
            raise TypeError(
                f"Spec.impl 必须是可调用对象（类或工厂函数），"
                f"得到 {self.impl!r}。字符串会把实现是否存在推迟到运行期"
            )
        clash = set(self.params) & set(self.sub)
        if clash:
            raise ValueError(
                f"Spec({_impl_name(self.impl)})：params 与 sub 的键冲突："
                f"{sorted(clash)}。同一个参数不许有两个来源"
            )

    def with_params(self, **extra: Any) -> Spec:
        """派生一个加了参数的新 spec。重名键抛错，与 ``build`` 同一条规则。"""
        clash = set(extra) & set(self.params)
        if clash:
            raise ValueError(
                f"Spec({_impl_name(self.impl)}).with_params：键 "
                f"{sorted(clash)} 已在 params 里"
            )
        return Spec(impl=self.impl, params={**self.params, **extra},
                    sub=self.sub)


def _impl_name(impl: Callable[..., Any]) -> str:
    """给错误消息用的实现名。lambda 也能给出个位置。"""
    name = getattr(impl, "__qualname__", None) or getattr(impl, "__name__", None)
    if name is None:
        return repr(impl)
    module = getattr(impl, "__module__", "?")
    return f"{module}.{name}"


def build(spec: Spec, /, **overrides: Any) -> Any:
    """递归构造 ``spec``。

    第一个形参是**仅限位置**的：``overrides`` 里几乎总有一个键叫 ``spec``
    （调用方把 ``StreamSpec`` / ``HeadSpec`` 注进去），不加 ``/`` 会撞名。

    ``overrides`` 是**调用方注入的上下文**（``spec=``、``geometry=`` 这类），
    与 ``spec.params`` 合并。**键冲突抛 ValueError** —— 这是刻意的严格：
    静默覆盖会让「YAML 里配的值到底生效了吗」变成一个要读代码才能回答的问题。

    子 spec 同样拿到 ``overrides``：上下文沿树向下传，所以子模块不必
    自己再去找 geometry。
    """
    clash = set(overrides) & set(spec.params)
    if clash:
        raise ValueError(
            f"build({_impl_name(spec.impl)})：overrides 与 spec.params 的键冲突："
            f"{sorted(clash)}。冲突不按优先级解决 —— 静默覆盖会让"
            "「配的值是否生效」无法从配置本身回答"
        )
    kwargs: dict[str, Any] = {**spec.params, **overrides}
    for name, child in spec.sub.items():
        kwargs[name] = build(child, **overrides)
    try:
        return spec.impl(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"build({_impl_name(spec.impl)}) 失败：{exc}。"
            f"传入的参数名为 {sorted(kwargs)}"
        ) from exc


# ---- provider 注册表 ----
# 键是 **model_type**（spec_config §3.8 的裁定）。模型身份的唯一出处是启动参数
# --model-name；model_type 是配置里那一个键，两者的映射由配置层负责。
# 六个拓扑族（dual_stream_wan 之类）**不在这里注册** —— 它们是工厂函数，
# 不是对外契约，YAML 与启动参数都看不见它们。

_PROVIDERS: dict[str, Callable[..., Any]] = {}


def register_provider(model_type: str) -> Callable[[Callable[..., Any]],
                                                   Callable[..., Any]]:
    """注册一个模型 provider。重名直接抛错。

    重名抛错而不是后注册者胜：两个文件注册同一个 ``model_type`` 时，
    胜者取决于 import 顺序，而 import 顺序会被 ``__init__.py`` 的改动悄悄改变。
    """

    def wrap(provider: Callable[..., Any]) -> Callable[..., Any]:
        if model_type in _PROVIDERS:
            prior = _PROVIDERS[model_type]
            raise ValueError(
                f"provider 重名：model_type={model_type!r} 已由 "
                f"{_impl_name(prior)} 注册"
            )
        _PROVIDERS[model_type] = provider
        return provider

    return wrap


def get_provider(model_type: str) -> Callable[..., Any]:
    """按 ``model_type`` 取 provider，缺失时列出全部已注册的键。"""
    try:
        return _PROVIDERS[model_type]
    except KeyError:
        raise KeyError(
            f"没有 model_type={model_type!r} 的 provider；"
            f"已注册：{sorted(_PROVIDERS)}"
        ) from None


def provider_names() -> tuple[str, ...]:
    """已注册的 ``model_type`` 全集。供 CLI 报错时列出可选值。"""
    return tuple(sorted(_PROVIDERS))
