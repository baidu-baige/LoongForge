# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Config layer core: ``Spec`` / ``build`` / provider registry.

Signatures follow ``docs/design/mot_spec_config.zh-CN.md`` §3.1 / §3.8.

``Spec`` borrows the shape of Megatron's ``ModuleSpec``: one impl, one bag of
keyword parameters, one bag of child specs, with ``build`` recursing over the
tree. There is exactly one deliberate difference from Megatron, and it matters:
**a key collision raises instead of being silently overwritten** (one of the
three places §3.8 chooses to be strict).

This module **imports nothing else under arch**. It is the root of the
dependency graph, and keeping it dependency-free is what keeps the whole tree
acyclic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Spec", "build", "register_provider", "get_provider",
           "provider_names"]


@dataclass(frozen=True)
class Spec:
    """Declaration of "which implementation, with which parameters".

    A construction-time object only; it holds no tensors.

    Three layers of semantics, all required:

    1. ``impl`` is a **callable** (a class or a factory function), not a string.
       A string would defer "does this implementation exist?" to run time;
       a callable makes a typo fail with ``NameError`` at import time.
    2. ``params`` are the keyword arguments of ``impl``. ``build``'s
       ``**overrides`` are merged into them, and **a key collision raises**.
    3. ``sub`` holds child specs; each key is a parameter name of ``impl``, and
       each value is built recursively and passed as that parameter.

    Attributes:
        impl: Class or factory function called by ``build``.
        params: Keyword arguments passed to ``impl``.
        sub: Child specs keyed by the ``impl`` parameter they fill.
    """

    impl: Callable[..., Any]
    params: Mapping[str, Any] = field(default_factory=dict)
    sub: Mapping[str, "Spec"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.impl):
            raise TypeError(
                f"Spec.impl must be callable (a class or a factory function), "
                f"got {self.impl!r}. A string would defer the question of "
                f"whether the implementation exists to run time"
            )
        clash = set(self.params) & set(self.sub)
        if clash:
            raise ValueError(
                f"Spec({_impl_name(self.impl)}): keys collide between params "
                f"and sub: {sorted(clash)}. One parameter must not have two "
                f"sources"
            )

    def with_params(self, **extra: Any) -> Spec:
        """Derive a new spec with extra parameters added.

        Args:
            **extra: Keyword arguments to add to ``params``.

        Returns:
            A new ``Spec`` with the same ``impl`` and ``sub`` and with
            ``params`` extended by ``extra``.

        Raises:
            ValueError: If a key of ``extra`` is already in ``params``. Same
                rule as ``build``: collisions are never resolved by priority.
        """
        clash = set(extra) & set(self.params)
        if clash:
            raise ValueError(
                f"Spec({_impl_name(self.impl)}).with_params: key "
                f"{sorted(clash)} is already in params"
            )
        return Spec(impl=self.impl, params={**self.params, **extra},
                    sub=self.sub)


def _impl_name(impl: Callable[..., Any]) -> str:
    """Implementation name used in error messages.

    Args:
        impl: Any callable, including a lambda.

    Returns:
        ``"<module>.<qualname>"`` when a name is available, otherwise
        ``repr(impl)``. A lambda still yields its defining module plus
        ``<lambda>``, which locates it.
    """
    name = getattr(impl, "__qualname__", None) or getattr(impl, "__name__", None)
    if name is None:
        return repr(impl)
    module = getattr(impl, "__module__", "?")
    return f"{module}.{name}"


def build(spec: Spec, /, **overrides: Any) -> Any:
    """Build ``spec`` recursively.

    The first parameter is **positional-only**: ``overrides`` almost always
    carries a key named ``spec`` (callers inject their ``StreamSpec`` /
    ``HeadSpec`` that way), which would collide with the parameter name
    without the ``/``.

    ``overrides`` is the **context injected by the caller** (``spec=``,
    ``geometry=`` and the like) and is merged with ``spec.params``. A key
    collision raises ``ValueError`` on purpose: silent overwriting turns "did
    the value I configured in YAML actually take effect?" into a question that
    can only be answered by reading code.

    Child specs receive the same ``overrides``, so context flows down the tree
    and a submodule never has to look up ``geometry`` on its own.

    Args:
        spec: The spec to build, positional-only.
        **overrides: Caller-injected context merged into ``spec.params`` and
            forwarded unchanged to every child spec.

    Returns:
        Whatever ``spec.impl(**kwargs)`` returns, with built children bound to
        their parameter names.

    Raises:
        ValueError: If a key of ``overrides`` is also in ``spec.params``.
        TypeError: If ``spec.impl`` rejects the assembled keyword arguments;
            the message lists the argument names that were passed.
    """
    clash = set(overrides) & set(spec.params)
    if clash:
        raise ValueError(
            f"build({_impl_name(spec.impl)}): keys collide between overrides "
            f"and spec.params: {sorted(clash)}. Collisions are not resolved "
            "by priority, because silent overwriting makes it impossible to "
            "tell from the config alone whether a configured value took effect"
        )
    kwargs: dict[str, Any] = {**spec.params, **overrides}
    for name, child in spec.sub.items():
        kwargs[name] = build(child, **overrides)
    try:
        return spec.impl(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"build({_impl_name(spec.impl)}) failed: {exc}. "
            f"The argument names passed were {sorted(kwargs)}"
        ) from exc


# ---- provider registry ----
# The key is **model_type** (the ruling in spec_config §3.8). The single source
# of a model's identity is the launch argument --model-name; model_type is the
# one key that appears in the config, and the config layer owns the mapping
# between the two.
# The six topology families (dual_stream_wan and friends) are **not registered
# here**: they are plain factory functions, not a public contract, and neither
# YAML nor the launch arguments can name them.

_PROVIDERS: dict[str, Callable[..., Any]] = {}


def register_provider(model_type: str) -> Callable[[Callable[..., Any]],
                                                   Callable[..., Any]]:
    """Register a model provider under ``model_type``.

    A duplicate name raises instead of letting the last registration win: when
    two files register the same ``model_type``, the winner depends on import
    order, and import order is silently changed by edits to ``__init__.py``.

    Args:
        model_type: Registry key, the same value as the ``model_type`` key in
            the model config.

    Returns:
        A decorator that registers the decorated callable and returns it
        unchanged.
    """

    def wrap(provider: Callable[..., Any]) -> Callable[..., Any]:
        if model_type in _PROVIDERS:
            prior = _PROVIDERS[model_type]
            raise ValueError(
                f"duplicate provider: model_type={model_type!r} is already "
                f"registered by {_impl_name(prior)}"
            )
        _PROVIDERS[model_type] = provider
        return provider

    return wrap


def get_provider(model_type: str) -> Callable[..., Any]:
    """Look up a provider by ``model_type``.

    Args:
        model_type: Registry key to look up.

    Returns:
        The registered provider callable.

    Raises:
        KeyError: If nothing is registered under ``model_type``; the message
            lists every registered key.
    """
    try:
        return _PROVIDERS[model_type]
    except KeyError:
        raise KeyError(
            f"no provider for model_type={model_type!r}; "
            f"registered: {sorted(_PROVIDERS)}"
        ) from None


def provider_names() -> tuple[str, ...]:
    """Every registered ``model_type``.

    Returns:
        The registered keys, sorted, for a CLI to list as valid choices when
        reporting an unknown model type.
    """
    return tuple(sorted(_PROVIDERS))
