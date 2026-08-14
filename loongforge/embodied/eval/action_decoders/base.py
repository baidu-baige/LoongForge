# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ActionDecoder infrastructure: base class + registry.

An ActionDecoder converts a model's raw action chunk (its own encoding, e.g.
ee6d ``pos+rot6d+grip``) into the action space a benchmark env expects. Concrete
decoders live in sibling modules grouped by **source encoding**:

* ``ee6d.py``  — models that output ee6d actions
* ``joint.py`` — models that output joint(-delta) actions

Two ways to author a decoder:

* **Stateless** transforms are plain functions ``fn(actions, ctx) -> actions``
  registered with ``@register_action_fn(key)``; they are wrapped in a
  :class:`FunctionDecoder` at build time. A single function may compose several
  primitive functions internally (e.g. rotation + joint for a hybrid action).
* **Stateful** transforms (needing cross-step memory + ``reset()``) are
  ``ActionDecoder`` subclasses registered with ``@register_action_decoder(key)``.

``build_action_decoder(key)`` returns an ``ActionDecoder`` instance either way
(empty key -> :class:`IdentityDecoder`). Pure rotation math lives in the sibling
``rotation.py`` leaf module.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ActionDecoder:
    """Convert a raw model action chunk into a benchmark env action chunk.

    Contract:
        ``__call__(actions, ctx) -> env_actions``
        - ``actions``: raw chunk ``[H, D]`` (already shape-normalised by
          ``call_predict_action`` server-side; form-B RoboTwin passes a
          single-row ``[1, D]`` chunk).
        - ``ctx``: per-step ``RuntimeContext`` dict. Stateless decoders ignore
          it; stateful / form-B decoders read fields like ``current_joint`` and
          ``is_fresh_chunk``.
        - returns: decoded chunk ``[H, D']`` in the benchmark's action space.

    Statefulness is expressed by overriding ``reset()`` (default no-op). The
    host (runner / bridge) calls ``reset()`` at each episode boundary.
    """

    def reset(self) -> None:
        """Clear per-episode state. Default: stateless -> no-op."""

    def __call__(self, actions: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        """Decode a raw model action chunk into a benchmark env action chunk."""
        raise NotImplementedError


class IdentityDecoder(ActionDecoder):
    """No-op decoder: model action encoding already matches the env action space.

    Used when ``resolve_action_decoder_key`` composes an identity key (source
    encoding == target action space, e.g. pi05 ``axis_angle`` × LIBERO
    ``axis_angle``). Returns the chunk unchanged.
    """

    def __call__(self, actions: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        """Return the action chunk unchanged."""
        return actions


class FunctionDecoder(ActionDecoder):
    """Adapt a stateless transform function into the ActionDecoder interface.

    Wraps ``fn(actions, ctx) -> actions`` so a plain function presents the same
    ``__call__`` + no-op ``reset()`` contract as a stateful decoder — callers
    never special-case function vs class. The wrapped ``fn`` may itself compose
    several primitive functions (rotation, joint, gripper, ...) for hybrid
    action encodings.
    """

    def __init__(self, fn: Callable[[np.ndarray, Dict[str, Any]], np.ndarray]) -> None:
        """Store the stateless transform function."""
        self._fn = fn

    def __call__(self, actions: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        """Apply the wrapped transform function."""
        return self._fn(actions, ctx)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Values are factories ``() -> ActionDecoder``: a stateful subclass (the class
# itself) or a ``FunctionDecoder``-wrapping lambda for a stateless function.

ACTION_DECODER_REGISTRY: Dict[str, Callable[[], ActionDecoder]] = {}


def register_action_decoder(key: str):
    """Decorator registering a **stateful** ActionDecoder subclass under ``key``.

    ``key`` is either an auto-composed ``{action_encoding}_to_{action_space}``
    (from ``resolve_action_decoder_key``) or a bridge-wired name (form-B
    RoboTwin, from ``_BRIDGE_WIRING``).
    """

    def decorator(cls):
        """Register ``cls`` (its own factory) and return it unchanged."""
        ACTION_DECODER_REGISTRY[key] = cls
        return cls

    return decorator


def register_action_fn(key: str):
    """Decorator registering a **stateless** transform function under ``key``.

    The function ``fn(actions, ctx) -> actions`` is wrapped in a
    :class:`FunctionDecoder` when built, so it satisfies the same interface as a
    stateful decoder without any class boilerplate.
    """

    def decorator(fn):
        """Register a FunctionDecoder factory around ``fn`` and return ``fn``."""
        ACTION_DECODER_REGISTRY[key] = lambda: FunctionDecoder(fn)
        return fn

    return decorator


def _auto_import_decoder_modules() -> None:
    """Import decoder modules so their decorators populate the registry."""
    for mod in (
        "loongforge.embodied.eval.action_decoders.ee6d",
        "loongforge.embodied.eval.action_decoders.ee_quat",
        "loongforge.embodied.eval.action_decoders.joint",
    ):
        try:
            importlib.import_module(mod)
        except ImportError as e:  # pragma: no cover - defensive
            logging.warning("Failed to import action decoder module %s: %s", mod, e)


def build_action_decoder(key: str) -> ActionDecoder:
    """Instantiate the ActionDecoder registered under ``key``.

    An empty key returns :class:`IdentityDecoder` (model encoding already
    matches the env action space — the composed key was identity).
    """
    if not key:
        return IdentityDecoder()
    _auto_import_decoder_modules()
    factory = ACTION_DECODER_REGISTRY.get(key)
    if factory is None:
        raise KeyError(
            f"Unknown action decoder key: {key!r}. "
            f"Registered: {sorted(ACTION_DECODER_REGISTRY.keys())}"
        )
    return factory()


def is_action_decoder_registered(key: str) -> bool:
    """Return whether ``key`` maps to a registered decoder.

    Empty key counts as available (it resolves to :class:`IdentityDecoder`).
    Used by ``resolve_action_decoder_key`` to decide between a registered
    decoder and the benchmark adapter's native action conversion.
    """
    if not key:
        return True
    _auto_import_decoder_modules()
    return key in ACTION_DECODER_REGISTRY
