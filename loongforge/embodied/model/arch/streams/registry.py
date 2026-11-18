# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 legacy on-ramp: ``StreamRegistry`` / ``StreamAdapter`` /
``plan_from_legacy_mask``.

Signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.7.

This module is migration scaffolding, not a long-term contract. It lets
existing models (an existing MoT block, an existing ``[B, 1, L, L]`` mask)
run first and be replaced by native implementations one at a time. Each of
the three entry points covers one migration path; once all are done this
file can be deleted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from ..layout import TokenLayout
from ..visibility.mask import AttentionPlan
from .spec import StreamSpec
from .state import QKV, StreamState

__all__ = ["StreamRegistry", "STREAMS", "StreamAdapter", "plan_from_legacy_mask"]


class StreamRegistry:
    """Registry of stream implementations.

    Keys are implementation names, values are factories producing a
    ``StreamModule``.

    This is not the provider registry in ``spec/core.py``: that one registers
    whole models keyed by ``model_type``, this one registers a single
    stream's block.
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]],
                                              Callable[..., Any]]:
        """Return a decorator registering one stream implementation.

        Args:
            name: Implementation name used as the registry key.

        Returns:
            A decorator that records the factory and returns it unchanged.
            The decorator raises ValueError on a duplicate name instead of
            overwriting silently.
        """

        def wrap(factory: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._factories:
                prior = self._factories[name]
                raise ValueError(
                    f"duplicate stream implementation name {name!r}; already "
                    f"registered by {prior.__module__}.{prior.__qualname__}"
                )
            self._factories[name] = factory
            return factory

        return wrap

    def get(self, name: str) -> Callable[..., Any]:
        """Look up a registered stream factory.

        Args:
            name: Implementation name.

        Returns:
            The factory callable.

        Raises:
            KeyError: If absent; the message lists all registered names.
        """
        try:
            return self._factories[name]
        except KeyError:
            raise KeyError(
                f"no stream implementation {name!r}; registered: "
                f"{sorted(self._factories)}"
            ) from None

    def names(self) -> tuple[str, ...]:
        """Return all registered implementation names, sorted."""
        return tuple(sorted(self._factories))


STREAMS = StreamRegistry()
"""Process-wide stream registry. Providers fetch block factories from it."""


class StreamAdapter:
    """Wrap an existing block into the four-stage ``StreamModule`` protocol.

    An existing block has the shape "one forward does a whole layer", while
    the protocol needs joint attention to be inserted between ``pre_attn``
    and ``post_attn``. The adapter therefore has to know how to split that
    legacy forward in half, and that cannot be done generically, so this
    class only fixes the signatures and each legacy model supplies its own
    subclass.

    Its only purpose during migration: run existing weights on the new
    engine with no checkpoint changes, compare numerics against the original
    implementation, then replace the block with a native one.

    Attributes:
        legacy: The wrapped legacy block.
        spec: Stream spec describing shapes for the wrapped block.
    """

    def __init__(self, legacy: Any, spec: StreamSpec) -> None:
        self.legacy = legacy
        self.spec = spec

    def prepare(self, x: Tensor, *,
                timestep: Tensor | None) -> StreamState:
        """Build the layer-0 state. Must be overridden per legacy model.

        Args:
            x: Raw input tensor for this stream.
            timestep: Diffusion timestep, or None.

        Raises:
            NotImplementedError: Always. Legacy blocks place patchify and
                modulation computation differently, so no generic split
                exists.
        """
        raise NotImplementedError(
            f"StreamAdapter.prepare must be implemented by the legacy-model "
            f"subclass {type(self).__name__}: legacy blocks place patchify "
            "and modulation computation differently, so no generic split "
            "exists"
        )

    def pre_attn(self, state: StreamState, *, local_layer_id: int) -> QKV:
        """Compute Q/K/V. Must be overridden per legacy model.

        Args:
            state: Current stream state.
            local_layer_id: Index into this stream's own layer list.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "StreamAdapter.pre_attn must be implemented by the legacy-model "
            "subclass"
        )

    def post_attn(self, state: StreamState, attn_out: Tensor, *,
                  local_layer_id: int) -> StreamState:
        """Apply the post-attention half. Must be overridden per model.

        Args:
            state: Current stream state.
            attn_out: This stream's slice of the joint attention output.
            local_layer_id: Index into this stream's own layer list.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "StreamAdapter.post_attn must be implemented by the legacy-model "
            "subclass"
        )

    def finalize(self, state: StreamState) -> StreamState:
        """Return the state unchanged.

        The default does nothing because a legacy block's final norm usually
        lives at the model top level rather than inside the block.

        Args:
            state: State after the last layer.

        Returns:
            The same state object.
        """
        return state


def plan_from_legacy_mask(mask: Tensor, layout: TokenLayout) -> AttentionPlan:
    """Convert a legacy boolean mask into the two channels of AttentionPlan.

    Args:
        mask: Boolean visibility mask. ``[L, L]``, ``[1, L, L]`` and
            ``[1, 1, L, L]`` are accepted: all three are batch-shared, so
            they are squeezed straight into the structural channel.
        layout: Token layout whose ``total`` must match the mask side length.

    Returns:
        An ``AttentionPlan`` whose ``structural`` channel is the squeezed
        mask and whose ``key_valid`` channel is left unset.

    Raises:
        ValueError: If the dtype is not bool, if a real batch axis survives
            the squeeze, or if the side length differs from
            ``layout.total``. A mask with B > 1 is rejected rather than
            downgraded: it violates INV-1, and pushing it into the
            structural channel would break BlockMask reuse and force the
            whole path back to dense. VPP2 clean-conditioning produces
            exactly such a mask, and its correct home is
            ``visibility.rules.noise_gate``, not this function.
    """
    if mask.dtype != torch.bool:
        raise ValueError(
            f"plan_from_legacy_mask: mask.dtype={mask.dtype}, must be bool. "
            "Convert an additive float bias to boolean visibility first"
        )
    squeezed = mask
    while squeezed.ndim > 2 and squeezed.shape[0] == 1:
        squeezed = squeezed.squeeze(0)
    if squeezed.ndim != 2:
        raise ValueError(
            f"plan_from_legacy_mask: shape {tuple(mask.shape)} carries a "
            "batch axis, which violates INV-1. Per-sample visibility goes "
            "through AttentionPlan.key_valid, noise-level dependent "
            "visibility through visibility.rules.noise_gate"
        )
    expected = (layout.total, layout.total)
    if tuple(squeezed.shape) != expected:
        raise ValueError(
            f"plan_from_legacy_mask: shape after squeeze "
            f"{tuple(squeezed.shape)} != {expected} "
            f"(layout.total={layout.total})"
        )
    return AttentionPlan(structural=squeezed, layout=layout)
