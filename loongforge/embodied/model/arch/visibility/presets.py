# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Common L1 rule combinations (Lane A).

The signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §3.6.

Visibility is algorithm, and **not one step of the whole path goes through
YAML**. Rules and combinators are Python expressions; a preset is just a reuse
function for "these few combinations keep showing up together", and its only
callers live in ``spec/providers/``. Three conventions:

1. **Preset names are not an external contract**; renaming one is a pure
   refactor and does not go through master plan §9.
2. **Presets are not mandatory.** A provider may write
   ``same_stream() & frame_causal(w)`` directly; a combination only earns a
   place in this file once two or more models reuse it.
3. **A monolithic model's provider may leave visibility unset entirely**; the
   default is ``full_bidirectional``.
"""

from __future__ import annotations

from collections.abc import Callable

from .combine import ALWAYS_VISIBLE, Visibility
from .rules import chunk_causal, frame_causal, noise_gate, prefix_only

__all__ = ["PRESETS", "preset", "get_preset"]

PRESETS: dict[str, Callable[..., Visibility]] = {}


def preset(name: str) -> Callable[[Callable[..., Visibility]],
                                  Callable[..., Visibility]]:
    """Register a common combination. A duplicate name raises, never overwrites.

    Args:
        name: Preset name to register under.

    Returns:
        A decorator registering the function in ``PRESETS``.
    """

    def wrap(fn: Callable[..., Visibility]) -> Callable[..., Visibility]:
        if name in PRESETS:
            raise ValueError(
                f"duplicate visibility preset name: {name!r} already "
                f"registered by {PRESETS[name].__module__}."
                f"{PRESETS[name].__qualname__}"
            )
        PRESETS[name] = fn
        return fn

    return wrap


def get_preset(name: str) -> Callable[..., Visibility]:
    """Look a preset up by name.

    Args:
        name: Registered preset name.

    Returns:
        The preset factory function.

    Raises:
        KeyError: If the name is not registered; all registered names are
            listed.
    """
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(
            f"no visibility preset {name!r}; registered: {sorted(PRESETS)}"
        ) from None


# ---- Five presets. The bodies are **real combination expressions**; only the
# nine rules being combined are Lane A implementations ----
# The extra parameters beyond interfaces §3.6 (prefix_tokens / threshold) are
# demanded by the combinations themselves: a preset cannot see the geometry, so
# how many tokens the first frame has can only be passed in by the provider.
# Neither preset names nor their parameters are an external contract; changing
# them is a pure refactor (§3.6, convention 1).


@preset("full_bidirectional")
def _full_bidirectional() -> Visibility:
    """Fully bidirectional.

    **The default for monolithic models and for motus**; it compiles to
    structural=None.

    Returns:
        The always-visible visibility.
    """
    return ALWAYS_VISIBLE


@preset("first_frame_causal")
def _first_frame_causal(prefix_tokens: int) -> Visibility:
    """Frame causality plus a first frame visible to every frame.

    Args:
        prefix_tokens: Number of leading tokens making up the first frame.

    Returns:
        The combined visibility.
    """
    return frame_causal() | prefix_only(prefix_tokens)


@preset("sliding_window_chunk")
def _sliding_window_chunk(window: int) -> Visibility:
    """Chunk causality plus a sliding window on the frame side.

    Args:
        window: Number of most recent frames visible.

    Returns:
        The combined visibility.
    """
    return chunk_causal() & frame_causal(window)


@preset("clean_conditioned")
def _clean_conditioned(threshold: int) -> Visibility:
    """Action stream sees only video keys with noise step <= threshold (VPP2).

    Args:
        threshold: Highest noise step that stays visible.

    Returns:
        The noise-gate visibility.
    """
    return noise_gate(threshold)


@preset("noise_gated_history")
def _noise_gated_history(window: int, threshold: int = 0) -> Visibility:
    """History within the last ``window`` frames and noise step <= threshold.

    ``noise_gated_history(3)`` as written by the VPP2 provider is exactly this:
    action queries read only the clean video keys of the last 3 frames.

    Args:
        window: Number of most recent frames visible.
        threshold: Highest noise step that stays visible.

    Returns:
        The combined visibility.
    """
    return frame_causal(window) & noise_gate(threshold)
