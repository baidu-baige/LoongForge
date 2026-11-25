# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""InternW0-Delta's checkpoint rename table and its four sibling payload keys."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

from ..base import (DUAL_STREAM_WAN_RENAME_RULES, RenameRules, attach_proprio,
                    remap_state_dict)

__all__ = ["FUTURE_DELTA_QUERY_PREFIX", "INTERNW0_DELTA_REJECTED",
           "INTERNW0_DELTA_RENAME_RULES", "internw0_delta_to_arch"]


_FUTURE_DELTA_RULES: RenameRules = (
    # The second cross-layer move in this family. The future-delta head is
    # ``Sequential(LayerNorm, Linear)`` grown **inside the video expert**
    # (``wan_video_dit.py:557-563``), while in the new architecture it is the
    # standalone L4 head ``future_delta``, a ``NormLinearHead`` whose two
    # submodules are ``norm`` and ``head``. Index 0 is the LayerNorm and index
    # 1 the Linear, so the two Sequential indices split into two rules rather
    # than one prefix swap.
    (r"^mixtures\.video\.future_delta_head\.0\.", "heads.future_delta.norm."),
    (r"^mixtures\.video\.future_delta_head\.1\.", "heads.future_delta.head."),
)

INTERNW0_DELTA_RENAME_RULES: RenameRules = (
    _FUTURE_DELTA_RULES + DUAL_STREAM_WAN_RENAME_RULES)
"""The family table plus InternW0-Delta's future-delta head.

Like fastwam it has no CLIP branch (``require_clip_embedding`` asserted off at
``wan_video_dit.py:474``), so its cross-attention KV goes through the plain
text rules. ``_FUTURE_DELTA_RULES`` comes first because the catch-all is last
in the shared table; among the non-catch-all groups the order is irrelevant,
they match disjoint key sets.
"""

FUTURE_DELTA_QUERY_PREFIX = "mixtures.video.future_delta_queries"
"""The learned future-delta queries, which the new architecture cannot host yet.

``[1, num_tokens, 3072]`` registered on the video expert
(``wan_video_dit.py:535-566``) and gated by ``future_delta.enabled``, which the
shipped config sets **true** (``configs/model/wam.yaml:64-65``) -- so a real
checkpoint normally carries it and this is not a rare path.

It has no home because the delta tokens have no declared position: they are
appended to the video sequence, yet ``GeometrySpec`` has no condition-frame or
delta fields and ``HiddenStateRef`` has no ``segment``, which is the same gap
the ``future_delta`` head's own comment records in
``providers/internw0_delta/spec.py``. Until that gap closes, pass this prefix
in ``drop`` to state "I know these queries are being discarded".

Leaving it out raises, and the check is explicit rather than left to
``remap_state_dict``: the catch-all rule matches this key perfectly well and
would rename it to ``blocks.video.future_delta_queries``, so without the check
the symptom is one unexpected key at ``strict=True`` with nothing to say why
it has no owner.
"""

INTERNW0_DELTA_REJECTED: dict[str, str] = {
    "action_proprio_encoder":
        "a second proprio encoder projecting into the VLM's context width "
        "(world_action_model.py:193-196), gated by understanding.enabled. The "
        "new architecture declares one proprio condition appended to the text "
        "slot and has no VLM slot for the second one to target",
    "understanding":
        "the Qwen-VL adapter weights (qwen_vl_encoder.py:737-741). The "
        "understanding tower is not part of the spec at all: no stream, no "
        "condition and no head describes it",
    "track_bridge":
        "the 4D distillation student with its learned track_queries "
        "(track_bridge.py:84-93). The spec does declare a track_align head, "
        "but its impl track_query_head raises NotImplementedError, so there "
        "is no module for these tensors to load into",
}
"""Top-level sibling payload keys this function refuses, and why.

Each is written by ``wam/training/checkpoint.py:17-24`` under its own config
gate. Refusing is not a gap in the remapping: for every one of them the new
architecture has no module at all, so the alternative is a ``strict=True``
failure one step later with a key list instead of a reason. Strip the entry
from the payload to say "load the rest without it".
"""


def internw0_delta_to_arch(payload: Mapping[str, Any], *,
                           drop: Iterable[str] = ()) -> dict[str, Any]:
    """Map an InternW0-Delta checkpoint onto ``WorldModel.state_dict()`` keys.

    ``payload`` may be the whole checkpoint written by
    ``save_wam_checkpoint`` (``wam/training/checkpoint.py:9-25``) or the
    ``mot`` subtree alone. ``step`` and ``torch_dtype`` are ignored; the four
    optional siblings are handled explicitly -- ``proprio_encoder`` is folded
    into ``condition_encoders.proprio`` (see ``attach_proprio``) and the other
    three are refused with the reasons in ``INTERNW0_DELTA_REJECTED``. An
    empty sibling counts as absent, which is the normal shape of
    ``understanding`` when the tower is off.

    Pass ``FUTURE_DELTA_QUERY_PREFIX`` in ``drop`` for any checkpoint trained
    with ``future_delta.enabled`` (the shipped default), and note that
    checkpoints predating the alias cleanup may spell expert keys with an extra
    ``.base.`` segment, which the legacy loader strips on the fly
    (``checkpoint.py:45-56``) and this function does not: converting such a
    file means running it through the legacy loader and re-saving, so the
    normalisation happens once rather than in two places.

    Args:
        payload: Either the full checkpoint dict or the ``mot`` subtree.
        drop: Key prefixes to discard, forwarded to ``remap_state_dict``.

    Returns:
        A dict keyed by the new architecture's parameter names.

    Raises:
        ValueError: The payload carries a non-empty refused sibling, or the
            future-delta queries without ``drop`` declaring them.
    """
    present = [key for key in INTERNW0_DELTA_REJECTED if payload.get(key)]
    if present:
        detail = "; ".join(f"{key}: {INTERNW0_DELTA_REJECTED[key]}"
                           for key in present)
        raise ValueError(
            f"this InternW0-Delta checkpoint carries {present}, which the new "
            f"architecture has no module for -- {detail}"
        )
    tree = payload["mot"] if "mot" in payload else payload
    declared = tuple(drop)
    if FUTURE_DELTA_QUERY_PREFIX in tree and not any(
            FUTURE_DELTA_QUERY_PREFIX.startswith(prefix)
            for prefix in declared):
        raise ValueError(
            f"this checkpoint carries {FUTURE_DELTA_QUERY_PREFIX!r}, the "
            "learned future-delta queries, which no module in the new "
            "architecture owns because the delta tokens have no declared "
            "position yet. Pass FUTURE_DELTA_QUERY_PREFIX in drop to discard "
            "them on purpose; the catch-all rule would otherwise rename them "
            "to a key that only fails at strict load"
        )
    out = remap_state_dict(tree, INTERNW0_DELTA_RENAME_RULES, drop=declared)
    attach_proprio(out, payload)
    return out
