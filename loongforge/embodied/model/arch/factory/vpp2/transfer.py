# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VPP2's checkpoint rename table: the family table plus a CLIP branch."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

from ..base import (CATCH_ALL, HEAD_RULES, TEXT_RULES, RenameRules,
                    attach_proprio, remap_state_dict)

__all__ = ["VPP2_RENAME_RULES", "VPP2_VIDEO_TOKEN_PREFIXES", "vpp2_to_arch"]


_CLIP_IMAGE_RULES: RenameRules = (
    # VPP2 is the only family member with an i2v CLIP branch. Legacy named it
    # ``img_emb`` on the projector side and ``k_img`` / ``v_img`` /
    # ``norm_k_img`` on the KV side (``wan_video_dit.py:183-185``), and split
    # one context by assuming "the first 257 tokens are the image".
    # In the new architecture it is just "the slot named clip_image", and its
    # length is decided by the data.
    # These precede the text rules only for readability: ``k_img\.`` and
    # ``k\.`` cannot both match (the latter requires a dot right after k), so
    # the order is in fact irrelevant.
    (r"^mixtures\.video\.img_emb\.proj\.",
     "blocks.video.context_embedding.clip_image.proj."),
    (r"^mixtures\.video\.blocks\.(\d+)\.cross_attn\.k_img\.",
     r"blocks.video.blocks.\1.cross_attn.kv.clip_image.k."),
    (r"^mixtures\.video\.blocks\.(\d+)\.cross_attn\.v_img\.",
     r"blocks.video.blocks.\1.cross_attn.kv.clip_image.v."),
    (r"^mixtures\.video\.blocks\.(\d+)\.cross_attn\.norm_k_img\.",
     r"blocks.video.blocks.\1.cross_attn.kv.clip_image.norm_k."),
)

VPP2_RENAME_RULES: RenameRules = (
    HEAD_RULES + _CLIP_IMAGE_RULES + TEXT_RULES + CATCH_ALL)
"""The family table plus VPP2's CLIP branch.

The legacy VPP2 checkpoint holds 2312 tensors (1298 video + 1014 action, with
``num_layers=40``); all but the CLIP ones are covered by the shared rules.
"""

# Weights the legacy model has that the new architecture does not build. Both
# groups hang off ``use_video_tokens``, and that path is hard-disabled in VPP2
# (``vpp2.py:96-102`` raises outright, ``:402`` forces False), so the
# checkpoints produced by both training configs contain none of these keys.
# They are therefore not declared by default: if a checkpoint from an enabled
# path ever shows up, pass them in explicitly and get a "this checkpoint has
# KVRoute weights" signal along the way.
VPP2_VIDEO_TOKEN_PREFIXES = (
    "mixtures.action.video_token_encoder.",
    "mixtures.action.video_kv_projections.",
    "mixtures.action.video_k_norms.",
)


def vpp2_to_arch(payload: Mapping[str, Any], *,
                 drop: Iterable[str] = ()) -> dict[str, Any]:
    """Map a VPP2 checkpoint onto ``WorldModel.state_dict()`` keys.

    ``payload`` may be the whole checkpoint (containing a ``"mot"`` key, the
    shape written at ``vpp2.py:1799``) or ``mot.state_dict()`` itself; both
    forms appear on the legacy loading side (``vpp2.py:1813-1819`` and
    ``weights.py:37-42``).

    ``PROPRIO_PAYLOAD_KEY`` is looked up at the **top level**, not inside the
    ``mot`` subtree, because that is where it lives. A bare
    ``mot.state_dict()`` payload therefore never carries the proprio encoder,
    and the difference between the two forms' key sets is exactly those two
    keys. That is a fact, not a defect: a genuinely missing encoder surfaces as
    a "missing" error at the ``strict=True`` step, and is not guessed at here.

    Args:
        payload: Either the full checkpoint dict or the ``mot`` subtree.
        drop: Key prefixes to discard, forwarded to ``remap_state_dict``.

    Returns:
        A dict keyed by the new architecture's parameter names.
    """
    tree = payload["mot"] if "mot" in payload else payload
    out = remap_state_dict(tree, VPP2_RENAME_RULES, drop=drop)
    attach_proprio(out, payload)
    return out
