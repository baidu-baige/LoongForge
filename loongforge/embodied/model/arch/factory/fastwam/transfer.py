# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""fastwam's checkpoint rename table.

No ``modeling.py`` and no ``config.py`` here, unlike the other two packages in
``factory/``: the registry key ``fastwam`` already belongs to
``FastWAMPolicy``, and two classes claiming one key would make the winner
depend on import order. fastwam reaches the arch stack through a parity test
that builds the spec directly, so what it needs from this package is the
checkpoint translation and nothing else.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

from ..base import (DUAL_STREAM_WAN_RENAME_RULES, attach_proprio,
                    remap_state_dict)

__all__ = ["fastwam_to_arch"]


def fastwam_to_arch(payload: Mapping[str, Any], *,
                    drop: Iterable[str] = ()) -> dict[str, Any]:
    """Map a fastwam checkpoint onto ``WorldModel.state_dict()`` keys.

    No rules of its own: fastwam's expert key names are exactly the family's
    (``DUAL_STREAM_WAN_RENAME_RULES``). What it does need is the same
    ``"mot"`` / proprio sibling unwrapping, which is why this is a function and
    not an alias for the table.

    ``payload`` may be the whole checkpoint written by
    ``FastWAM.save_checkpoint`` (``mot/fastwam.py:1164-1177``: ``mot`` plus the
    siblings ``step`` / ``torch_dtype`` / ``proprio_encoder`` /
    ``optimizer``) or the ``mot`` subtree alone. The siblings this function
    does not consume are ignored rather than rejected: they are training state,
    not weights.

    Two things that path deliberately does **not** need:

      1. **No ``_extra_state`` filtering.** The saver already calls
         ``drop_extra_state`` (``:1169``, defined
         ``utils/state_dict.py:21-53``); those entries only exist at all when
         ``rmsnorm_impl="te"``.
      2. **No ``norm3`` special case.** fastwam's action blocks reuse the
         video ``DiTBlock`` (``wan/dit.py:426``), so their ``norm3`` is affine
         and the key is present -- which is why the fastwam provider leaves
         ``cross_attn_norm`` at its default ``True`` while the VPP2 provider
         sets it ``False`` (``vpp2/spec.py:219``). The remapping is unaffected
         either way; a mismatch would surface at ``strict=True``.

    A flat ``FastWAMPolicy.state_dict()`` from the trainer
    (``distributed/checkpoint.py:605``) is **not** accepted: its keys are
    prefixed ``core.video_expert.`` / ``core.action_expert.`` rather than
    ``mixtures.<stream>.``, and it also carries ``core.vae.*``. Convert that
    form with ``FastWAM.save_checkpoint`` first, so that "which file layout is
    this" stays a question asked once at load time.

    Args:
        payload: Either the full checkpoint dict or the ``mot`` subtree.
        drop: Key prefixes to discard, forwarded to ``remap_state_dict``.

    Returns:
        A dict keyed by the new architecture's parameter names.
    """
    tree = payload["mot"] if "mot" in payload else payload
    out = remap_state_dict(tree, DUAL_STREAM_WAN_RENAME_RULES, drop=drop)
    attach_proprio(out, payload)
    return out
