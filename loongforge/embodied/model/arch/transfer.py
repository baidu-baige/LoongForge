# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Key remapping from legacy checkpoints to ``WorldModel.state_dict()``.

**This belongs to none of L0-L4**: it is a one-off migration tool, whereas the
five layers are the model itself. It lives here rather than under ``tools/``
because the right-hand column of the mapping table has to track the submodule
names in ``streams/blocks/`` and ``heads/`` -- a rename must happen in the
same commit, and across directories it would be missed.

Why an **ordered regex table** instead of one dict of all keys: the legacy
VPP2 checkpoint holds 2312 tensors (1298 video + 1014 action, with
``num_layers=40``), so a key-by-key table would be longer than the model and
would be entirely wrong for a different layer count. The rule table's size is
independent of the layer count.

One fact runs through the whole table: weights live in exactly three
namespaces (``mot_spec_dataflow.svg``) -- L2's ``blocks.<stream>.*`` and
``condition_encoders.<condition>.*``, plus L4's ``heads.<head>.*``. So the
legacy habit of growing the head inside the expert (``WanVideoDiT.head``) is a
**cross-layer move** here, not a prefix rename: it is the only class of rule
in this table that changes the tree shape, which is why it comes first.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from os import PathLike
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .builder import WorldModel

__all__ = ["PROPRIO_PAYLOAD_KEY", "RenameRules", "VPP2_RENAME_RULES",
           "load_vpp2_initializers", "remap_state_dict", "vpp2_to_arch"]

RenameRules = tuple[tuple[str, str], ...]


def remap_state_dict(state_dict: Mapping[str, Any], rules: RenameRules, *,
                     drop: Iterable[str] = ()) -> dict[str, Any]:
    """Rename keys by an **ordered** rule table; the first match wins.

    Three situations deliberately raise instead of being skipped:
      1. a key matches no rule -- skipping it shows up later as a strict-load
         "missing" error, and by then you cannot tell "the rules did not cover
         it" from "the checkpoint never had it";
      2. two source keys map to the same target key -- the later write would
         silently overwrite the earlier one;
      3. ``drop`` names a prefix that does not exist -- that drop entry is
         stale.

    ``drop`` is a **declared** discard (weights the legacy model has and the
    new architecture does not, for example ``video_kv_projections``, which is
    only built when ``use_video_tokens=True``). It is kept separate from "no
    rule matched" on purpose: the first is a decision, the second is an
    oversight.

    Args:
        state_dict: Source mapping of legacy key to tensor.
        rules: Ordered ``(pattern, replacement)`` pairs applied with
            ``re.subn(..., count=1)``.
        drop: Key prefixes to discard, each of which must match something.

    Returns:
        A new dict keyed by the new architecture's names.

    Raises:
        ValueError: On a mapping collision or a stale ``drop`` prefix.
        KeyError: If any key matches no rule.
    """
    dropped: set[str] = set()
    out: dict[str, Any] = {}
    sources: dict[str, str] = {}
    unmatched: list[str] = []
    prefixes = tuple(drop)
    for key, value in state_dict.items():
        hit = next((prefix for prefix in prefixes
                    if key.startswith(prefix)), None)
        if hit is not None:
            dropped.add(hit)
            continue
        for pattern, replacement in rules:
            new_key, count = re.subn(pattern, replacement, key, count=1)
            if count:
                if new_key in out:
                    raise ValueError(
                        f"remap collision: {key!r} and {sources[new_key]!r} "
                        f"both map to {new_key!r}. The later write would "
                        f"silently overwrite the earlier one"
                    )
                out[new_key] = value
                sources[new_key] = key
                break
        else:
            unmatched.append(key)
    if unmatched:
        raise KeyError(
            f"{len(unmatched)} keys match no rule, first 5: "
            f"{unmatched[:5]}. Skipping them turns into a strict-load "
            "missing error, and by then you cannot tell a rule gap from a "
            "key the ckpt never had"
        )
    stale = sorted(set(prefixes) - dropped)
    if stale:
        raise ValueError(
            f"drop prefixes {stale} matched no key in this ckpt; "
            "a stale drop silently discards real weights on the next ckpt"
        )
    return out


# ---- VPP2 ----

# The source namespace is ``mixtures.<stream>.``: saving goes through
# ``{"mot": self.mot.state_dict()}`` (``vpp2.py:1799``), and ``MoT`` has
# exactly one submodule, ``self.mixtures = nn.ModuleDict`` (``mot.py:28``).
# Inside ``VPP2``, ``video_expert`` / ``mot`` / ``dit`` are **three
# registrations of the same objects** (``vpp2.py:104-107``), so one tensor has
# three keys in ``VPP2.state_dict()`` -- we match the prefix that is on disk,
# not the in-memory aliases.
VPP2_RENAME_RULES: RenameRules = (
    # 1. Cross-layer move: legacy grows the output head inside the expert
    #    (``WanVideoDiT.head`` / ``ActionDiT.head``), while in the new
    #    architecture the head is a standalone L4 object.
    #    Must precede the catch-all rule 4, or it would be treated as an
    #    in-stream parameter.
    #    The video head's inner Linear is also named head, so the source key
    #    literally reads ``head.head.weight``.
    (r"^mixtures\.video\.head\.head\.", "heads.video_flow.head."),
    (r"^mixtures\.video\.head\.modulation$", "heads.video_flow.modulation"),
    #    The action head is a bare ``nn.Linear`` (``action_dit.py:150``), so it
    #    has no modulation and no inner head. The ``ActionHead`` at
    #    ``:20-33`` of the same file is dead code.
    (r"^mixtures\.action\.head\.", "heads.action_flow.head."),
    # 2. Condition projectors are filed by **slot name**: legacy gave each of
    #    the two conditions its own name (``img_emb`` / ``text_embedding``),
    #    while the new architecture unifies them under
    #    ``context_embedding.<slot>``, where the slot is ``ConditionSpec.name``.
    (r"^mixtures\.video\.img_emb\.proj\.",
     "blocks.video.context_embedding.clip_image.proj."),
    (r"^mixtures\.(video|action)\.text_embedding\.",
     r"blocks.\1.context_embedding.text."),
    # 3. The cross-attention KV side is grouped by slot name. Legacy i2v used
    #    ``k_img`` / ``v_img`` / ``norm_k_img`` for "the CLIP branch"
    #    (``wan_video_dit.py:183-185``) and split one context by assuming "the
    #    first 257 tokens are the image".
    #    In the new architecture it is just "the slot named clip_image", and
    #    its length is decided by the data.
    #    These three precede the text rule only for readability: ``k_img\.``
    #    and ``k\.`` cannot both match (the latter requires a dot right after
    #    k), so the order is in fact irrelevant.
    (r"^mixtures\.video\.blocks\.(\d+)\.cross_attn\.k_img\.",
     r"blocks.video.blocks.\1.cross_attn.kv.clip_image.k."),
    (r"^mixtures\.video\.blocks\.(\d+)\.cross_attn\.v_img\.",
     r"blocks.video.blocks.\1.cross_attn.kv.clip_image.v."),
    (r"^mixtures\.video\.blocks\.(\d+)\.cross_attn\.norm_k_img\.",
     r"blocks.video.blocks.\1.cross_attn.kv.clip_image.norm_k."),
    (r"^mixtures\.(video|action)\.blocks\.(\d+)\.cross_attn\.(k|v|norm_k)\.",
     r"blocks.\1.blocks.\2.cross_attn.kv.text.\3."),
    # 4. Everything else only swaps the top-level prefix: the submodule names
    #    ``self_attn.{q,k,v,o,norm_q,norm_k}`` / ``norm3`` / ``ffn.{0,2}`` /
    #    ``modulation`` / ``time_embedding.{0,2}`` / ``time_projection.1`` /
    #    ``patch_embedding`` / ``action_encoder`` are **deliberately** kept
    #    byte-identical to legacy in the L2 implementation, so this single
    #    rule covers the vast majority of the 2200-plus tensors.
    (r"^mixtures\.(video|action)\.", r"blocks.\1."),
)

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


PROPRIO_PAYLOAD_KEY = "proprio_encoder"
"""The key of the proprio encoder in legacy checkpoints, **a sibling of** ``"mot"``.

Written at ``vpp2.py:1800-1801``, read at ``:1841-1842``. It is not inside the
``mot`` subtree because the legacy ``nn.Linear`` hangs off ``VPP2`` rather than
off any single expert -- its output is concatenated into the context shared by
both streams, so assigning it to either stream would be wrong.

The new architecture states this plainly as the third parameter namespace,
``condition_encoders.<condition>.*``, so this top-level key lands exactly on
``condition_encoders.proprio.{weight,bias}`` (``TokenEncoder`` subclasses
``nn.Linear`` instead of wrapping one precisely so those two keys match
literally).
"""


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
    encoder = payload.get(PROPRIO_PAYLOAD_KEY)
    if encoder is not None:
        for name, value in encoder.items():
            out[f"condition_encoders.proprio.{name}"] = value
    return out


# MASTER ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES. Video-token paths are
# disabled in VPP2; they are NOT permission to leave arbitrary core weights
# random, or to discard unexpected tensors from an initializer.
_ACTION_BACKBONE_SKIP_PREFIXES = (
    "action_encoder.", "head.", "video_kv_projections.",
    "video_k_norms.", "video_token_encoder.",
)
_VPP2_RANDOM_KEYS = frozenset({
    "blocks.action.action_encoder.weight", "blocks.action.action_encoder.bias",
    "heads.action_flow.head.weight", "heads.action_flow.head.bias",
    "condition_encoders.proprio.weight", "condition_encoders.proprio.bias",
})


def _initializer_state(path: str | PathLike[str], formats: set[str],
                       field: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format") not in formats:
        raise ValueError(f"Invalid VPP2 {field} initializer format at {path!s}; "
                         f"expected {sorted(formats)}")
    state = payload.get(field)
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"VPP2 initializer at {path!s} has no nonempty {field}")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"VPP2 {field} entry {key!r} must be a named tensor")
        if value.is_meta:
            raise ValueError(f"VPP2 {field} tensor {key!r} has no checkpoint storage")
    return payload, state


def _strict_initializer_check(label: str, mapped: Mapping[str, torch.Tensor],
                              expected: Mapping[str, torch.Tensor]) -> None:
    missing = sorted(expected.keys() - mapped.keys())
    unexpected = sorted(mapped.keys() - expected.keys())
    mismatches = [
        f"{key}: source={tuple(mapped[key].shape)} target={tuple(expected[key].shape)}"
        for key in sorted(mapped.keys() & expected.keys())
        if mapped[key].shape != expected[key].shape
    ]
    if missing or unexpected or mismatches:
        raise RuntimeError(
            f"Strict VPP2 {label} check failed: missing={missing}, "
            f"unexpected={unexpected}, shape_mismatches={mismatches}")


def load_vpp2_initializers(
    core: WorldModel, video_path: str | PathLike[str],
    action_path: str | PathLike[str],
) -> dict[str, Any]:
    """Validate and load the split Wan2.1 VPP2 initializers into a WorldModel.

    Video coverage includes its head and every condition projector. Action
    coverage excludes ONLY MASTER's random action entry/output projections.
    Proprio is also kept as constructed, exactly as in MASTER's split load.
    The action backbone metadata (not dataset-dependent action_dim) must match
    the actual core. Counts are derived from that core, not hardcoded to 40
    layers: the production files have 1303 video and 1010 backbone tensors.

    Both files use CPU ``torch.load(mmap=True, weights_only=True)``; all key,
    shape and policy checks finish before any parameter is modified. An
    entirely meta core is validation-only, remains meta and consumes no weight
    storage. A materialized core is copied into with ``strict=True``, preserving
    parameter identity, device, dtype and the random tensors. Mixed meta/real
    parameters are rejected. Callers must initialize random tensors normally
    before loading (``to_empty`` alone does not initialize them).

    Returns a report with video/action-backbone tensor counts, sorted
    ``random_keys``, and ``validation_only``. No merged tensor allocation,
    checkpoint export, model materialization, or auxiliary tower load occurs.
    """
    from .builder import WorldModel

    if not isinstance(core, WorldModel) or core.spec.model_type != "vpp2":
        raise TypeError("load_vpp2_initializers expects an actual VPP2 WorldModel")
    parameters = tuple(core.parameters())
    meta_flags = {parameter.is_meta for parameter in parameters}
    if len(meta_flags) != 1:
        raise ValueError("VPP2 initializer core must be entirely meta or materialized")
    validation_only = meta_flags == {True}
    target = core.state_dict()
    video_expected = {
        key: value for key, value in target.items()
        if key.startswith(("blocks.video.", "heads.video_flow."))
    }
    action_expected = {
        key: value for key, value in target.items()
        if key.startswith(("blocks.action.", "heads.action_flow."))
        and key not in _VPP2_RANDOM_KEYS
    }
    random_keys = _VPP2_RANDOM_KEYS & target.keys()
    unexplained = target.keys() - video_expected.keys() - action_expected.keys() - random_keys
    if unexplained:
        raise RuntimeError(f"VPP2 core has unclassified parameters: {sorted(unexplained)}")

    _, video = _initializer_state(video_path, {"wan21_video_v1"}, "dit")
    action_payload, action = _initializer_state(
        action_path, {"wan21_action_v1", "vpp2_wan21_action_v1"},
        "backbone_state_dict")
    stream = core.blocks["action"]
    spec = stream.spec
    expected_meta = {
        "hidden_dim": spec.residual_width, "ffn_dim": spec.mlp_width,
        "num_layers": spec.num_layers, "num_heads": spec.num_heads,
        "attn_head_dim": spec.head_dim,
        "text_dim": core.spec.condition("text").width,
        "freq_dim": spec.timestep_embed_width,
        "eps": stream.blocks[0].norm1.eps,
    }
    meta = action_payload.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("VPP2 action initializer is missing its meta dictionary")
    for key, expected in expected_meta.items():
        got = meta.get(key)
        if (not isinstance(got, (int, float)) or isinstance(got, bool)
                or not math.isfinite(got)
                or (abs(got - expected) > 1e-12 if key == "eps" else got != expected)):
            raise ValueError(f"Action initializer meta.{key}={got!r} "
                             f"does not match core value {expected!r}")
    policy = action_payload.get("policy")
    if (not isinstance(policy, Mapping)
            or policy.get("projection_initialization") != "random"
            or policy.get("skip_prefixes") != list(_ACTION_BACKBONE_SKIP_PREFIXES)):
        raise ValueError("VPP2 action initializer policy must declare MASTER's "
                         "skip_prefixes and projection_initialization='random'")

    video_mapped = remap_state_dict(
        {f"mixtures.video.{key}": value for key, value in video.items()},
        VPP2_RENAME_RULES)
    action_mapped = remap_state_dict(
        {f"mixtures.action.{key}": value for key, value in action.items()},
        VPP2_RENAME_RULES)
    _strict_initializer_check("video", video_mapped, video_expected)
    _strict_initializer_check("action backbone", action_mapped, action_expected)
    if not validation_only:
        # Shallow dictionary only: checkpoint values remain mmap-backed until
        # PyTorch copies them into the existing destination parameters.
        target.update(video_mapped)
        target.update(action_mapped)
        core.load_state_dict(target, strict=True)
    return {
        "video_tensors": len(video_mapped),
        "action_backbone_tensors": len(action_mapped),
        "random_keys": tuple(sorted(random_keys)),
        "validation_only": validation_only,
    }
