# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Key remapping from a legacy VPP2 ckpt to the five-layer arch, and the bar.

The acceptance bar is four steps: remap, load with ``strict=True`` and report
zero missing and zero unexpected keys, run one training forward producing a
0-dim loss, and after backward find a gradient on every parameter. Each step
fails with a different symptom, which is why they are four test cases and not
one.

No real 14B checkpoint is needed: the source-side key names are generated from
``model.state_dict()`` by the **inverse** of the rule table (``_to_legacy``).
So "which keys exist on the source side" is pinned down by two tables at once:
if the forward table misses a key class the inverse table cannot produce it,
and if the inverse table gets a key class wrong the forward table raises at
"matches no rule". The two tables are written independently and check each
other, which is harder to get wrong than one table plus a hardcoded list.

The inverse table is deliberately not in ``transfer.py``: the production path
has one direction only, legacy to new arch, and the reverse exists solely to
build fixtures here. Putting it in the module under test would suggest it can
export a ckpt backwards, a path that has never been validated.
"""

from __future__ import annotations

import re

import pytest
import torch

from loongforge.embodied.model.arch.transfer import (
    PROPRIO_PAYLOAD_KEY, VPP2_RENAME_RULES, VPP2_VIDEO_TOKEN_PREFIXES,
    remap_state_dict, vpp2_to_arch)

from test_arch_blocks import tiny_spec
from test_arch_world_model import fake_batch, world

# New arch key -> legacy key. Dual to ``VPP2_RENAME_RULES`` one for one, in the
# same order: special cases first, wildcards last, since the final rule
# ``^blocks\.(video|action)\.`` swallows everything.
LEGACY_RULES: tuple[tuple[str, str], ...] = (
    (r"^heads\.video_flow\.head\.", "mixtures.video.head.head."),
    (r"^heads\.video_flow\.modulation$", "mixtures.video.head.modulation"),
    (r"^heads\.action_flow\.head\.", "mixtures.action.head."),
    # The proprio encoder is a top-level payload key alongside ``mot``, not
    # under ``mixtures``; ``legacy_payload`` uses this prefix to lift it out
    # into its own level.
    (r"^condition_encoders\.proprio\.", f"{PROPRIO_PAYLOAD_KEY}."),
    (r"^blocks\.video\.context_embedding\.clip_image\.proj\.",
     "mixtures.video.img_emb.proj."),
    (r"^blocks\.(video|action)\.context_embedding\.text\.",
     r"mixtures.\1.text_embedding."),
    (r"^blocks\.video\.blocks\.(\d+)\.cross_attn\.kv\.clip_image\.(k|v)\.",
     r"mixtures.video.blocks.\1.cross_attn.\2_img."),
    (r"^blocks\.video\.blocks\.(\d+)\.cross_attn\.kv\.clip_image\.norm_k\.",
     r"mixtures.video.blocks.\1.cross_attn.norm_k_img."),
    (r"^blocks\.(video|action)\.blocks\.(\d+)\.cross_attn\.kv\.text\.",
     r"mixtures.\1.blocks.\2.cross_attn."),
    (r"^blocks\.(video|action)\.", r"mixtures.\1."),
)


def _to_legacy(key: str) -> str:
    """Map one new-arch key to its legacy ckpt name; raise if no rule matches."""
    for pattern, replacement in LEGACY_RULES:
        new_key, count = re.subn(pattern, replacement, key, count=1)
        if count:
            return new_key
    raise AssertionError(f"no inverse rule covers {key!r}")


def legacy_payload(model) -> dict[str, object]:
    """Build a legacy-shaped **whole ckpt**: ``{"mot": ..., "proprio_encoder": ...}``.

    Two levels, not one, because the legacy file has two
    (``vpp2.py:1799-1801``): the ``mot`` subtree holds the two experts, and the
    proprio encoder sits beside it because it hangs off ``VPP2`` itself and
    belongs to neither expert.

    The values deliberately differ from the model's current parameters;
    otherwise "loaded successfully" and "loaded nothing" would be numerically
    indistinguishable.
    """
    flat = {_to_legacy(key): torch.randn_like(value)
            for key, value in model.state_dict().items()}
    prefix = f"{PROPRIO_PAYLOAD_KEY}."
    encoder = {key[len(prefix):]: flat.pop(key)
               for key in [k for k in flat if k.startswith(prefix)]}
    return {"mot": flat, PROPRIO_PAYLOAD_KEY: encoder}


def mot_tree(payload: dict) -> dict[str, torch.Tensor]:
    """The payload's ``mot`` subtree, for cases that inject one bad key."""
    return payload["mot"]


# ---- Key sets ----


def test_remapped_keys_equal_the_model_state_dict_exactly():
    """The remapped key set **equals** ``state_dict()``; containment is not enough.

    Each direction is one kind of load failure: a key short is missing, a key
    over is unexpected. Set equality pins both, and the difference is listed
    in the failure message.
    """
    model = world(tiny_spec("arch_vpp2"))
    remapped = vpp2_to_arch(legacy_payload(model))
    want = set(model.state_dict())
    assert set(remapped) - want == set()
    assert want - set(remapped) == set()


def test_payload_may_be_wrapped_in_mot_or_bare():
    """Both payload shapes are accepted, differing **exactly** in proprio's keys.

    Legacy readers produced both forms (a whole ckpt, or just
    ``mot.state_dict()``), so the entry point has to recognise them itself
    instead of making callers remember which form a file uses. The bare form
    lacking proprio is not a defect: those two tensors were never in the
    ``mot`` subtree, and a genuine absence is reported as missing by
    ``strict=True`` rather than guessed at during remapping.
    """
    model = world(tiny_spec("arch_vpp2"))
    payload = legacy_payload(model)
    full = set(vpp2_to_arch(payload))
    bare = set(vpp2_to_arch(mot_tree(payload)))
    assert full - bare == {"condition_encoders.proprio.weight",
                           "condition_encoders.proprio.bias"}
    assert bare - full == set()


# ---- The acceptance bar ----


def test_strict_load_reports_nothing_missing_and_nothing_unexpected():
    """Both lists from a ``strict=True`` load must be empty.

    Under ``strict=False`` the two lists are just a return value nobody reads,
    so a set of weights that never entered the model still trains, with a
    curve that merely converges more slowly. Hence strict here.
    """
    model = world(tiny_spec("arch_vpp2"))
    report = model.load_state_dict(vpp2_to_arch(legacy_payload(model)),
                                   strict=True)
    assert not report.missing_keys
    assert not report.unexpected_keys


def test_loaded_values_land_on_the_parameters():
    """Matching keys is not the same as transferred values, so compare tensors.

    The usual cause of right shapes with unchanged contents is feeding the
    remapped dict to a different model instance.
    """
    model = world(tiny_spec("arch_vpp2"))
    payload = legacy_payload(model)
    remapped = vpp2_to_arch(payload)
    model.load_state_dict(remapped, strict=True)
    loaded = model.state_dict()
    for key, value in remapped.items():
        assert torch.equal(loaded[key], value), key


def test_loaded_model_trains_one_step():
    """After loading, one forward yields a 0-dim loss and backward reaches all.

    These are the last two steps of the acceptance bar, placed after the load
    rather than on their own: "it loads" and "it still trains once loaded" are
    two things. A head whose weights were loaded but which no loss term uses
    shows up only at the backward step, as a parameter without a gradient.
    """
    spec = tiny_spec("arch_vpp2")
    model = world(spec)
    model.load_state_dict(vpp2_to_arch(legacy_payload(model)), strict=True)
    loss, metrics = model(fake_batch(spec))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["loss"].ndim == 0
    loss.backward()
    missing = [name for name, param in model.named_parameters()
               if param.grad is None]
    assert not missing, missing


# ---- Three paths that raise on purpose ----


def test_key_from_an_unknown_stream_raises_instead_of_being_skipped():
    """A third stream on the source side raises ``KeyError`` instead of skipping.

    This happens when a three-expert ckpt is fed in. Skipping would show up as
    a missing key at strict load, where "no rule covered it" can no longer be
    told apart from "the ckpt never had this weight", and the two call for
    opposite fixes.

    Note **which keys reach this check**: the last rule
    ``^mixtures\\.(video|action)\\.`` is a wildcard, so an extra submodule
    under one of the two known streams is not reported here but becomes an
    unexpected key at strict load. Both are hard failures, one step apart in
    timing, and in exchange the rule table's size is independent of the layer
    count and of the submodule list.
    """
    model = world(tiny_spec("arch_vpp2"))
    payload = legacy_payload(model)
    mot_tree(payload)["mixtures.vl.blocks.0.self_attn.q.weight"] = \
        torch.zeros(1)
    with pytest.raises(KeyError, match="mixtures.vl"):
        vpp2_to_arch(payload)


def test_unknown_submodule_under_a_known_stream_surfaces_at_load_time():
    """Following on: an unknown submodule of a known stream is caught by strict.

    The remap passes it through under the new prefix, and ``strict=True``
    rejects it. This case is the evidence for "one step apart in timing": it
    pins down that the error is caught, not where it is caught.
    """
    model = world(tiny_spec("arch_vpp2"))
    payload = legacy_payload(model)
    mot_tree(payload)["mixtures.video.something_new.weight"] = torch.zeros(1)
    remapped = vpp2_to_arch(payload)
    assert "blocks.video.something_new.weight" in remapped
    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict(remapped, strict=True)


def test_two_sources_mapping_to_one_target_raises():
    """Two source keys landing on one target raise instead of overwriting.

    After an overwrite the strict load is clean, because the key set is
    unchanged, while the values come from the wrong tensor.
    """
    rules = ((r"^a\.", "z."), (r"^b\.", "z."))
    with pytest.raises(ValueError, match="remap collision"):
        remap_state_dict({"a.w": torch.zeros(1), "b.w": torch.zeros(1)},
                         rules)


def test_stale_drop_prefix_raises():
    """A ``drop`` prefix matching no key in this ckpt raises.

    A stale drop is harmless on its own, but at the next ckpt change it
    silently discards real weights, and the symptom then is a randomly
    initialised subnetwork rather than an error.
    """
    with pytest.raises(ValueError, match="matched no key in this ckpt"):
        remap_state_dict({"mixtures.video.x": torch.zeros(1)},
                         (("^mixtures\\.video\\.", "blocks.video."),),
                         drop=("mixtures.action.video_kv_projections.",))


def test_declared_video_token_prefixes_are_dropped():
    """The ``use_video_tokens`` weights are dropped only when declared.

    Dropping them keeps the key set equal to ``state_dict()``. Not dropping
    them by default is deliberate: that path is hardwired off in VPP2
    (``vpp2.py:96-102``), so a normal ckpt has none of these keys, and for a
    ckpt that did enable it, passing the prefixes explicitly also serves as a
    note that this file carries KVRoute weights.
    """
    model = world(tiny_spec("arch_vpp2"))
    payload = legacy_payload(model)
    for index, prefix in enumerate(VPP2_VIDEO_TOKEN_PREFIXES):
        mot_tree(payload)[f"{prefix}{index}.weight"] = torch.zeros(1)
    assert set(vpp2_to_arch(payload, drop=VPP2_VIDEO_TOKEN_PREFIXES)) \
        == set(model.state_dict())
