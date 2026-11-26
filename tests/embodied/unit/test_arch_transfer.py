# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Key remapping from a legacy ckpt to the five-layer arch, and the bar.

The acceptance bar is four steps: remap, load with ``strict=True`` and report
zero missing and zero unexpected keys, run one training forward producing a
0-dim loss, and after backward find a gradient on every parameter. Each step
fails with a different symptom, which is why they are four test cases and not
one.

All three ``dual_stream_wan`` members are covered, VPP2 in full and the other
two in the last section. The bar is reached in full by VPP2 and fastwam;
InternW0-Delta stops after the load, for reasons its own section records.

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

from loongforge.embodied.model.arch.factory import (
    FUTURE_DELTA_QUERY_PREFIX, INTERNW0_DELTA_REJECTED, PROPRIO_PAYLOAD_KEY,
    VPP2_VIDEO_TOKEN_PREFIXES, fastwam_to_arch,
    internw0_delta_to_arch, remap_state_dict, vpp2_to_arch)

from arch_parity import drop_heads
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

# InternW0-Delta's extra head, whose legacy home is a ``Sequential`` inside the
# video expert: index 0 is the LayerNorm the arch calls ``norm`` and index 1 the
# Linear it calls ``head``. Prepended rather than appended, for the same reason
# the forward table prepends it: ``^blocks\.`` would not match these keys, but
# keeping the "special cases first" shape makes the two tables read alike.
FUTURE_DELTA_LEGACY_RULES: tuple[tuple[str, str], ...] = (
    (r"^heads\.future_delta\.norm\.", "mixtures.video.future_delta_head.0."),
    (r"^heads\.future_delta\.head\.", "mixtures.video.future_delta_head.1."),
)


def _to_legacy(key: str,
               rules: tuple[tuple[str, str], ...] = LEGACY_RULES) -> str:
    """Map one new-arch key to its legacy ckpt name; raise if no rule matches."""
    for pattern, replacement in rules:
        new_key, count = re.subn(pattern, replacement, key, count=1)
        if count:
            return new_key
    raise AssertionError(f"no inverse rule covers {key!r}")


def legacy_payload(model, rules: tuple[tuple[str, str], ...] = LEGACY_RULES
                   ) -> dict[str, object]:
    """Build a legacy-shaped **whole ckpt**: ``{"mot": ..., "proprio_encoder": ...}``.

    Two levels, not one, because the legacy file has two
    (``vpp2.py:1799-1801``): the ``mot`` subtree holds the two experts, and the
    proprio encoder sits beside it because it hangs off the top-level model
    itself and belongs to neither expert.

    The values deliberately differ from the model's current parameters;
    otherwise "loaded successfully" and "loaded nothing" would be numerically
    indistinguishable.

    Args:
        model: The assembled ``WorldModel`` whose key set is inverted.
        rules: Inverse rule table; the extra member of the family passes a
            table of its own.
    """
    flat = {_to_legacy(key, rules): torch.randn_like(value)
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
    model = world(tiny_spec("vpp2"))
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
    model = world(tiny_spec("vpp2"))
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
    model = world(tiny_spec("vpp2"))
    report = model.load_state_dict(vpp2_to_arch(legacy_payload(model)),
                                   strict=True)
    assert not report.missing_keys
    assert not report.unexpected_keys


def test_loaded_values_land_on_the_parameters():
    """Matching keys is not the same as transferred values, so compare tensors.

    The usual cause of right shapes with unchanged contents is feeding the
    remapped dict to a different model instance.
    """
    model = world(tiny_spec("vpp2"))
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
    spec = tiny_spec("vpp2")
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
    model = world(tiny_spec("vpp2"))
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
    model = world(tiny_spec("vpp2"))
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
    model = world(tiny_spec("vpp2"))
    payload = legacy_payload(model)
    for index, prefix in enumerate(VPP2_VIDEO_TOKEN_PREFIXES):
        mot_tree(payload)[f"{prefix}{index}.weight"] = torch.zeros(1)
    assert set(vpp2_to_arch(payload, drop=VPP2_VIDEO_TOKEN_PREFIXES)) \
        == set(model.state_dict())


# ---- The other two family members ----


def test_fastwam_remapped_keys_equal_the_model_state_dict_exactly():
    """fastwam reaches set equality with **no rules of its own**.

    This is the test that earns ``DUAL_STREAM_WAN_RENAME_RULES`` the right to
    be shared: fastwam's video expert has no CLIP branch
    (``wan/dit.py:567-568`` asserts both flags off), so the four VPP2 rules
    that handle ``img_emb`` / ``k_img`` never fire, and everything else lands
    through the family table. A new rule would have shown up here as a key the
    inverse table cannot produce.
    """
    model = world(tiny_spec("fastwam"))
    remapped = fastwam_to_arch(legacy_payload(model))
    assert set(remapped) == set(model.state_dict())


def test_fastwam_loads_strictly_and_trains_one_step():
    """The whole bar for fastwam, in one case because the model is buildable.

    Kept as one test rather than the four VPP2 has: the four exist to separate
    symptoms while the mapping is being written, and this table is the same
    one, already separated above. What is new here is the model, so what this
    pins down is "the per-token modulation path also survives a real load" --
    fastwam is the first model to reach the loss through a ``[B, S, 6, D]``
    modulation.
    """
    spec = tiny_spec("fastwam")
    model = world(spec)
    remapped = fastwam_to_arch(legacy_payload(model))
    report = model.load_state_dict(remapped, strict=True)
    assert not report.missing_keys
    assert not report.unexpected_keys
    for key, value in remapped.items():
        assert torch.equal(model.state_dict()[key], value), key
    loss, _ = model(fake_batch(spec))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    missing = [name for name, param in model.named_parameters()
               if param.grad is None]
    assert not missing, missing


INTERNW0_DELTA_LEGACY_RULES = FUTURE_DELTA_LEGACY_RULES + LEGACY_RULES


def loadable_internw0_delta():
    """``tiny_spec("internw0_delta")`` minus the head that cannot be built.

    ``track_align`` is a decoder with learned queries, so its impl
    ``track_query_head`` raises ``NotImplementedError``
    (``providers/internw0_delta/spec.py``) and the whole model cannot be
    assembled -- which is why this model is absent from ``BUILDABLE`` in
    ``test_arch_world_model``. The loss term that consumes it goes with it,
    because V6 requires every ``LossSpec.head`` to name a head that exists.

    The remaining four heads do build, and that is enough for the part of the
    bar this model can reach: remap, strict load, values. The **forward** is
    still out of reach, and not for a loading reason -- ``future_delta``
    predicts the 96 delta tokens, which have no declared position in
    ``GeometrySpec``, so its output is compared against the whole video
    tensor and the shapes disagree. The same gap the head's own comment
    records; it closes when Lane A adds the condition-frame and delta fields.
    """
    spec = tiny_spec("internw0_delta")
    return drop_heads(spec, ("track_align",))


def internw0_delta_payload(model) -> dict[str, object]:
    """A legacy payload including the future-delta queries a real ckpt has.

    They are injected rather than derived: the new architecture has no
    parameter for them, so the inverse table cannot produce them, and a
    fixture without them would never exercise the ``drop`` declaration that
    every real ckpt needs.
    """
    payload = legacy_payload(model, INTERNW0_DELTA_LEGACY_RULES)
    mot_tree(payload)[FUTURE_DELTA_QUERY_PREFIX] = torch.zeros(1, 4, 32)
    return payload


def test_internw0_delta_remapped_keys_equal_the_model_state_dict_exactly():
    """Set equality for the four buildable heads, delta queries declared away.

    The one key class this model adds over fastwam is the future-delta head,
    and it is the second **cross-layer move** in the table: legacy grows it
    inside the video expert as a ``Sequential``, the arch files it under
    ``heads.future_delta``. If that pair of rules were missing, its two
    submodules would land on ``blocks.video.future_delta_head.*`` and show up
    here as two missing and two unexpected keys at once.
    """
    model = world(loadable_internw0_delta())
    remapped = internw0_delta_to_arch(internw0_delta_payload(model),
                                      drop=(FUTURE_DELTA_QUERY_PREFIX,))
    assert set(remapped) == set(model.state_dict())


def test_internw0_delta_loads_strictly_and_the_values_land():
    """Strict load with both lists empty, and every tensor transferred.

    Stops one step short of the full bar on purpose; ``loadable_internw0_delta``
    records why the forward is unreachable.
    """
    model = world(loadable_internw0_delta())
    remapped = internw0_delta_to_arch(internw0_delta_payload(model),
                                      drop=(FUTURE_DELTA_QUERY_PREFIX,))
    report = model.load_state_dict(remapped, strict=True)
    assert not report.missing_keys
    assert not report.unexpected_keys
    for key, value in remapped.items():
        assert torch.equal(model.state_dict()[key], value), key


def test_internw0_delta_future_delta_queries_must_be_declared():
    """Not declaring the delta queries raises instead of silently passing them.

    Without the ``drop`` the catch-all rule would happily rename them to
    ``blocks.video.future_delta_queries``, which no module owns, and the
    symptom would be one unexpected key at load time with nothing to say why.
    That is what the explicit check buys: the discard has to be a decision the
    caller writes down.
    """
    model = world(loadable_internw0_delta())
    with pytest.raises(ValueError, match="future_delta_queries"):
        internw0_delta_to_arch(internw0_delta_payload(model))


@pytest.mark.parametrize("sibling", sorted(INTERNW0_DELTA_REJECTED))
def test_internw0_delta_refuses_the_siblings_with_no_module(sibling):
    """Each refused sibling is named, with its reason, before any remapping.

    All three are real top-level payload keys under their own config gates, and
    for each one the new architecture has no module at all -- not a renaming
    gap. Refusing here rather than at ``strict=True`` is the difference between
    a sentence saying which component is missing and a list of tensor names.

    An **empty** sibling counts as absent, which is the normal shape of
    ``understanding`` when the tower is off: it is written as ``{}`` rather
    than omitted.
    """
    model = world(loadable_internw0_delta())
    payload = internw0_delta_payload(model)
    payload[sibling] = {}
    assert set(internw0_delta_to_arch(payload,
                                      drop=(FUTURE_DELTA_QUERY_PREFIX,)))
    payload[sibling] = {"weight": torch.zeros(1)}
    with pytest.raises(ValueError, match=sibling):
        internw0_delta_to_arch(payload, drop=(FUTURE_DELTA_QUERY_PREFIX,))


def test_proprio_type_embedding_is_folded_into_the_bias():
    """InternW0-Delta's third proprio parameter folds exactly into the bias.

    ``ProprioContextEncoder.forward`` is ``Linear(x) + type_embedding``
    (``proprio_encoder.py:16-17``) with nothing non-linear between the two
    additions, so the two biases are one. Folding keeps the arch's encoder a
    plain ``nn.Linear``; the alternative would be a parameter whose only job
    is to be added to another parameter. The fold must be exact, which is what
    this checks -- the earlier set-equality tests would pass just as well if
    ``type_embedding`` were silently dropped.
    """
    model = world(loadable_internw0_delta())
    payload = internw0_delta_payload(model)
    encoder = payload[PROPRIO_PAYLOAD_KEY]
    encoder["type_embedding"] = torch.randn_like(encoder["bias"])
    remapped = internw0_delta_to_arch(payload,
                                      drop=(FUTURE_DELTA_QUERY_PREFIX,))
    assert "condition_encoders.proprio.type_embedding" not in remapped
    assert torch.equal(remapped["condition_encoders.proprio.bias"],
                       encoder["bias"] + encoder["type_embedding"])
