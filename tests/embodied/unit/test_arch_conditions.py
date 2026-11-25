# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Construction-time checks on ``ConditionSpec``, plus the condition encoder
(``TokenEncoder``).

The two live in one file because they are neighbouring links in one chain: how
the three fields ``APPEND_TO_SLOT`` / ``source_width`` / ``append_to`` are
written in the declaration decides the two widths of the ``Linear`` that
``linear_token_encoder`` builds. Get one link wrong and the other cannot be
built, so this file pins down which combinations are rejected at construction
time.

The V7-level reconciliation (does the host slot exist, are the widths equal)
lives in ``test_arch_providers.py``: those checks need a whole
``WorldModelSpec``, while everything here needs a single declaration.

The last section covers the slot-level key padding mask, which rides on
``Condition`` rather than on a channel of its own. It is tested here because
the mask has to be concatenated in exactly the same order as the value it
belongs to, and that order is the ``APPEND_TO_SLOT`` order this file already
pins down.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from loongforge.embodied.model.arch.spec.core import Spec, build, get_provider
from loongforge.embodied.model.arch.streams.blocks.wan_dit import (
    SlottedCrossAttention)
from loongforge.embodied.model.arch.streams.conditions import (
    Condition, ConditionSpec, InjectionKind)
from loongforge.embodied.model.arch.streams.encoders import (
    TokenEncoder, linear_token_encoder)

import loongforge.embodied.model.arch.spec.providers  # noqa: F401  registers on import

from test_arch_blocks import (BATCH, CONTEXT_LEN, build_streams, fake_bus,
                              fake_inputs, tiny_spec)

SOURCE_WIDTH = 8
SLOT_WIDTH = 4096


def appendee(**extra) -> ConditionSpec:
    """A valid ``APPEND_TO_SLOT`` declaration, modelled on vpp2's proprio."""
    kwargs = dict(name="proprio", kind=InjectionKind.APPEND_TO_SLOT,
                  targets=("video", "action"), width=SLOT_WIDTH,
                  append_to="text", source_width=SOURCE_WIDTH,
                  encoder=Spec(linear_token_encoder), source_key="proprio")
    kwargs.update(extra)
    return ConditionSpec(**kwargs)


# ---- declarations: four ways of writing it wrong ----


def test_append_to_and_its_kind_must_come_together():
    """``append_to`` and ``kind=APPEND_TO_SLOT`` are checked in both directions.

    With only the kind set, ``world.appended_to`` always comes back empty (the
    condition is appended to nothing); with only ``append_to`` set it is treated
    as an independent KV slot. Neither form raises on its own.
    """
    with pytest.raises(ValueError, match="must be set together or not at all"):
        appendee(append_to=None)
    with pytest.raises(ValueError, match="must be set together or not at all"):
        ConditionSpec(name="text", kind=InjectionKind.CROSS_ATTENTION,
                      targets=("video",), width=SLOT_WIDTH,
                      append_to="clip_image", source_key="context")


def test_appending_to_itself_is_rejected():
    """``append_to == name`` would make ``appended_to`` list the condition in
    its own append table."""
    with pytest.raises(ValueError, match="points at itself"):
        appendee(append_to="proprio")


def test_appended_condition_without_width_is_rejected():
    """An ``APPEND_TO_SLOT`` condition needs ``width``: V7 has nothing to
    compare against without it, and the encoder has no output width."""
    with pytest.raises(ValueError, match="kind=APPEND_TO_SLOT requires width"):
        appendee(width=None)


def test_source_width_without_an_encoder_is_rejected():
    """With no encoder the source tensor *is* the value, so ``width`` is the
    only width there is.

    Keeping both numbers lets one be edited without the other, and which one is
    authoritative can only be answered by reading the consumer.
    """
    with pytest.raises(ValueError, match="is only meaningful with an encoder"):
        ConditionSpec(name="text", kind=InjectionKind.CROSS_ATTENTION,
                      targets=("video",), width=SLOT_WIDTH,
                      source_width=SOURCE_WIDTH, source_key="context")


# ---- the encoder: one source for each of its two widths ----


def test_encoder_widths_come_from_the_declaration_and_the_host_slot():
    """Input width is ``source_width`` (configurable); output width is the host
    slot's ``width`` (not configurable).

    Configuring the output width again would give it two sources, and a
    disagreement shows up only as a ``cat`` shape error inside
    ``_embed_context`` during the forward pass, far from the declaration.
    """
    spec = get_provider("vpp2")()
    cond = spec.condition("proprio")
    encoder = build(cond.encoder, spec=cond, geometry=spec.geometry,
                    world=spec)
    assert isinstance(encoder, TokenEncoder)
    assert encoder.in_features == cond.source_width
    assert encoder.out_features == spec.condition("text").width
    # The existing weights are keyed exactly {weight, bias}, so TokenEncoder
    # subclasses Linear instead of wrapping one
    assert set(encoder.state_dict()) == {"weight", "bias"}


def test_encoder_for_a_non_appended_condition_raises():
    """For other injection kinds the output width has no source yet, so the
    builder raises instead of defaulting to ``source_width``."""
    cond = ConditionSpec(name="proprio", kind=InjectionKind.CONCAT_TOKENS,
                         targets=("video",), width=SLOT_WIDTH,
                         source_width=SOURCE_WIDTH, source_key="proprio",
                         encoder=Spec(linear_token_encoder))
    with pytest.raises(NotImplementedError, match="no other injection kind has a defined output width source"):
        linear_token_encoder(spec=cond, geometry=None,
                             world=get_provider("vpp2")())


def test_encoder_without_source_width_raises():
    """The input width cannot be inferred from anywhere else: the width of the
    batch tensor is not visible at build time."""
    spec = get_provider("vpp2")()
    with pytest.raises(ValueError, match="must give source_width"):
        linear_token_encoder(spec=appendee(source_width=None),
                             geometry=spec.geometry, world=spec)


def test_encoder_whose_host_slot_has_no_width_raises():
    """A host slot without a ``width`` leaves the output width with no source.

    This path is **unreachable with a real spec**: V7 requires the appended
    condition and its host to have equal widths, and ``APPEND_TO_SLOT`` must
    itself carry a width, which together force the host to have one too. Hence
    the stand-in object implementing only ``condition``: what this pins down is
    that the branch raises rather than passing ``None`` into ``Linear``, whose
    symptom would be a bare ``TypeError``.
    """
    host = ConditionSpec(name="text", kind=InjectionKind.CROSS_ATTENTION,
                         targets=("video", "action"), source_key="context")
    assert host.width is None
    world = SimpleNamespace(condition=lambda name: host)
    with pytest.raises(ValueError, match="the encoder's output width has nowhere to come from"):
        linear_token_encoder(spec=appendee(), geometry=None, world=world)


# ---- TokenEncoder: growing the token axis, and two shape checks ----


def test_two_dim_input_grows_a_token_axis():
    """A ``[B, D]`` input becomes ``[B, 1, D]``, matching the existing
    ``unsqueeze(1)``.

    The existing call site slices down to two dimensions (``proprio[:, 0, :]``
    at ``vpp2.py:845``), but what sits on the bus has to be token-shaped, so
    growing that axis belongs to the encoder, not the call site.
    """
    encoder = TokenEncoder(SOURCE_WIDTH, 5)
    flat = torch.randn(2, SOURCE_WIDTH)
    assert tuple(encoder(flat).shape) == (2, 1, 5)
    assert torch.equal(encoder(flat), encoder(flat.unsqueeze(1)))


def test_rank_and_width_are_both_checked():
    """Rank other than 2 or 3 and a wrong last-dim width are both rejected.

    The axis inserted is **axis 1**, not the last one; inserting the last axis
    would still pass through ``Linear`` whenever D == 1.
    """
    encoder = TokenEncoder(SOURCE_WIDTH, 5)
    with pytest.raises(ValueError, match="TokenEncoder takes"):
        encoder(torch.randn(2, 3, 4, SOURCE_WIDTH))
    with pytest.raises(ValueError, match="TokenEncoder input width is"):
        encoder(torch.randn(2, SOURCE_WIDTH + 1))


# ---- the slot-level key padding mask ----

CROSS = dict(name="text", kind=InjectionKind.CROSS_ATTENTION,
             targets=frozenset({"video"}))


def test_key_valid_is_checked_against_the_value_it_rides_on():
    """Both a float mask and a wrong length are rejected at construction.

    A float mask is read as an additive bias by one attention backend and as a
    keep-flag by another, and a shorter mask would broadcast instead of
    raising. Both symptoms are silent, which is why ``key_valid`` sits on the
    condition: the value it must agree with is right there.
    """
    value = torch.randn(2, 5, 4)
    with pytest.raises(ValueError, match="must be bool"):
        Condition(**CROSS, value=value, key_valid=torch.ones(2, 5))
    with pytest.raises(ValueError, match=r"expected \[B, Lc\]"):
        Condition(**CROSS, value=value,
                  key_valid=torch.ones(2, 4, dtype=torch.bool))
    Condition(**CROSS, value=value,
              key_valid=torch.ones(2, 5, dtype=torch.bool))


def cross_attention() -> SlottedCrossAttention:
    torch.manual_seed(0)
    return SlottedCrossAttention(residual_width=8, num_heads=2, head_dim=4,
                                 slots=("text",)).eval()


def test_masking_the_padded_tail_equals_passing_a_shorter_context():
    """The invariant that makes the mask worth having.

    Without it the existing VPP2 behaviour (``ctx_mask[..., 257:]`` on the text
    branch, ``wan_video_dit.py:194,202``) has no counterpart here, and padded
    text tokens contribute real keys.
    """
    attn = cross_attention()
    x = torch.randn(BATCH, 3, 8)
    real = torch.randn(BATCH, 4, 8)
    padded = torch.cat([real, torch.randn(BATCH, 2, 8)], dim=1)
    mask = torch.zeros(BATCH, 6, dtype=torch.bool)
    mask[:, :4] = True
    with torch.no_grad():
        want = attn(x, {"text": real})
        got = attn(x, {"text": padded}, key_valid={"text": mask})
    assert torch.allclose(got, want, atol=1e-6)


def test_a_slot_left_out_of_key_valid_stays_maskless():
    """Omitting a slot is not the same code path as an all-True mask.

    The maskless call is the one the existing code makes, so keeping it
    maskless is what makes the no-padding case bit-identical rather than
    merely close.
    """
    attn = cross_attention()
    x = torch.randn(BATCH, 3, 8)
    ctx = {"text": torch.randn(BATCH, 4, 8)}
    with torch.no_grad():
        assert torch.equal(attn(x, ctx, key_valid={}), attn(x, ctx))


def test_a_mask_for_an_undeclared_slot_and_a_wrong_length_both_raise():
    attn = cross_attention()
    x = torch.randn(BATCH, 3, 8)
    ctx = {"text": torch.randn(BATCH, 4, 8)}
    good = torch.ones(BATCH, 4, dtype=torch.bool)
    with pytest.raises(KeyError, match="does not declare"):
        attn(x, ctx, key_valid={"clip_image": good})
    with pytest.raises(ValueError, match="expected the context's"):
        attn(x, ctx, key_valid={"text": torch.ones(BATCH, 3,
                                                   dtype=torch.bool)})


def prepared_video_state(bus):
    """Run vpp2's video stream entry once on the given bus."""
    spec = tiny_spec("vpp2")
    module = build_streams(spec)["video"]
    return module.prepare(fake_inputs(spec)["video"],
                          timestep=torch.rand(BATCH), conditions=bus)


def test_the_appended_mask_follows_the_value_concatenation_order():
    """proprio is appended to the tail of the text slot, so its mask lands at
    the tail too, and the piece without a mask counts as all-valid.

    The two concatenations run off the same ``appended_to`` order, which is why
    they cannot drift apart.
    """
    spec = tiny_spec("vpp2")
    bus = fake_bus(spec)
    mask = torch.ones(BATCH, CONTEXT_LEN, dtype=torch.bool)
    mask[:, -1] = False
    bus.conditions["text"] = replace(bus.conditions["text"], key_valid=mask)
    extras = prepared_video_state(bus).extras
    assert set(extras["cross_kv_mask"]) == {"text"}
    got = extras["cross_kv_mask"]["text"]
    assert tuple(got.shape) == (BATCH, CONTEXT_LEN + 1)
    assert tuple(extras["cross_kv"]["text"].shape[:2]) == tuple(got.shape)
    assert torch.equal(got[:, :CONTEXT_LEN], mask)
    # proprio declares no mask, so its one token is valid
    assert got[:, -1].all()


def test_a_bus_with_no_masks_puts_no_mask_key_in_extras():
    """Absence rather than a dict of all-True masks, for the same reason the
    per-slot omission stays maskless."""
    assert "cross_kv_mask" not in prepared_video_state(
        fake_bus(tiny_spec("vpp2"))).extras
