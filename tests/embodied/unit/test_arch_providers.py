# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end consistency of the three providers: geometry, layout, visibility.

This file checks only that **declarations reconcile with each other**, never
numerics; numeric equivalence belongs to the weight-transfer step. Every
assertion asks "do these two places say the same thing": the token count the
geometry computes against the layout's segment lengths, the capture points the
head table declares against ``captures()``, the backend the provider declares
against what capability negotiation concludes.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from loongforge.embodied.model.arch.builder import (
    join_stream_visibility, query_in_stream)
from loongforge.embodied.model.arch.layout import TokenLayout
from loongforge.embodied.model.arch.spec.core import (
    Spec, get_provider, provider_names)
from loongforge.embodied.model.arch.streams.conditions import (
    ConditionSpec, InjectionKind)
from loongforge.embodied.model.arch.streams.spec import StreamSpec
from loongforge.embodied.model.arch.visibility.mask import (
    AttentionPlan, Capability, CompiledMask, Scenario)
from loongforge.embodied.model.arch.visibility.rules import same_stream

import loongforge.embodied.model.arch.spec.providers  # noqa: F401  registers on import

MODEL_TYPES = ("arch_vpp2", "fastwam", "internw0_delta")


@pytest.fixture(params=MODEL_TYPES)
def model_spec(request):
    """The production spec of one of the three registered models."""
    return get_provider(request.param)()


def test_all_three_are_registered():
    """The provider registry is keyed by ``model_type`` and holds only models.

    The six topology families are plain factory functions that are not
    registered, so neither YAML nor a launch argument can name one.
    """
    for model_type in MODEL_TYPES:
        assert model_type in provider_names()
    # The six topology families are unregistered factory functions, invisible
    # to both YAML and the launch arguments
    assert "dual_stream_wan" not in provider_names()


def test_provider_rejects_non_none_config(model_spec):
    """A provider raises on a non-None config instead of ignoring it silently.

    YAML override of the spec is not implemented yet, so accepting and dropping
    a config dict would make a user's override disappear without a trace.
    """
    provider = get_provider(model_spec.model_type)
    with pytest.raises(NotImplementedError):
        provider({"geometry": {"num_frames": 9}})


def test_unknown_stream_scalar_is_rejected(model_spec):
    """An unrecognised keyword scalar fails rather than being swallowed.

    Providers take a small set of named scalars; a misspelt one must not look
    like a successful override.
    """
    provider = get_provider(model_spec.model_type)
    with pytest.raises(ValueError):
        provider(video_layer_count=12)


def test_layout_segment_lengths_come_from_geometry(model_spec):
    """Segment lengths and labels are exactly what the geometry computes.

    The video segment is ``padded_seq_len`` tokens and the action segment is
    ``action_horizon``; the frame label runs in groups of ``tokens_per_frame``
    and action tokens carry ``chunk == frame == step // ratio``.
    """
    geo = model_spec.geometry
    layout = TokenLayout.from_geometry(geo, model_spec.streams)
    assert layout.stream_names == model_spec.stream_names
    assert layout.range_of("video").length == geo.padded_seq_len
    assert layout.range_of("action").length == geo.action_horizon
    assert layout.total == geo.padded_seq_len + geo.action_horizon
    # The video segment's frame label runs in groups of tokens_per_frame
    frame = layout.labels_of("frame")[:geo.padded_seq_len]
    assert frame[:geo.tokens_per_frame].tolist() == [0] * geo.tokens_per_frame
    assert int(frame.max()) == geo.patch_grid[0] - 1
    # In the action segment, chunk == frame == step // ratio
    action = slice(geo.padded_seq_len, layout.total)
    assert torch.equal(layout.labels_of("chunk")[action],
                       layout.labels_of("frame")[action])


def test_joint_group_is_one_to_one_over_both_streams(model_spec):
    """All three models are one joint group of two equally deep streams.

    Unequal depths would be reconciled by ``LayerMapping``, but here layer i of
    the video stream pairs with layer i of the action stream. V3 constrains
    ``head_dim`` only: the two streams having different ``residual_width`` is
    the normal case, which is why width and attention width are separate.
    """
    assert len(model_spec.groups) == 1
    group = model_spec.groups[0]
    assert group.stream_names == ("video", "action")
    # Unequal depths are reconciled by LayerMapping, but these three models are
    # equally deep and ONE_TO_ONE
    assert group.layer_index_maps() == {
        name: {i: i for i in range(group.num_layers)}
        for name in group.stream_names}
    # V3 constrains head_dim only; unequal residual_width across streams is the
    # normal case
    assert group.head_dim == model_spec.stream("video").head_dim
    widths = {s.name: s.residual_width for s in group.members}
    assert widths["video"] != widths["action"], widths
    assert group.routes == ()


@pytest.fixture
def layout(model_spec):
    """The concatenated token layout for the spec under test."""
    return TokenLayout.from_geometry(model_spec.geometry, model_spec.streams)


def test_visibility_keys_are_stream_names(model_spec):
    """Visibility is keyed by stream name, so every key must be a real stream."""
    assert set(model_spec.visibility) <= set(model_spec.stream_names)


def test_each_declared_rule_actually_hides_something(model_spec, layout):
    """A declared rule must mask something; an all-True one is a lost config.

    A rule that evaluates to all True is indistinguishable from not declaring
    it, which is the typical symptom of a dropped configuration value.
    """
    for name, vis in model_spec.visibility.items():
        out = vis.evaluate(layout)
        assert out.shape == (layout.total, layout.total), name
        assert out.dtype == torch.bool, name
        assert not bool(out.all()), (name, vis.expr)


def test_cross_stream_edge_is_one_way(model_spec, layout):
    """Action tokens read video tokens; video tokens never read action tokens.

    All three models express this as an in-edge with no out-edge. If the video
    rows opened onto action columns, the video trunk would see the actions it
    is supposed to predict from.
    """
    geo = model_spec.geometry
    video = slice(0, geo.padded_seq_len)
    action = slice(geo.padded_seq_len, layout.total)
    on_action = model_spec.visibility["action"].evaluate(layout)
    assert bool(on_action[action, video].any())
    # Not fully open, otherwise the prefix_only / frame_causal term was written
    # for nothing
    assert not bool(on_action[action, video].all())
    # No video query row may open an action column
    on_video = model_spec.visibility["video"].evaluate(layout)
    assert not bool(on_video[video, action].any()), \
        model_spec.visibility["video"].expr


def test_block_size_is_given_exactly_when_backend_is_flex(model_spec):
    """V9: ``attention_block_size`` is set if and only if the backend is flex.

    dense and sdpa cannot use a block size, so carrying one there would be a
    value nobody reads, and a flex backend without one cannot build a
    BlockMask.
    """
    assert (model_spec.attention_block_size is not None) == \
        (model_spec.attention_backend == "flex")


def test_declared_backend_negotiates_exact(model_spec):
    """The backend a provider declares can express the visibility it declares.

    A failure here means the spec contradicts itself: the backend line and the
    rule lines were written by two different people.
    """
    scenario = Scenario(phase="train",
                        backend=model_spec.attention_backend,
                        block_size=model_spec.attention_block_size)
    for name, vis in model_spec.visibility.items():
        assert CompiledMask.supports(vis, scenario) is Capability.EXACT, \
            (name, vis.expr)


def test_plan_materializes_on_the_declared_backend(model_spec, layout):
    """The action-stream plan materialises to ``[1, 1, L, L]`` on both backends.

    dense is the reference form; the provider's own backend must produce the
    same shape, so the engine can swap backends without reshaping anything.
    """
    plan = AttentionPlan.from_visibility(model_spec.visibility["action"],
                                         layout)
    shape = (1, 1, layout.total, layout.total)
    assert plan.materialize("dense").shape == shape
    got = plan.materialize(model_spec.attention_backend,
                           block_size=model_spec.attention_block_size)
    assert tuple(got.shape) == shape


def test_every_head_sources_a_real_stream_and_layer(model_spec):
    """Each head reads an existing stream and, if pinned, an in-range layer.

    An out-of-range layer index would only fail when the engine tried to
    capture it, deep inside a forward pass.
    """
    for head in model_spec.heads:
        assert head.source.stream in model_spec.stream_names, head.name
        if head.source.layer is not None:
            depth = model_spec.stream(head.source.stream).num_layers
            assert 0 <= head.source.layer < depth, head.name


def test_captures_are_derived_from_the_head_table(model_spec):
    """``captures()`` is derived from the head table, not declared a second time.

    It must be exactly the heads whose ``source.layer`` is not None. Two
    independent declarations fail either way: the engine captures a layer
    nobody reads (wasted memory), or a head asks for a layer nobody captured
    (``KeyError`` only at forward time).
    """
    want: dict[str, list[int]] = {}
    for head in model_spec.heads:
        if head.source.layer is not None:
            want.setdefault(head.source.stream, []).append(head.source.layer)
    assert model_spec.captures() == {
        stream: tuple(sorted(set(layers))) for stream, layers in want.items()}
    # Heads that consume the final output (layer=None) stay out of the capture
    # table; all three models have at least two such heads
    assert sum(h.source.layer is None for h in model_spec.heads) >= 2


def test_each_loss_names_an_existing_head(model_spec):
    """Every loss term names a head that exists, so losses cannot dangle."""
    heads = {head.name for head in model_spec.heads}
    for loss in model_spec.diffusion.losses:
        assert loss.name in heads, (loss.name, sorted(heads))


def test_joint_mask_gives_each_stream_its_own_rows(model_spec, layout):
    """In the joined mask, each stream's query rows equal its own rule's rows.

    That is the whole semantics of ``join_stream_visibility``. An
    implementation using ``&`` is caught here: the video rows would also be
    masked by the action rule, with the shape still exactly right.
    """
    joint = join_stream_visibility(model_spec.visibility).evaluate(layout)
    for name, vis in model_spec.visibility.items():
        rows = layout.range_of(name)
        own = vis.evaluate(layout)[rows.start:rows.stop]
        assert torch.equal(joint[rows.start:rows.stop], own), name


def test_joint_mask_leaves_undeclared_streams_fully_visible(layout):
    """Declaring one stream leaves the other's rows fully visible, not masked.

    A stream with no rule is unconstrained, and an empty visibility dict maps
    to None, the "no structural mask at all" case.
    """
    only_action = {"action": same_stream()}
    joint = join_stream_visibility(only_action).evaluate(layout)
    video = layout.range_of("video")
    assert bool(joint[video.start:video.stop].all())
    assert join_stream_visibility({}) is None


def test_query_in_stream_rejects_unknown_name(layout):
    """``query_in_stream`` raises on a stream name the layout does not have.

    Matching nothing instead would silently turn the whole rule into a no-op.
    """
    with pytest.raises(KeyError, match="layout has no stream"):
        query_in_stream("typo").evaluate(layout)


# ---- Placing shape fields: one number may have only one source ----

# Per model: (latent_channels, sum of CHANNEL_CONCAT condition widths,
# entry channel count). The third column is today's ``in_dim``: vpp2's 36 is
# 16 + 4 (mask) + 16 (condition-frame latent).
ENTRY_CHANNELS = {
    "arch_vpp2": (16, 20, 36),
    "fastwam": (48, 0, 48),
    "internw0_delta": (48, 0, 48),
}


def test_video_head_width_reads_the_geometry(model_spec):
    """``video_flow``'s output width is ``latent_channels``, not a second source.

    Writing it twice fails quietly: swapping the VAE changes the geometry but
    not the head, strict loading still passes because nobody reconciles that
    Linear's shape, and the damage only shows up in the loss during backprop.
    """
    want, _, _ = ENTRY_CHANNELS[model_spec.model_type]
    assert model_spec.geometry.latent_channels == want
    assert model_spec.head("video_flow").output_width == want


def test_entry_channels_are_derived_from_the_condition_table(model_spec):
    """``in_dim`` is derived: latent channels plus channel-axis condition widths.

    For vpp2 that is 16 latent + 20 condition (4 mask + 16 condition-frame
    latent) = 36. The action stream concatenates nothing on the channel axis,
    so it must answer 0 rather than raise.
    """
    latent, concat, entry = ENTRY_CHANNELS[model_spec.model_type]
    assert model_spec.concat_channels("video") == concat
    assert latent + model_spec.concat_channels("video") == entry
    # The action stream has no channel-axis concatenation: 0, not an error
    assert model_spec.concat_channels("action") == 0


def test_concat_channels_rejects_unknown_stream(model_spec):
    """``concat_channels`` raises on a stream the spec does not declare."""
    with pytest.raises(KeyError, match="has no stream"):
        model_spec.concat_channels("typo")


def test_cross_attention_conditions_carry_their_kv_width(model_spec):
    """Every cross-attention condition declares ``width``, the KV input width.

    That width is the input width of the target stream's projection layer.
    Without it the consumer has to guess (today's code hardcodes 4096 inside
    the stream), and swapping the text encoder forces that 4096 to change too.
    """
    for cond in model_spec.conditions:
        if cond.kind is InjectionKind.CROSS_ATTENTION:
            assert cond.width is not None, cond.name
            assert cond.width > 0, cond.name


def test_cross_slots_and_conditions_reconcile_both_ways(model_spec):
    """V7 holds in both directions on all three providers, not just by raising.

    Each stream's ``cross_kv_slots`` is exactly the set of enabled
    cross-attention conditions targeting it. vpp2's video stream has **two**
    slots: ``I2VCrossAttention`` runs attention twice and sums the results
    (``wan_video_dit.py:187-204``), so CLIP and text are two slots rather than
    two sources concatenated into one.
    """
    filled: dict[str, set[str]] = {n: set() for n in model_spec.stream_names}
    for cond in model_spec.conditions:
        if cond.kind is InjectionKind.CROSS_ATTENTION and cond.enabled:
            for target in cond.targets:
                filled[target].add(cond.name)
    for stream in model_spec.streams:
        assert set(stream.cross_kv_slots) == filled[stream.name], stream.name
    want = {"arch_vpp2": {"clip_image", "text"}}.get(model_spec.model_type,
                                                {"text"})
    assert set(model_spec.stream("video").cross_kv_slots) == want


def test_timestep_embed_width_is_declared_exactly_when_consumed(model_spec):
    """``freq_dim`` belongs to the stream and tracks ``needs_timestep`` exactly.

    All three models have both streams consuming a timestep at an embedding
    width of 256, so the two declarations must turn on and off together.
    """
    for stream in model_spec.streams:
        assert stream.needs_timestep is True, stream.name
        assert stream.timestep_embed_width == 256, stream.name


def test_disabled_condition_still_counts_toward_entry_channels():
    """Disabling a condition changes its value, not the entry conv's shape.

    vpp2 keeps 20 concatenated channels either way. If ``enabled=False`` could
    change the shape, one YAML switch would make the same checkpoint
    unloadable, the hardest class of "weights do not match" bug to trace.
    """
    spec = get_provider("arch_vpp2")()
    before = spec.concat_channels("video")
    off = tuple(replace(c, enabled=False) if c.name == "cond_latent" else c
                for c in spec.conditions)
    after = replace(spec, conditions=off).concat_channels("video")
    assert after == before == 20


def test_channel_concat_without_width_is_rejected():
    """A CHANNEL_CONCAT condition without ``width`` fails at construction time.

    The entry convolution cannot be built without it, so the failure has to
    happen while building the spec, not while building the module.
    """
    with pytest.raises(ValueError, match="kind=CHANNEL_CONCAT requires width"):
        ConditionSpec(name="cond_latent", kind=InjectionKind.CHANNEL_CONCAT,
                      targets=("video",), source_key="condition_latents")


# ---- APPEND_TO_SLOT: conditions that ride an existing host slot ----

# The encoder's **input** width, which differs per dataset (the only reason
# source_width is configurable at all): libero is 8 for vpp2/fastwam
# (``train_action.yaml:169``), and InternW0-Delta's 4D descriptor is 80
# (``proprio_encoder.py:46-47``).
PROPRIO_SOURCE_WIDTH = {"arch_vpp2": 8, "fastwam": 8, "internw0_delta": 80}


def test_proprio_rides_the_text_slot_in_all_three(model_spec):
    """Proprio appends to the text slot instead of being a fifth KV slot.

    Declaring it as CROSS_ATTENTION would add 3 x num_layers tensors that do
    not exist in any checkpoint (one k/v/norm_k set per layer), and that only
    surfaces at strict load time. Its post-encoder width equals the host
    slot's, which is what V7 reconciles; the pre-encoder width is the
    dataset's number.
    """
    cond = model_spec.condition("proprio")
    assert cond.kind is InjectionKind.APPEND_TO_SLOT
    assert cond.append_to == "text"
    # Both streams consume it: today there is a single context shared by the
    # two streams (``vpp2.py:843``)
    assert set(cond.targets) == set(model_spec.stream_names)
    # It owns no slot name, so it appears in no stream's cross_kv_slots
    for stream in model_spec.streams:
        assert "proprio" not in stream.cross_kv_slots, stream.name
    # The post-encoder width equals the host slot's (V7 reconciles on this);
    # the pre-encoder width is the dataset's number
    assert cond.width == model_spec.condition("text").width
    assert cond.source_width == PROPRIO_SOURCE_WIDTH[model_spec.model_type]
    assert cond.encoder is not None


def test_appended_to_lists_the_condition_for_each_target_stream(model_spec):
    """``appended_to`` lists the appended conditions of one slot and stream.

    A slot with no appended condition answers with an empty tuple rather than
    raising, which is the case for most slots.
    """
    for stream in model_spec.stream_names:
        assert model_spec.appended_to("text", stream) == ("proprio",)
    # A slot with no appended condition gives an empty tuple, not an error
    assert model_spec.appended_to("clip_image", "video") == ()


def test_appended_to_rejects_unknown_stream(model_spec):
    """``appended_to`` raises on a stream the spec does not declare."""
    with pytest.raises(KeyError, match="has no stream"):
        model_spec.appended_to("text", "typo")


def test_timestep_declarations_must_agree():
    """``needs_timestep`` and ``timestep_embed_width`` must agree, both ways.

    A stream that consumes a timestep without an embedding width cannot build
    its embedder; a width on a stream that never consumes one is a parameter
    that is never called. Both are caught at construction. Turning both off is
    legal and is exactly EWAM's vl stream.
    """
    common = dict(impl=Spec(impl=object), residual_width=16, num_heads=2,
                  head_dim=8, num_layers=2)
    with pytest.raises(ValueError, match="requires timestep_embed_width"):
        StreamSpec(name="video", **common)
    with pytest.raises(ValueError, match="is never called yet still enters state_dict"):
        StreamSpec(name="vl", needs_timestep=False,
                   timestep_embed_width=256, **common)
    # Both off is legal, and is exactly EWAM's vl stream
    assert StreamSpec(name="vl", needs_timestep=False,
                      **common).timestep_embed_width is None


# ---- V7 continued: the four reconciliations of APPEND_TO_SLOT ----


def vpp2_proprio(**changes):
    """Rebuild the whole vpp2 spec with some proprio fields changed.

    ``replace`` re-runs V1-V9 (they all live in ``__post_init__``), so
    validation has already happened by the time this helper returns; each case
    below wraps it in ``pytest.raises``.
    """
    spec = get_provider("arch_vpp2")()
    return replace(spec, conditions=tuple(
        replace(cond, **changes) if cond.name == "proprio" else cond
        for cond in spec.conditions))


def test_appending_to_a_nonexistent_slot_is_rejected():
    """``append_to`` must name a condition this model actually declares."""
    with pytest.raises(ValueError, match="but this model has no such condition"):
        vpp2_proprio(append_to="typo")


def test_appending_to_a_channel_concat_condition_is_rejected():
    """Only a CROSS_ATTENTION condition has a tail to append to.

    Letting this through would produce a dimension mismatch at a ``cat`` during
    forward: the host is the 5-D ``[B, C, F, H, W]`` latent while the appended
    value is a 3-D token sequence.
    """
    with pytest.raises(ValueError, match="only a CROSS_ATTENTION slot has a tail to append to"):
        vpp2_proprio(append_to="cond_latent")


def test_width_differing_from_the_host_slot_is_rejected():
    """An appended condition must match its host slot's width.

    Both go through the **same** condition projector, so unequal widths cannot
    be concatenated at all.
    """
    with pytest.raises(ValueError, match="different widths cannot be concatenated"):
        vpp2_proprio(width=1024)


def without_text_slot(spec, *streams: str):
    """Drop ``text`` from the named streams' ``cross_kv_slots``, keep the rest.

    Both cases below need this first: V7's slot/condition reconciliation in
    both directions runs **before** the four append checks, so editing only
    the condition table trips "a stream declares a text slot that nobody
    fills" and the check under test is never reached.
    """
    group = spec.groups[0]
    members = tuple(
        replace(stream, cross_kv_slots=tuple(
            slot for slot in stream.cross_kv_slots if slot != "text"))
        if stream.name in streams else stream
        for stream in group.members)
    return replace(group, members=members)


def test_appended_condition_cannot_reach_a_stream_its_host_skips():
    """An appended condition's targets must be a subset of its host's targets.

    The out-of-range stream would receive an over-long context while its
    projector was built for the host condition's width. Since both share one
    ``width``, no shape error fires; that stream simply reads KV it was never
    meant to see.
    """
    spec = get_provider("arch_vpp2")()
    narrow = tuple(replace(cond, targets=("video",)) if cond.name == "text"
                   else cond for cond in spec.conditions)
    with pytest.raises(ValueError, match="does not inject them"):
        replace(spec, conditions=narrow,
                groups=(without_text_slot(spec, "action"),))


def test_appended_condition_must_be_off_when_its_host_is_off():
    """Disabling the host slot while the appended condition stays on raises.

    With the host gone there is nowhere for the appended value to go.
    """
    spec = get_provider("arch_vpp2")()
    off = tuple(replace(cond, enabled=False) if cond.name == "text" else cond
                for cond in spec.conditions)
    with pytest.raises(ValueError, match="the append has nowhere to go"):
        replace(spec, conditions=off,
                groups=(without_text_slot(spec, "video", "action"),))


def test_append_order_follows_the_declaration_not_the_alphabet():
    """With two conditions on one slot, the concat order is the declared order.

    Relying on alphabetical order or on dict insertion order fails invisibly:
    renaming a condition would move the KV the attention reads, while the
    shapes, the key set and the loss magnitude all stay the same.
    """
    spec = get_provider("arch_vpp2")()
    later = replace(spec.condition("proprio"), name="aaa_later",
                    encoder=None, source_width=None, source_key="aaa")
    spec = replace(spec, conditions=spec.conditions + (later,))
    assert spec.appended_to("text", "video") == ("proprio", "aaa_later")


def test_disabled_appended_condition_drops_out_of_the_order():
    """Disabling an appended condition really removes a token, unlike channels.

    A cross-attention KV length is data dependent and enters no weight shape,
    so dropping it here cannot make the same checkpoint unloadable. This is the
    opposite of ``concat_channels``, where disabling must not change the shape.
    """
    spec = get_provider("arch_vpp2")()
    off = tuple(replace(cond, enabled=False) if cond.name == "proprio"
                else cond for cond in spec.conditions)
    assert replace(spec, conditions=off).appended_to("text", "video") == ()
