# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 engine: cat -> one joint attention -> split (interface I3).

Two main assertions:
  1. **cat->attn->split is equivalent to per-stream attention** when the
     visibility is exactly block diagonal over the layout segments. This is the
     evidence that the single joint attention does not mix another stream's
     tokens in.
  2. **The call order cannot be rearranged**: within one layer every
     participating stream's ``pre_attn`` has to finish before any stream's
     ``post_attn`` runs.
"""

from __future__ import annotations

import math

import pytest
import torch

from loongforge.embodied.model.arch.layout import SegmentRange, TokenLayout
from loongforge.embodied.model.arch.spec.core import Spec
from loongforge.embodied.model.arch.streams.conditions import ConditionBus
from loongforge.embodied.model.arch.streams.engine import (
    AttentionGroup, EngineOutput, JointAttentionEngine, assert_shape_signature,
    restrict_to_streams)
from loongforge.embodied.model.arch.streams.routing import KVRoute
from loongforge.embodied.model.arch.streams.spec import LayerMapping, StreamSpec
from loongforge.embodied.model.arch.streams.state import QKV, StreamState
from loongforge.embodied.model.arch.visibility.mask import AttentionPlan
from loongforge.embodied.model.arch.visibility.rules import (
    same_stream, stream_edge)

HEAD_DIM = 8
NUM_HEADS = 2


def spec(name: str, *, num_layers: int = 3, residual_width: int = 16,
         **extra) -> StreamSpec:
    """A ``StreamSpec`` backed by ``_ToyStream``, at test widths."""
    # ``needs_timestep`` defaults to True, which makes
    # ``timestep_embed_width`` mandatory (the if-and-only-if check in
    # StreamSpec). It is a setdefault rather than a fixed value so a
    # needs_timestep=False case can override it to None at the same time.
    extra.setdefault("timestep_embed_width", 256)
    return StreamSpec(name=name, impl=Spec(impl=_ToyStream),
                      residual_width=residual_width, num_heads=NUM_HEADS,
                      head_dim=HEAD_DIM, num_layers=num_layers, **extra)


class _ToyStream:
    """Minimal ``StreamModule``: each of the four stages does one observable
    thing, and it holds no ``nn.Module``.

    It lives in the test rather than in the package because it is scaffolding,
    not a reference implementation: one more toy DiT in the package and someone
    will eventually treat it as a template.
    """

    def __init__(self, stream_spec: StreamSpec, *, trace: list, seed: int = 0):
        self.spec = stream_spec
        self.trace = trace
        gen = torch.Generator().manual_seed(seed)
        attn_width = stream_spec.num_heads * stream_spec.head_dim
        scale = attn_width ** -0.5
        self.to_qkv = [torch.randn(stream_spec.residual_width, 3 * attn_width,
                                   generator=gen) * scale
                       for _ in range(stream_spec.num_layers)]
        self.from_attn = [torch.randn(attn_width, stream_spec.residual_width,
                                      generator=gen) * scale
                          for _ in range(stream_spec.num_layers)]

    def prepare(self, x: torch.Tensor, *, timestep, conditions=None):
        """Wrap the already token-shaped input into a ``StreamState``."""
        self.trace.append((self.spec.name, "prepare", timestep is None))
        # The toy stream's input is already [B, L, W], so prepare only wraps it.
        # A real implementation patchifies here (video [B,C,F,H,W]) or projects
        # (action [B,T,D]); the "raw tensor in, StreamState out" contract is the
        # same in both cases.
        return StreamState(stream=self.spec.name, hidden=x)

    def pre_attn(self, state: StreamState, *, local_layer_id: int) -> QKV:
        """Project the hidden state into per-head q/k/v of shape [B, L, H, D]."""
        self.trace.append((self.spec.name, "pre_attn", local_layer_id))
        batch, tokens, _ = state.hidden.shape
        packed = state.hidden @ self.to_qkv[local_layer_id]
        shape = (batch, tokens, self.spec.num_heads, self.spec.head_dim)
        q, k, v = (part.reshape(shape) for part in packed.chunk(3, dim=-1))
        return QKV(q=q, k=k, v=v)

    def post_attn(self, state: StreamState, attn_out: torch.Tensor, *,
                  local_layer_id: int) -> StreamState:
        """Project the attention output back and add it to the residual."""
        self.trace.append((self.spec.name, "post_attn", local_layer_id))
        batch, tokens = attn_out.shape[:2]
        flat = attn_out.reshape(batch, tokens, -1)
        return state.replace_hidden(
            state.hidden + flat @ self.from_attn[local_layer_id])

    def finalize(self, state: StreamState) -> StreamState:
        """Scale by 2 so a skipped ``finalize`` is visible in the output."""
        self.trace.append((self.spec.name, "finalize", None))
        return state.replace_hidden(state.hidden * 2.0)


VIDEO_TOKENS = 8
ACTION_TOKENS = 4
BATCH = 2


def joint_layout(*, video: int = VIDEO_TOKENS,
                 action: int = ACTION_TOKENS) -> TokenLayout:
    """8 video tokens then 4 action tokens, with a ``stream`` label."""
    return TokenLayout(
        segments=(SegmentRange("video", 0, video),
                  SegmentRange("action", video, video + action)),
        labels={"stream_id": torch.tensor([0] * video + [1] * action)})


def two_streams(*, num_layers: int = 3, action_extra: dict | None = None):
    """Every argument the engine needs: group / layout / modules / inputs /
    trace.

    ``inputs`` holds **bare tensors** rather than ``StreamState`` objects:
    step 1 of the engine calls ``prepare``, and ``prepare`` consumes raw inputs
    (tokenization belongs to the stream).
    """
    video_spec = spec("video", num_layers=num_layers)
    action_spec = spec("action", num_layers=num_layers, residual_width=12,
                       **(action_extra or {}))
    group = AttentionGroup(streams=(video_spec, action_spec),
                           num_layers=num_layers)
    trace: list = []
    modules = {"video": _ToyStream(video_spec, trace=trace, seed=1),
               "action": _ToyStream(action_spec, trace=trace, seed=2)}
    gen = torch.Generator().manual_seed(7)
    inputs = {
        "video": torch.randn(BATCH, VIDEO_TOKENS, video_spec.residual_width,
                             generator=gen),
        "action": torch.randn(BATCH, ACTION_TOKENS, action_spec.residual_width,
                              generator=gen),
    }
    return group, joint_layout(), modules, inputs, trace


def run(engine, group, layout, modules, inputs, *, plan=None, **extra):
    """Call ``engine.run`` with an empty condition bus and no timesteps."""
    return engine.run(group, inputs, modules,
                      plan if plan is not None else AttentionPlan(layout=layout),
                      layout, conditions=ConditionBus(),
                      timesteps={name: None for name in group.stream_names},
                      **extra)


def test_call_order_is_not_reorderable():
    """Per layer: every participating stream's ``pre_attn``, then one joint
    attention, then every stream's ``post_attn`` (§7).

    With 3 layers and 2 streams the body is exactly 12 calls in that order,
    bracketed by two ``prepare`` and two ``finalize`` calls.
    """
    group, layout, modules, states, trace = two_streams()
    run(JointAttentionEngine(), group, layout, modules, states)
    assert trace[:2] == [("video", "prepare", True),
                         ("action", "prepare", True)]
    assert trace[-2:] == [("video", "finalize", None),
                          ("action", "finalize", None)]
    body = trace[2:-2]
    assert len(body) == 3 * 4
    for layer in range(3):
        window = body[layer * 4:(layer + 1) * 4]
        assert window == [("video", "pre_attn", layer),
                          ("action", "pre_attn", layer),
                          ("video", "post_attn", layer),
                          ("action", "post_attn", layer)]


def test_output_shapes_and_stream_set():
    """Each stream comes back at its own residual width (16 and 12 here), so the
    split undoes the concatenation rather than returning a shared tensor."""
    group, layout, modules, states, _ = two_streams()
    out = run(JointAttentionEngine(), group, layout, modules, states)
    assert isinstance(out, EngineOutput)
    assert set(out.states) == {"video", "action"}
    assert out.require("video").hidden.shape == (BATCH, VIDEO_TOKENS, 16)
    assert out.require("action").hidden.shape == (BATCH, ACTION_TOKENS, 12)
    assert out.captured == {}
    with pytest.raises(KeyError, match="has no state for stream"):
        out.require("vl")


def test_joint_attention_equals_per_stream_when_block_diagonal():
    """With ``same_stream()`` visibility, cat->attn->split equals running each
    stream's attention on its own.

    This is the direct evidence that the single joint attention does not mix in
    tokens from the other stream.
    """
    group, layout, modules, states, _ = two_streams()
    plan = AttentionPlan.from_visibility(same_stream(), layout)
    joint = run(JointAttentionEngine(), group, layout, modules, states,
                plan=plan)
    reference = _per_stream_reference(group, modules, states)
    for name in group.stream_names:
        got = joint.require(name).hidden
        want = reference[name].hidden
        assert torch.allclose(got, want, atol=1e-6), \
            (name, float((got - want).abs().max()))


def test_cross_stream_edge_actually_changes_the_result():
    """Adding one cross-stream edge has to change the result, otherwise the
    equivalence test above would be vacuous.

    The edge is directed: ``stream_edge("video", "action")`` lets action attend
    to video, so the video output must stay bit-comparable.
    """
    group, layout, modules, states, _ = two_streams()
    diag = AttentionPlan.from_visibility(same_stream(), layout)
    crossed = AttentionPlan.from_visibility(
        same_stream() | stream_edge("video", "action"), layout)
    a = run(JointAttentionEngine(), group, layout, modules, states, plan=diag)
    b = run(JointAttentionEngine(), group, layout, modules, states,
            plan=crossed)
    assert not torch.allclose(a.require("action").hidden,
                              b.require("action").hidden, atol=1e-6)
    # Directed: an incoming edge on action does not change the video result
    assert torch.allclose(a.require("video").hidden,
                          b.require("video").hidden, atol=1e-6)


def _self_attend(qkv: QKV) -> torch.Tensor:
    """Per-stream attention reference. In and out are both [B, L, H, D]."""
    q, k, v = (t.transpose(1, 2) for t in (qkv.q, qkv.k, qkv.v))
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(qkv.head_dim)
    out = torch.softmax(scores, dim=-1) @ v
    return out.transpose(1, 2)


def _per_stream_reference(group, modules, inputs) -> dict[str, StreamState]:
    """Run every layer stream by stream, bypassing the engine.

    Each stream only sees itself, which is the expected result under block
    diagonal visibility.
    """
    final: dict[str, StreamState] = {}
    for stream_spec in group.streams:
        module = modules[stream_spec.name]
        state = module.prepare(inputs[stream_spec.name], timestep=None)
        for layer in range(stream_spec.num_layers):
            qkv = module.pre_attn(state, local_layer_id=layer)
            state = module.post_attn(state, _self_attend(qkv),
                                     local_layer_id=layer)
        final[stream_spec.name] = module.finalize(state)
    return final


def test_single_stream_is_an_n_equals_one_degeneration():
    """A single stream takes the same path: ``plan.structural`` is None,
    ``routes`` is empty, and there is no special-case branch."""
    only = spec("video", num_layers=2)
    group = AttentionGroup(streams=(only,), num_layers=2)
    assert group.routes == ()
    layout = TokenLayout(segments=(SegmentRange("video", 0, VIDEO_TOKENS),),
                         labels={})
    trace: list = []
    modules = {"video": _ToyStream(only, trace=trace, seed=3)}
    gen = torch.Generator().manual_seed(11)
    states = {"video": torch.randn(BATCH, VIDEO_TOKENS, 16, generator=gen)}
    plan = AttentionPlan.from_visibility(None, layout)
    assert plan.structural is None
    out = run(JointAttentionEngine(), group, layout, modules, states, plan=plan)
    want = _per_stream_reference(group, modules, states)["video"].hidden
    assert torch.allclose(out.require("video").hidden, want, atol=1e-6)


def test_captures_are_keyed_by_stream_and_joint_layer():
    """Captures are keyed by ``(stream, joint layer id)`` and hold the hidden
    state **after that layer**, not the finalized output."""
    group, layout, modules, states, _ = two_streams()
    out = run(JointAttentionEngine(), group, layout, modules, states,
              captures={"action": (0, 2)})
    assert sorted(out.captured) == [("action", 0), ("action", 2)]
    grabbed = out.require_capture("action", 0)
    assert grabbed.shape == (BATCH, ACTION_TOKENS, 12)
    # What is captured is the hidden state after that layer, not post-finalize
    assert not torch.allclose(grabbed, out.require("action").hidden)
    with pytest.raises(KeyError, match="has no capture point"):
        out.require_capture("video", 0)


def test_captures_reject_unknown_stream_and_layer():
    """A capture naming a stream outside the group, or a layer that stream does
    not participate in, is rejected rather than silently dropped."""
    group, layout, modules, states, _ = two_streams()
    with pytest.raises(KeyError, match="requested streams outside this group"):
        run(JointAttentionEngine(), group, layout, modules, states,
            captures={"vl": (0,)})
    with pytest.raises(ValueError, match="does not participate in this layer"):
        run(JointAttentionEngine(), group, layout, modules, states,
            captures={"action": (7,)})


def test_strided_stream_participates_in_a_subset_of_layers():
    """Unequal layer counts are reconciled by ``LayerMapping``, not by requiring
    all streams to have the same depth.

    With 3 joint layers and ``stride=2`` the action stream must declare 2 layers
    and maps onto joint layers 0 and 2; declaring 3 is rejected.
    """
    group, layout, modules, states, trace = two_streams(
        num_layers=3, action_extra={"layer_mapping": LayerMapping.STRIDED,
                                    "stride": 2})
    # action still declares num_layers=3, which stride=2 over 3 joint layers
    # cannot resolve to 3 local layers
    with pytest.raises(ValueError, match="The two must be equal"):
        group.layer_index_maps()

    video_spec = spec("video", num_layers=3)
    action_spec = spec("action", num_layers=2, residual_width=12,
                       layer_mapping=LayerMapping.STRIDED, stride=2)
    group = AttentionGroup(streams=(video_spec, action_spec), num_layers=3)
    assert group.layer_index_maps() == {"video": {0: 0, 1: 1, 2: 2},
                                        "action": {0: 0, 2: 1}}
    trace = []
    modules = {"video": _ToyStream(video_spec, trace=trace, seed=1),
               "action": _ToyStream(action_spec, trace=trace, seed=2)}
    gen = torch.Generator().manual_seed(7)
    states = {
        "video": torch.randn(BATCH, VIDEO_TOKENS, 16, generator=gen),
        "action": torch.randn(BATCH, ACTION_TOKENS, 12, generator=gen),
    }
    plan = AttentionPlan.from_visibility(same_stream(), layout)
    out = run(JointAttentionEngine(), group, layout, modules, states,
              plan=plan)
    # Only video participates in joint layer 1, so the action stream's local
    # layer ids are just 0 and 1
    assert [item for item in trace if item[0] == "action"
            and item[1] == "pre_attn"] == [("action", "pre_attn", 0),
                                           ("action", "pre_attn", 1)]
    assert out.require("action").hidden.shape == (BATCH, ACTION_TOKENS, 12)
    assert out.require("video").hidden.shape == (BATCH, VIDEO_TOKENS, 16)


def test_restrict_to_streams_is_identity_on_the_common_case():
    """When every stream participates, ``restrict_to_streams`` returns the same
    objects, so the common case pays no ``index_select``."""
    layout = joint_layout()
    plan = AttentionPlan.from_visibility(same_stream(), layout)
    sub_layout, sub_plan = restrict_to_streams(layout, plan,
                                               ("video", "action"))
    assert sub_layout is layout
    assert sub_plan is plan


def test_restrict_to_streams_renumbers_and_selects_both_channels():
    """Restricting to a subset renumbers the segments from 0 and selects both
    mask channels: the structural mask and the per-sample ``key_valid``."""
    layout = joint_layout()
    key_valid = torch.ones((BATCH, layout.total), dtype=torch.bool)
    key_valid[1, -1] = False
    plan = AttentionPlan.from_visibility(same_stream(), layout,
                                         key_valid=key_valid)
    sub_layout, sub_plan = restrict_to_streams(layout, plan, ("action",))
    # The surviving segments are renumbered from 0 without gaps, because
    # TokenLayout requires the segments to cover [0, total)
    assert [(s.stream, s.start, s.stop) for s in sub_layout.segments] == \
        [("action", 0, ACTION_TOKENS)]
    assert sub_layout.labels_of("stream_id").tolist() == [1] * ACTION_TOKENS
    assert sub_plan.structural.shape == (ACTION_TOKENS, ACTION_TOKENS)
    assert bool(sub_plan.structural.all())      # fully visible within action
    assert sub_plan.key_valid.shape == (BATCH, ACTION_TOKENS)
    assert not bool(sub_plan.key_valid[1, -1])
    with pytest.raises(KeyError, match="layout has no stream"):
        restrict_to_streams(layout, plan, ("vl",))


def test_num_heads_must_match_across_participating_streams():
    """Joint attention concatenates along the token axis, so unequal
    ``num_heads`` is rejected at run time with a named error."""
    video_spec = spec("video")
    action_spec = StreamSpec(name="action", impl=Spec(impl=_ToyStream),
                             residual_width=12, num_heads=NUM_HEADS + 1,
                             head_dim=HEAD_DIM, num_layers=3,
                             timestep_embed_width=256)
    group = AttentionGroup(streams=(video_spec, action_spec), num_layers=3)
    trace: list = []
    modules = {"video": _ToyStream(video_spec, trace=trace, seed=1),
               "action": _ToyStream(action_spec, trace=trace, seed=2)}
    gen = torch.Generator().manual_seed(7)
    states = {
        "video": torch.randn(BATCH, VIDEO_TOKENS, 16, generator=gen),
        "action": torch.randn(BATCH, ACTION_TOKENS, 12, generator=gen),
    }
    with pytest.raises(ValueError, match="share num_heads"):
        run(JointAttentionEngine(), group, joint_layout(), modules, states)


def test_shape_signature_guards_c8_and_c9():
    """C8/C9: the set of streams and each stream's token count are invariant
    across the engine run, so a module cannot add or drop tokens."""
    before = {"video": StreamState(stream="video",
                                   hidden=torch.zeros(1, 8, 16))}
    assert_shape_signature(before, dict(before))
    with pytest.raises(ValueError, match="stream set changed across the layer"):
        assert_shape_signature(before, {})
    grown = {"video": StreamState(stream="video",
                                  hidden=torch.zeros(1, 9, 16))}
    with pytest.raises(ValueError, match="token count went from"):
        assert_shape_signature(before, grown)


def test_prepare_eats_a_raw_tensor_and_tokenizes_it():
    """``prepare`` receives the raw tensor: tokenization (patchify) belongs to
    the stream.

    This is nailed down with a stream whose input is not ``[B, L, W]``: if the
    engine wrapped the input into a ``StreamState`` before calling ``prepare``,
    that object could not even be constructed (``StreamState.hidden`` requires
    rank 3) and the test would fail before the run.
    """
    only = spec("video", num_layers=2)
    group = AttentionGroup(streams=(only,), num_layers=2)
    layout = TokenLayout(segments=(SegmentRange("video", 0, VIDEO_TOKENS),),
                         labels={})
    trace: list = []

    class _Patchifying(_ToyStream):
        def prepare(self, x, *, timestep, conditions=None):
            # The raw input is [B, C, L] and needs a transpose to be a token
            # sequence
            trace.append((self.spec.name, "prepare", timestep is None))
            return StreamState(stream=self.spec.name, hidden=x.transpose(1, 2))

    modules = {"video": _Patchifying(only, trace=trace, seed=3)}
    raw = {"video": torch.zeros(BATCH, 16, VIDEO_TOKENS)}
    out = run(JointAttentionEngine(), group, layout, modules, raw,
              plan=AttentionPlan.from_visibility(None, layout))
    assert out.require("video").hidden.shape == (BATCH, VIDEO_TOKENS, 16)


def test_prepare_must_produce_the_layout_segment_length():
    """The token count out of ``prepare`` must equal the layout segment length,
    which is where the geometry and the patchify are reconciled.

    Nowhere else reports the cause of an off-by-one: the concatenation only
    reports a wrong total, and the total is the sum over both streams.
    """
    group, layout, modules, raw, _ = two_streams()

    class _Short(_ToyStream):
        def prepare(self, x, *, timestep, conditions=None):
            return StreamState(stream=self.spec.name, hidden=x[:, :-1])

    modules["action"] = _Short(group.spec_of("action"), trace=[], seed=2)
    with pytest.raises(ValueError, match="the layout gives it a segment length"):
        run(JointAttentionEngine(), group, layout, modules, raw)


def test_module_changing_token_count_is_rejected():
    """C9: a stream whose ``finalize`` drops a token is rejected, because the
    caller indexes the output by the same layout it passed in."""
    group, layout, modules, states, _ = two_streams()

    class _Grower(_ToyStream):
        def finalize(self, state):
            return StreamState(stream=state.stream,
                               hidden=state.hidden[:, :-1])

    modules["action"] = _Grower(group.spec_of("action"), trace=[], seed=2)
    with pytest.raises(ValueError, match="C9"):
        run(JointAttentionEngine(), group, layout, modules, states)


def test_run_validates_stream_sets_and_segment_order():
    """``inputs``, ``modules`` and the layout segment order must all agree with
    the group (C8), so a reordered layout is an error rather than a silent
    permutation of the concatenated tokens."""
    group, layout, modules, states, _ = two_streams()
    engine = JointAttentionEngine()
    with pytest.raises(ValueError, match="inputs stream set"):
        run(engine, group, layout, modules,
            {"video": states["video"]})
    with pytest.raises(ValueError, match="modules stream set"):
        run(engine, group, layout, {"video": modules["video"]}, states)
    swapped = TokenLayout(
        segments=(SegmentRange("action", 0, ACTION_TOKENS),
                  SegmentRange("video", ACTION_TOKENS,
                               ACTION_TOKENS + VIDEO_TOKENS)),
        labels={})
    with pytest.raises(ValueError, match="C8"):
        run(engine, group, swapped, modules, states)


def test_dead_layer_is_rejected():
    """A joint layer in which no stream participates is a configuration error,
    not a no-op."""
    strided = spec("video", num_layers=2, residual_width=16,
                   layer_mapping=LayerMapping.AT_LAYERS,
                   participating_layers=(0, 1))
    group = AttentionGroup(streams=(strided,), num_layers=3)
    layout = TokenLayout(segments=(SegmentRange("video", 0, VIDEO_TOKENS),),
                         labels={})
    trace: list = []
    modules = {"video": _ToyStream(strided, trace=trace, seed=1)}
    states = {"video": torch.zeros(BATCH, VIDEO_TOKENS, 16)}
    with pytest.raises(ValueError, match="which makes it a dead layer"):
        run(JointAttentionEngine(), group, layout, modules, states)


def test_group_validates_members_and_routes():
    """``AttentionGroup`` rejects an empty member list, a non-positive layer
    count, duplicate stream names, unequal ``head_dim`` and bad routes.

    V3 allows ``num_heads`` and ``residual_width`` to differ per stream but not
    ``head_dim``, since the joint attention shares one score scale.
    """
    video_spec = spec("video")
    assert AttentionGroup(streams=(video_spec,),
                          num_layers=3).head_dim == HEAD_DIM
    with pytest.raises(ValueError, match="must not be empty"):
        AttentionGroup(streams=(), num_layers=3)
    with pytest.raises(ValueError, match="num_layers must be positive"):
        AttentionGroup(streams=(video_spec,), num_layers=0)
    with pytest.raises(ValueError, match="has duplicate stream names"):
        AttentionGroup(streams=(video_spec, spec("video")), num_layers=3)
    # V3: num_heads and residual_width may differ, head_dim may not
    wide = StreamSpec(name="action", impl=Spec(impl=_ToyStream),
                      residual_width=12, num_heads=NUM_HEADS,
                      head_dim=HEAD_DIM * 2, num_layers=3,
                      timestep_embed_width=256)
    with pytest.raises(ValueError, match="must share head_dim"):
        AttentionGroup(streams=(video_spec, wide), num_layers=3)
    with pytest.raises(ValueError, match="points at .*streams outside this group"):
        AttentionGroup(streams=(video_spec,), num_layers=3,
                       routes=(KVRoute(src="video", dst="vl",
                                       src_layers=(0,), dst_layers=(0,)),))
    # Route layer ids are **local** layer ids, not joint layer ids
    action_spec = spec("action", num_layers=2, residual_width=12,
                       layer_mapping=LayerMapping.STRIDED, stride=2)
    with pytest.raises(ValueError, match="local.. layer indices"):
        AttentionGroup(streams=(video_spec, action_spec), num_layers=3,
                       routes=(KVRoute(src="video", dst="action",
                                       src_layers=(0,), dst_layers=(2,)),))


def test_group_routes_into():
    """``routes_into`` returns the KV routes targeting one stream, and an empty
    tuple is the normal answer rather than an error."""
    video_spec = spec("video")
    action_spec = spec("action", residual_width=12)
    route = KVRoute(src="video", dst="action", src_layers=(0,),
                    dst_layers=(0,))
    group = AttentionGroup(streams=(video_spec, action_spec), num_layers=3,
                           routes=(route,))
    assert group.routes_into("action") == (route,)
    # An empty tuple is normal (always empty for a single-stream model)
    assert group.routes_into("video") == ()
    with pytest.raises(KeyError, match="AttentionGroup has no stream"):
        group.spec_of("vl")


def test_engine_backend_is_validated_at_construction():
    """An unsupported backend name and a ``flex`` backend without a block size
    are both rejected when the engine is built, not on the first forward."""
    with pytest.raises(ValueError, match="is not supported"):
        JointAttentionEngine(backend="flash")
    with pytest.raises(ValueError, match="requires block_size"):
        JointAttentionEngine(backend="flex")


def test_sdpa_backend_matches_dense_backend():
    """Both backends consume the same two-channel plan, so their results have to
    agree to 1e-6."""
    group, layout, modules, states, _ = two_streams()
    plan = AttentionPlan.from_visibility(
        same_stream() | stream_edge("video", "action"), layout)
    dense = run(JointAttentionEngine(backend="dense"), group, layout, modules,
                states, plan=plan)
    sdpa = run(JointAttentionEngine(backend="sdpa"), group, layout, modules,
               states, plan=plan)
    for name in group.stream_names:
        assert torch.allclose(dense.require(name).hidden,
                              sdpa.require(name).hidden, atol=1e-6)


class _ZeroOverride:
    """Takes over one layer and zeroes the attention output.

    Layers that go through an override are not captured into a CUDA graph.
    """

    def __init__(self, layer_id: int, *, drop: str | None = None):
        self.layer_id = layer_id
        self.drop = drop
        self.calls: list[int] = []

    def handles_layer(self, layer_id: int) -> bool:
        return layer_id == self.layer_id

    def run_attention(self, layer_id, qkvs, plan, layout):
        self.calls.append(layer_id)
        return {name: torch.zeros_like(qkv.q)
                for name, qkv in qkvs.items() if name != self.drop}


def test_override_takes_over_one_layer():
    """An override replaces the attention of exactly the layer it claims, and
    the result differs from the plain run."""
    group, layout, modules, states, _ = two_streams()
    plan = AttentionPlan.from_visibility(same_stream(), layout)
    override = _ZeroOverride(1)
    with_override = run(JointAttentionEngine(), group, layout, modules, states,
                        plan=plan, override=override)
    plain = run(JointAttentionEngine(), group, layout, modules, states,
                plan=plan)
    assert override.calls == [1]
    assert not torch.allclose(with_override.require("video").hidden,
                              plain.require("video").hidden, atol=1e-6)


def test_override_must_return_every_active_stream():
    """An override that omits an active stream is rejected, instead of leaving
    that stream without an attention output for the layer."""
    group, layout, modules, states, _ = two_streams()
    plan = AttentionPlan.from_visibility(same_stream(), layout)
    with pytest.raises(ValueError, match="LayerAttentionOverride returned stream set"):
        run(JointAttentionEngine(), group, layout, modules, states, plan=plan,
            override=_ZeroOverride(1, drop="action"))
