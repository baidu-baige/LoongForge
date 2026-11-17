# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 引擎：cat → 一次联合注意力 → split（交界面 I3）。

两条主断言：
  1. **cat→attn→split ≡ 逐流独立注意力**（当可见性恰好是按段块对角时）。
     这是「一次联合注意力」这条路没有把别的流的 token 混进来的证据。
  2. **调用顺序不可重排**：一层里全部参与流的 ``pre_attn`` 必须在
     任何流的 ``post_attn`` 之前完成。
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
    return StreamSpec(name=name, impl=Spec(impl=_ToyStream),
                      residual_width=residual_width, num_heads=NUM_HEADS,
                      head_dim=HEAD_DIM, num_layers=num_layers, **extra)


class _ToyStream:
    """最小 ``StreamModule``：四段各做一件可观测的事，不含 nn.Module。

    放在测试里而不是包里：它是脚手架，不是参考实现 ——
    包里多一个玩具 DiT，迟早有人把它当成模板。
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

    def prepare(self, state: StreamState, *, timestep):
        self.trace.append((self.spec.name, "prepare", timestep is None))
        return state

    def pre_attn(self, state: StreamState, *, local_layer_id: int) -> QKV:
        self.trace.append((self.spec.name, "pre_attn", local_layer_id))
        batch, tokens, _ = state.hidden.shape
        packed = state.hidden @ self.to_qkv[local_layer_id]
        shape = (batch, tokens, self.spec.num_heads, self.spec.head_dim)
        q, k, v = (part.reshape(shape) for part in packed.chunk(3, dim=-1))
        return QKV(q=q, k=k, v=v)

    def post_attn(self, state: StreamState, attn_out: torch.Tensor, *,
                  local_layer_id: int) -> StreamState:
        self.trace.append((self.spec.name, "post_attn", local_layer_id))
        batch, tokens = attn_out.shape[:2]
        flat = attn_out.reshape(batch, tokens, -1)
        return state.replace_hidden(
            state.hidden + flat @ self.from_attn[local_layer_id])

    def finalize(self, state: StreamState) -> StreamState:
        self.trace.append((self.spec.name, "finalize", None))
        return state.replace_hidden(state.hidden * 2.0)


VIDEO_TOKENS = 8
ACTION_TOKENS = 4
BATCH = 2


def joint_layout(*, video: int = VIDEO_TOKENS,
                 action: int = ACTION_TOKENS) -> TokenLayout:
    return TokenLayout(
        segments=(SegmentRange("video", 0, video),
                  SegmentRange("action", video, video + action)),
        labels={"stream": torch.tensor([0] * video + [1] * action)})


def two_streams(*, num_layers: int = 3, action_extra: dict | None = None):
    """两条流的完整入参：group / layout / modules / states / trace。"""
    video_spec = spec("video", num_layers=num_layers)
    action_spec = spec("action", num_layers=num_layers, residual_width=12,
                       **(action_extra or {}))
    group = AttentionGroup(members=(video_spec, action_spec),
                           num_layers=num_layers)
    trace: list = []
    modules = {"video": _ToyStream(video_spec, trace=trace, seed=1),
               "action": _ToyStream(action_spec, trace=trace, seed=2)}
    gen = torch.Generator().manual_seed(7)
    states = {
        "video": StreamState(stream="video", hidden=torch.randn(
            BATCH, VIDEO_TOKENS, video_spec.residual_width, generator=gen)),
        "action": StreamState(stream="action", hidden=torch.randn(
            BATCH, ACTION_TOKENS, action_spec.residual_width, generator=gen)),
    }
    return group, joint_layout(), modules, states, trace


def run(engine, group, layout, modules, states, *, plan=None, **extra):
    return engine.run(group, states, modules,
                      plan if plan is not None else AttentionPlan(layout=layout),
                      layout, conditions=ConditionBus(),
                      timesteps={name: None for name in group.stream_names},
                      **extra)


def test_call_order_is_not_reorderable():
    """一层里全部参与流 pre_attn → 一次注意力 → 各流 post_attn（§7）。"""
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
    group, layout, modules, states, _ = two_streams()
    out = run(JointAttentionEngine(), group, layout, modules, states)
    assert isinstance(out, EngineOutput)
    assert set(out.states) == {"video", "action"}
    assert out.require("video").hidden.shape == (BATCH, VIDEO_TOKENS, 16)
    assert out.require("action").hidden.shape == (BATCH, ACTION_TOKENS, 12)
    assert out.captured == {}
    with pytest.raises(KeyError, match="没有流"):
        out.require("vl")


def test_joint_attention_equals_per_stream_when_block_diagonal():
    """cat→attn→split ≡ 逐流独立注意力（可见性恰为 same_stream() 时）。

    这是「一次联合注意力」没有把别的流的 token 混进来的直接证据。
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
    """加一条跨流边必须改变结果 —— 否则上一条等价性测试是空的。"""
    group, layout, modules, states, _ = two_streams()
    diag = AttentionPlan.from_visibility(same_stream(), layout)
    crossed = AttentionPlan.from_visibility(
        same_stream() | stream_edge("video", "action"), layout)
    a = run(JointAttentionEngine(), group, layout, modules, states, plan=diag)
    b = run(JointAttentionEngine(), group, layout, modules, states,
            plan=crossed)
    assert not torch.allclose(a.require("action").hidden,
                              b.require("action").hidden, atol=1e-6)
    # 有向：action 的入边不改变 video 的结果
    assert torch.allclose(a.require("video").hidden,
                          b.require("video").hidden, atol=1e-6)


def _self_attend(qkv: QKV) -> torch.Tensor:
    """逐流独立注意力基准。进出都是 [B, L, H, D]。"""
    q, k, v = (t.transpose(1, 2) for t in (qkv.q, qkv.k, qkv.v))
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(qkv.head_dim)
    out = torch.softmax(scores, dim=-1) @ v
    return out.transpose(1, 2)


def _per_stream_reference(group, modules, states) -> dict[str, StreamState]:
    """不经引擎、逐流跑完全部层。各流只看自己 ⇒ 块对角可见性的应有结果。"""
    final: dict[str, StreamState] = {}
    for stream_spec in group.members:
        module = modules[stream_spec.name]
        state = module.prepare(states[stream_spec.name], timestep=None)
        for layer in range(stream_spec.num_layers):
            qkv = module.pre_attn(state, local_layer_id=layer)
            state = module.post_attn(state, _self_attend(qkv),
                                     local_layer_id=layer)
        final[stream_spec.name] = module.finalize(state)
    return final


def test_single_stream_is_an_n_equals_one_degeneration():
    """单流走的是同一条路：plan.structural 为 None、routes 为空，没有特例分支。"""
    only = spec("video", num_layers=2)
    group = AttentionGroup(members=(only,), num_layers=2)
    assert group.routes == ()
    layout = TokenLayout(segments=(SegmentRange("video", 0, VIDEO_TOKENS),),
                         labels={})
    trace: list = []
    modules = {"video": _ToyStream(only, trace=trace, seed=3)}
    gen = torch.Generator().manual_seed(11)
    states = {"video": StreamState(
        stream="video",
        hidden=torch.randn(BATCH, VIDEO_TOKENS, 16, generator=gen))}
    plan = AttentionPlan.from_visibility(None, layout)
    assert plan.structural is None
    out = run(JointAttentionEngine(), group, layout, modules, states, plan=plan)
    want = _per_stream_reference(group, modules, states)["video"].hidden
    assert torch.allclose(out.require("video").hidden, want, atol=1e-6)


def test_captures_are_keyed_by_stream_and_joint_layer():
    group, layout, modules, states, _ = two_streams()
    out = run(JointAttentionEngine(), group, layout, modules, states,
              captures={"action": (0, 2)})
    assert sorted(out.captured) == [("action", 0), ("action", 2)]
    grabbed = out.require_capture("action", 0)
    assert grabbed.shape == (BATCH, ACTION_TOKENS, 12)
    # 抓的是**该层之后**的隐状态，不是 finalize 之后的
    assert not torch.allclose(grabbed, out.require("action").hidden)
    with pytest.raises(KeyError, match="没有抓取点"):
        out.require_capture("video", 0)


def test_captures_reject_unknown_stream_and_layer():
    group, layout, modules, states, _ = two_streams()
    with pytest.raises(KeyError, match="不在本组里的流"):
        run(JointAttentionEngine(), group, layout, modules, states,
            captures={"vl": (0,)})
    with pytest.raises(ValueError, match="不参与这一层"):
        run(JointAttentionEngine(), group, layout, modules, states,
            captures={"action": (7,)})


def test_strided_stream_participates_in_a_subset_of_layers():
    """层数不等靠 ``LayerMapping`` 协调，不靠「三者全等」校验。"""
    group, layout, modules, states, trace = two_streams(
        num_layers=3, action_extra={"layer_mapping": LayerMapping.STRIDED,
                                    "stride": 2})
    # action 的 num_layers 仍是 3 ⇒ 与 stride=2 在联合层数 3 下解析不出 3 层
    with pytest.raises(ValueError, match="两者必须相等"):
        group.layer_index_maps()

    video_spec = spec("video", num_layers=3)
    action_spec = spec("action", num_layers=2, residual_width=12,
                       layer_mapping=LayerMapping.STRIDED, stride=2)
    group = AttentionGroup(members=(video_spec, action_spec), num_layers=3)
    assert group.layer_index_maps() == {"video": {0: 0, 1: 1, 2: 2},
                                        "action": {0: 0, 2: 1}}
    trace = []
    modules = {"video": _ToyStream(video_spec, trace=trace, seed=1),
               "action": _ToyStream(action_spec, trace=trace, seed=2)}
    gen = torch.Generator().manual_seed(7)
    states = {
        "video": StreamState(stream="video", hidden=torch.randn(
            BATCH, VIDEO_TOKENS, 16, generator=gen)),
        "action": StreamState(stream="action", hidden=torch.randn(
            BATCH, ACTION_TOKENS, 12, generator=gen)),
    }
    plan = AttentionPlan.from_visibility(same_stream(), layout)
    out = run(JointAttentionEngine(), group, layout, modules, states,
              plan=plan)
    # 联合层 1 只有 video 参与 ⇒ action 的局部层号只出现 0 和 1
    assert [item for item in trace if item[0] == "action"
            and item[1] == "pre_attn"] == [("action", "pre_attn", 0),
                                           ("action", "pre_attn", 1)]
    assert out.require("action").hidden.shape == (BATCH, ACTION_TOKENS, 12)
    assert out.require("video").hidden.shape == (BATCH, VIDEO_TOKENS, 16)


def test_restrict_to_streams_is_identity_on_the_common_case():
    """全部流都参与是首批两个模型的常态，不该为它付一次 index_select。"""
    layout = joint_layout()
    plan = AttentionPlan.from_visibility(same_stream(), layout)
    sub_layout, sub_plan = restrict_to_streams(layout, plan,
                                               ("video", "action"))
    assert sub_layout is layout
    assert sub_plan is plan


def test_restrict_to_streams_renumbers_and_selects_both_channels():
    layout = joint_layout()
    key_valid = torch.ones((BATCH, layout.total), dtype=torch.bool)
    key_valid[1, -1] = False
    plan = AttentionPlan.from_visibility(same_stream(), layout,
                                         key_valid=key_valid)
    sub_layout, sub_plan = restrict_to_streams(layout, plan, ("action",))
    # 剩下的段重新从 0 连续编号 —— TokenLayout 要求覆盖 [0, total)
    assert [(s.stream, s.start, s.stop) for s in sub_layout.segments] == \
        [("action", 0, ACTION_TOKENS)]
    assert sub_layout.labels_of("stream").tolist() == [1] * ACTION_TOKENS
    assert sub_plan.structural.shape == (ACTION_TOKENS, ACTION_TOKENS)
    assert bool(sub_plan.structural.all())      # action 段内部全可见
    assert sub_plan.key_valid.shape == (BATCH, ACTION_TOKENS)
    assert not bool(sub_plan.key_valid[1, -1])
    with pytest.raises(KeyError, match="布局里没有流"):
        restrict_to_streams(layout, plan, ("vl",))


def test_num_heads_must_match_across_participating_streams():
    video_spec = spec("video")
    action_spec = StreamSpec(name="action", impl=Spec(impl=_ToyStream),
                             residual_width=12, num_heads=NUM_HEADS + 1,
                             head_dim=HEAD_DIM, num_layers=3)
    group = AttentionGroup(members=(video_spec, action_spec), num_layers=3)
    trace: list = []
    modules = {"video": _ToyStream(video_spec, trace=trace, seed=1),
               "action": _ToyStream(action_spec, trace=trace, seed=2)}
    gen = torch.Generator().manual_seed(7)
    states = {
        "video": StreamState(stream="video", hidden=torch.randn(
            BATCH, VIDEO_TOKENS, 16, generator=gen)),
        "action": StreamState(stream="action", hidden=torch.randn(
            BATCH, ACTION_TOKENS, 12, generator=gen)),
    }
    with pytest.raises(ValueError, match="num_heads 相等"):
        run(JointAttentionEngine(), group, joint_layout(), modules, states)


def test_shape_signature_guards_c8_and_c9():
    before = {"video": StreamState(stream="video",
                                   hidden=torch.zeros(1, 8, 16))}
    assert_shape_signature(before, dict(before))
    with pytest.raises(ValueError, match="流集合变了"):
        assert_shape_signature(before, {})
    grown = {"video": StreamState(stream="video",
                                  hidden=torch.zeros(1, 9, 16))}
    with pytest.raises(ValueError, match="token 数"):
        assert_shape_signature(before, grown)


def test_module_changing_token_count_is_rejected():
    group, layout, modules, states, _ = two_streams()

    class _Grower(_ToyStream):
        def finalize(self, state):
            return StreamState(stream=state.stream,
                               hidden=state.hidden[:, :-1])

    modules["action"] = _Grower(group.spec_of("action"), trace=[], seed=2)
    with pytest.raises(ValueError, match="C9"):
        run(JointAttentionEngine(), group, layout, modules, states)


def test_run_validates_stream_sets_and_segment_order():
    group, layout, modules, states, _ = two_streams()
    engine = JointAttentionEngine()
    with pytest.raises(ValueError, match="states 的流集合"):
        run(engine, group, layout, modules,
            {"video": states["video"]})
    with pytest.raises(ValueError, match="modules 的流集合"):
        run(engine, group, layout, {"video": modules["video"]}, states)
    swapped = TokenLayout(
        segments=(SegmentRange("action", 0, ACTION_TOKENS),
                  SegmentRange("video", ACTION_TOKENS,
                               ACTION_TOKENS + VIDEO_TOKENS)),
        labels={})
    with pytest.raises(ValueError, match="C8"):
        run(engine, group, swapped, modules, states)


def test_dead_layer_is_rejected():
    """没有任何流参与的联合层是配置错误，不是空操作。"""
    strided = spec("video", num_layers=2, residual_width=16,
                   layer_mapping=LayerMapping.AT_LAYERS,
                   participating_layers=(0, 1))
    group = AttentionGroup(members=(strided,), num_layers=3)
    layout = TokenLayout(segments=(SegmentRange("video", 0, VIDEO_TOKENS),),
                         labels={})
    trace: list = []
    modules = {"video": _ToyStream(strided, trace=trace, seed=1)}
    states = {"video": StreamState(stream="video",
                                   hidden=torch.zeros(BATCH, VIDEO_TOKENS, 16))}
    with pytest.raises(ValueError, match="死层"):
        run(JointAttentionEngine(), group, layout, modules, states)


def test_group_validates_members_and_routes():
    video_spec = spec("video")
    assert AttentionGroup(members=(video_spec,),
                          num_layers=3).head_dim == HEAD_DIM
    with pytest.raises(ValueError, match="不得为空"):
        AttentionGroup(members=(), num_layers=3)
    with pytest.raises(ValueError, match="num_layers 必须为正"):
        AttentionGroup(members=(video_spec,), num_layers=0)
    with pytest.raises(ValueError, match="重名流"):
        AttentionGroup(members=(video_spec, spec("video")), num_layers=3)
    # V3：num_heads 与 residual_width 可以不等，head_dim 不行
    wide = StreamSpec(name="action", impl=Spec(impl=_ToyStream),
                      residual_width=12, num_heads=NUM_HEADS,
                      head_dim=HEAD_DIM * 2, num_layers=3)
    with pytest.raises(ValueError, match="head_dim 必须相等"):
        AttentionGroup(members=(video_spec, wide), num_layers=3)
    with pytest.raises(ValueError, match="不在本组里的流"):
        AttentionGroup(members=(video_spec,), num_layers=3,
                       routes=(KVRoute(src="video", dst="vl",
                                       src_layers=(0,), dst_layers=(0,)),))
    # 路由的层号是**局部**层号，不是联合层号
    action_spec = spec("action", num_layers=2, residual_width=12,
                       layer_mapping=LayerMapping.STRIDED, stride=2)
    with pytest.raises(ValueError, match="局部"):
        AttentionGroup(members=(video_spec, action_spec), num_layers=3,
                       routes=(KVRoute(src="video", dst="action",
                                       src_layers=(0,), dst_layers=(2,)),))


def test_group_routes_into():
    video_spec = spec("video")
    action_spec = spec("action", residual_width=12)
    route = KVRoute(src="video", dst="action", src_layers=(0,),
                    dst_layers=(0,))
    group = AttentionGroup(members=(video_spec, action_spec), num_layers=3,
                           routes=(route,))
    assert group.routes_into("action") == (route,)
    # 空元组是常态（单流模型恒为空），不是错误
    assert group.routes_into("video") == ()
    with pytest.raises(KeyError, match="没有流"):
        group.spec_of("vl")


def test_engine_backend_is_validated_at_construction():
    with pytest.raises(ValueError, match="不支持"):
        JointAttentionEngine(backend="flash")
    with pytest.raises(ValueError, match="必须给 block_size"):
        JointAttentionEngine(backend="flex")


def test_sdpa_backend_matches_dense_backend():
    """两个后端吃的是同一份两通道 plan，结果必须一致。"""
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
    """接管一层并把注意力输出清零。走 override 的层不进 CUDA graph 捕获。"""

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
    group, layout, modules, states, _ = two_streams()
    plan = AttentionPlan.from_visibility(same_stream(), layout)
    with pytest.raises(ValueError, match="返回的流集合"):
        run(JointAttentionEngine(), group, layout, modules, states, plan=plan,
            override=_ZeroOverride(1, drop="action"))
