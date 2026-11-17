# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""三个 provider 的端到端一致性：几何 → 布局 → 可见性 → 物化。

本文件只查**声明之间对不对账**，不查数值 —— 数值等价是迁权重那一步的事。
每条断言的对象都是「两个地方说的是不是同一件事」：
几何算出的 token 数与布局的段长、头声明的抓取点与 ``captures()``、
provider 声明的后端与能力协商的结论。
"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.builder import (
    join_stream_visibility, query_in_stream)
from loongforge.embodied.model.arch.layout import TokenLayout
from loongforge.embodied.model.arch.spec.core import get_provider, provider_names
from loongforge.embodied.model.arch.visibility.mask import (
    AttentionPlan, Capability, CompiledMask, Scenario)
from loongforge.embodied.model.arch.visibility.rules import same_stream

import loongforge.embodied.model.arch.spec.providers  # noqa: F401  import 即注册

MODEL_TYPES = ("vpp2", "fastwam", "internw0_delta")


@pytest.fixture(params=MODEL_TYPES)
def model_spec(request):
    return get_provider(request.param)()


def test_all_three_are_registered():
    for model_type in MODEL_TYPES:
        assert model_type in provider_names()
    # 六个拓扑族是不注册的工厂函数，YAML 与启动参数都看不见它们
    assert "dual_stream_wan" not in provider_names()


def test_provider_rejects_non_none_config(model_spec):
    """YAML 覆写尚未落地 ⇒ 对非 None 的 config 抛错而不是静默忽略。"""
    provider = get_provider(model_spec.model_type)
    with pytest.raises(NotImplementedError):
        provider({"geometry": {"num_frames": 9}})


def test_unknown_stream_scalar_is_rejected(model_spec):
    provider = get_provider(model_spec.model_type)
    with pytest.raises(ValueError):
        provider(video_layer_count=12)


def test_layout_segment_lengths_come_from_geometry(model_spec):
    geo = model_spec.geometry
    layout = TokenLayout.from_geometry(geo, model_spec.streams)
    assert layout.stream_names == model_spec.stream_names
    assert layout.range_of("video").length == geo.padded_seq_len
    assert layout.range_of("action").length == geo.action_horizon
    assert layout.total == geo.padded_seq_len + geo.action_horizon
    # 视频段的帧标签恰好是 tokens_per_frame 个一组
    frame = layout.labels_of("frame")[:geo.padded_seq_len]
    assert frame[:geo.tokens_per_frame].tolist() == [0] * geo.tokens_per_frame
    assert int(frame.max()) == geo.patch_grid[0] - 1
    # 动作段的 chunk == frame == step // ratio
    action = slice(geo.padded_seq_len, layout.total)
    assert torch.equal(layout.labels_of("chunk")[action],
                       layout.labels_of("frame")[action])


def test_joint_group_is_one_to_one_over_both_streams(model_spec):
    assert len(model_spec.groups) == 1
    group = model_spec.groups[0]
    assert group.stream_names == ("video", "action")
    # 层数不等靠 LayerMapping 协调，但这三个模型都是等深的 ONE_TO_ONE
    assert group.layer_index_maps() == {
        name: {i: i for i in range(group.num_layers)}
        for name in group.stream_names}
    # V3 只管住 head_dim；residual_width 两条流不等才是常态
    assert group.head_dim == model_spec.stream("video").head_dim
    widths = {s.name: s.residual_width for s in group.members}
    assert widths["video"] != widths["action"], widths
    assert group.routes == ()


@pytest.fixture
def layout(model_spec):
    return TokenLayout.from_geometry(model_spec.geometry, model_spec.streams)


def test_visibility_keys_are_stream_names(model_spec):
    assert set(model_spec.visibility) <= set(model_spec.stream_names)


def test_each_declared_rule_actually_hides_something(model_spec, layout):
    """声明了却求值全 True 的规则等于没声明 —— 那是配置丢了的典型症状。"""
    for name, vis in model_spec.visibility.items():
        out = vis.evaluate(layout)
        assert out.shape == (layout.total, layout.total), name
        assert out.dtype == torch.bool, name
        assert not bool(out.all()), (name, vis.expr)


def test_cross_stream_edge_is_one_way(model_spec, layout):
    """动作读视频、视频不读动作。三个模型都靠「有入边无出边」表达这件事。"""
    geo = model_spec.geometry
    video = slice(0, geo.padded_seq_len)
    action = slice(geo.padded_seq_len, layout.total)
    on_action = model_spec.visibility["action"].evaluate(layout)
    assert bool(on_action[action, video].any())
    # 不是全开 —— 否则 prefix_only / frame_causal 那一项白写了
    assert not bool(on_action[action, video].all())
    # 视频 query 行不得开任何动作列
    on_video = model_spec.visibility["video"].evaluate(layout)
    assert not bool(on_video[video, action].any()), \
        model_spec.visibility["video"].expr


def test_block_size_is_given_exactly_when_backend_is_flex(model_spec):
    """V9 的账：flex 要块大小，dense/sdpa 给了也用不上 ⇒ 不许给。"""
    assert (model_spec.attention_block_size is not None) == \
        (model_spec.attention_backend == "flex")


def test_declared_backend_negotiates_exact(model_spec):
    """provider 自己声明的后端必须能精确表达它自己声明的可见性。

    这条不成立意味着 spec 内部打架：后端那一行与规则那几行是两个人写的。
    """
    scenario = Scenario(phase="train",
                        backend=model_spec.attention_backend,
                        block_size=model_spec.attention_block_size)
    for name, vis in model_spec.visibility.items():
        assert CompiledMask.supports(vis, scenario) is Capability.EXACT, \
            (name, vis.expr)


def test_plan_materializes_on_the_declared_backend(model_spec, layout):
    plan = AttentionPlan.from_visibility(model_spec.visibility["action"],
                                         layout)
    shape = (1, 1, layout.total, layout.total)
    assert plan.materialize("dense").shape == shape
    got = plan.materialize(model_spec.attention_backend,
                           block_size=model_spec.attention_block_size)
    assert tuple(got.shape) == shape


def test_every_head_sources_a_real_stream_and_layer(model_spec):
    for head in model_spec.heads:
        assert head.source.stream in model_spec.stream_names, head.name
        if head.source.layer is not None:
            depth = model_spec.stream(head.source.stream).num_layers
            assert 0 <= head.source.layer < depth, head.name


def test_captures_are_derived_from_the_head_table(model_spec):
    """``captures()`` 不是第二处声明，它必须恰好是头表里 layer 不为 None 的那些。

    两处各写一遍的症状：引擎抓了没人用的层（白耗显存），
    或者头去取一个没被抓的层（KeyError 在 forward 里才炸）。
    """
    want: dict[str, list[int]] = {}
    for head in model_spec.heads:
        if head.source.layer is not None:
            want.setdefault(head.source.stream, []).append(head.source.layer)
    assert model_spec.captures() == {
        stream: tuple(sorted(set(layers))) for stream, layers in want.items()}
    # 吃最终输出的头（layer=None）不进抓取表 —— 三个模型都至少有两个这样的头
    assert sum(h.source.layer is None for h in model_spec.heads) >= 2


def test_each_loss_names_an_existing_head(model_spec):
    heads = {head.name for head in model_spec.heads}
    for loss in model_spec.diffusion.losses:
        assert loss.name in heads, (loss.name, sorted(heads))


def test_joint_mask_gives_each_stream_its_own_rows(model_spec, layout):
    """合成后的联合 mask：每条流的 query 行恰好等于它自己那条规则的那些行。

    这是 ``join_stream_visibility`` 的全部语义。写成 ``&`` 的那一版在这里
    露馅：video 的行会被 action 的规则一起遮掉，形状却分毫不差。
    """
    joint = join_stream_visibility(model_spec.visibility).evaluate(layout)
    for name, vis in model_spec.visibility.items():
        rows = layout.range_of(name)
        own = vis.evaluate(layout)[rows.start:rows.stop]
        assert torch.equal(joint[rows.start:rows.stop], own), name


def test_joint_mask_leaves_undeclared_streams_fully_visible(layout):
    """只声明一条流 ⇒ 另一条流的行全可见，而不是跟着受限。"""
    only_action = {"action": same_stream()}
    joint = join_stream_visibility(only_action).evaluate(layout)
    video = layout.range_of("video")
    assert bool(joint[video.start:video.stop].all())
    assert join_stream_visibility({}) is None


def test_query_in_stream_rejects_unknown_name(layout):
    with pytest.raises(KeyError, match="布局里没有流"):
        query_in_stream("typo").evaluate(layout)
