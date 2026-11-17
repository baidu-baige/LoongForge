# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 物化：两通道、四个后端与能力协商（交界面 I1）。

核心断言是 **flex ≡ dense 逐元素相等**：同一份 ``structural`` 物化出的
BlockMask 与 dense bool 必须选中同一批 (q, kv)。两者分叉的症状是
训练收敛但等价性测试看见了不该看见的 token，在 loss 上看不出来。
"""

from __future__ import annotations

import math

import pytest
import torch

from loongforge.embodied.model.arch.layout import SegmentRange, TokenLayout
from loongforge.embodied.model.arch.visibility.mask import (
    KNOWN_BACKENDS, AttentionPlan, Capability, CompiledMask, Scenario)
from loongforge.embodied.model.arch.visibility.rules import (
    frame_causal, same_stream, stream_edge)

HAS_FLEX = torch.cuda.is_available()
try:  # flex 只在 torch>=2.5 且有 CUDA 时可用
    from torch.nn.attention.flex_attention import (  # noqa: F401
        create_block_mask, flex_attention)
except ImportError:  # pragma: no cover - 取决于 torch 版本
    HAS_FLEX = False
needs_flex = pytest.mark.skipif(
    not HAS_FLEX, reason="FlexAttention 需要 CUDA 与 torch>=2.5")


@pytest.fixture
def exact_matmul():
    """关掉 TF32。等价性比较必须在同一精度下做，否则精度差盖住语义差。"""
    prior = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = prior


TOTAL = 128      # 取 block_size 的整数倍，免得 BlockMask 的尾块干扰逐元素比较
VIDEO = 96


def layout(total: int = TOTAL) -> TokenLayout:
    frame = torch.arange(total, dtype=torch.long) // 32
    stream = (torch.arange(total, dtype=torch.long) >= VIDEO).long()
    return TokenLayout(
        segments=(SegmentRange("video", 0, VIDEO),
                  SegmentRange("action", VIDEO, total)),
        labels={"stream": stream, "frame": frame})


def plan(**extra) -> AttentionPlan:
    lay = layout()
    vis = same_stream() | (stream_edge("video", "action") & frame_causal(2))
    return AttentionPlan.from_visibility(vis, lay, **extra)


def test_structural_must_not_carry_batch_dim():
    lay = layout()
    with pytest.raises(ValueError, match="INV-1"):
        AttentionPlan(structural=torch.ones((2, TOTAL, TOTAL),
                                            dtype=torch.bool), layout=lay)
    with pytest.raises(ValueError, match="必须是方阵"):
        AttentionPlan(structural=torch.ones((TOTAL, TOTAL - 1),
                                            dtype=torch.bool))
    with pytest.raises(ValueError, match="不一致"):
        AttentionPlan(structural=torch.ones((8, 8), dtype=torch.bool),
                      layout=lay)


def test_key_valid_must_be_two_dims():
    lay = layout()
    with pytest.raises(ValueError, match=r"\[B, total\]"):
        AttentionPlan(key_valid=torch.ones(TOTAL, dtype=torch.bool),
                      layout=lay)
    with pytest.raises(ValueError, match="!= layout.total"):
        AttentionPlan(key_valid=torch.ones((2, 8), dtype=torch.bool),
                      layout=lay)


def test_structural_none_is_the_single_stream_norm():
    """单流 + 无因果 ⇒ 全可见，此时不物化 [L, L] 全 True 张量。"""
    lay = layout()
    empty = AttentionPlan.from_visibility(None, lay)
    assert empty.structural is None
    assert empty.materialize("dense") is None
    assert empty.total == lay.total


def test_dense_bool_broadcasts_when_key_valid_absent():
    dense = plan().dense_bool()
    assert dense.shape == (1, 1, TOTAL, TOTAL)
    assert dense.dtype == torch.bool


def test_two_channels_combine_without_touching_structural():
    lay = layout()
    key_valid = torch.ones((2, TOTAL), dtype=torch.bool)
    key_valid[1, -4:] = False
    both = plan(key_valid=key_valid)
    dense = both.dense_bool()
    assert dense.shape == (2, 1, TOTAL, TOTAL)
    structural = both.structural
    assert torch.equal(dense[0, 0], structural)
    assert not bool(dense[1, 0, :, -4:].any())
    # 结构通道本身没有被就地改掉 —— 它是 batch 共享、可缓存的那一半
    assert torch.equal(both.structural, structural)


def test_sdpa_bias_is_additive_minus_inf():
    p = plan()
    bias = p.materialize("sdpa")
    dense = p.dense_bool()
    assert bias.dtype == torch.float32
    assert bias.shape == dense.shape
    assert torch.equal(bias == 0.0, dense)
    assert torch.isinf(bias[~dense]).all()


def test_unknown_backend_lists_registered_ones():
    with pytest.raises(ValueError, match="未知后端"):
        plan().materialize("flash")
    assert KNOWN_BACKENDS == ("dense", "sdpa", "flex", "varlen")


def test_flex_requires_block_size_and_structural():
    p = plan()
    with pytest.raises(ValueError, match="必须给 block_size"):
        p.materialize("flex")
    only_valid = AttentionPlan(key_valid=torch.ones((2, TOTAL),
                                                    dtype=torch.bool),
                               layout=layout())
    with pytest.raises(ValueError, match="结构通道为 None"):
        only_valid.materialize("flex", block_size=64)


@needs_flex
def test_flex_block_mask_equals_dense_attention(exact_matmul):
    """flex 与 dense 必须选中同一批 (q, kv)。

    比的是**注意力输出**而不是 BlockMask 的块级 ``to_dense()``：
    块级表示在块内是全 1，逐元素的差异只有把 mask 真正用进 kernel 才看得见。
    """
    device = torch.device("cuda")
    lay = layout()
    vis = same_stream() | (stream_edge("video", "action") & frame_causal(2))
    structural = vis.evaluate(lay).to(device)
    p = AttentionPlan(structural=structural, layout=lay)
    got = _flex_out(p.materialize("flex", block_size=64), device)
    want = _dense_out(p.dense_bool(), device)
    assert torch.allclose(got, want, atol=1e-6), \
        float((got - want).abs().max())


@needs_flex
def test_flex_block_mask_carries_key_valid(exact_matmul):
    device = torch.device("cuda")
    lay = layout()
    structural = same_stream().evaluate(lay).to(device)
    key_valid = torch.ones((2, TOTAL), dtype=torch.bool, device=device)
    key_valid[1, -8:] = False
    p = AttentionPlan(structural=structural, key_valid=key_valid, layout=lay)
    block = p.materialize("flex", block_size=64)
    assert tuple(block.shape) == (2, 1, TOTAL, TOTAL)
    got = _flex_out(block, device, batch=2)
    want = _dense_out(p.dense_bool(), device, batch=2)
    assert torch.allclose(got, want, atol=1e-6), \
        float((got - want).abs().max())


def _qkv(device: torch.device, batch: int = 1):
    """固定种子的 [B, H, L, D] 三元组，两条路径共用同一份输入。"""
    gen = torch.Generator(device="cpu").manual_seed(0)
    shape = (batch, 2, TOTAL, 16)
    return tuple(torch.randn(shape, generator=gen).to(device)
                 for _ in range(3))


def _flex_out(block_mask, device: torch.device, batch: int = 1):
    from torch.nn.attention.flex_attention import flex_attention
    q, k, v = _qkv(device, batch)
    return flex_attention(q, k, v, block_mask=block_mask)


def _dense_out(dense: torch.Tensor, device: torch.device, batch: int = 1):
    """手写 masked softmax 基准。

    不用 ``scaled_dot_product_attention`` 做基准：它在 CUDA 上走 TF32 矩阵乘，
    与 flex 的 triton kernel 有 ~5e-4 的精度差，会盖住真正要查的
    「两条路径选中的 (q, kv) 不同」这类差异（后者量级在 1e-1）。
    """
    q, k, v = _qkv(device, batch)
    mask = dense.to(device)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def test_as_segments_requires_block_diagonal():
    lay = layout()
    # same_stream() 恰好是按段块对角的 ⇒ varlen 表达得了
    diag = same_stream().compile(lay, backend="varlen")
    assert diag.as_segments() == lay.segments
    # 多一条跨流边就不是了，必须抛错而不是返回段表
    crossed = (same_stream() | stream_edge("video", "action")).compile(
        lay, backend="varlen")
    with pytest.raises(ValueError, match="不是按段块对角"):
        crossed.as_segments()


def test_compiled_mask_block_size_guard():
    compiled = same_stream().compile(layout(), backend="flex")
    with pytest.raises(ValueError, match="block_size 必须为正"):
        compiled.as_block_mask(block_size=0)


def test_scenario_rejects_auto_backend():
    with pytest.raises(ValueError, match="必须已解析成具体后端"):
        Scenario(phase="train", backend="auto")


def test_supports_negotiation_table():
    vis = same_stream() | stream_edge("video", "action")

    def cap(**kwargs) -> Capability:
        return CompiledMask.supports(vis, Scenario(phase="train", **kwargs))

    assert cap(backend="dense") is Capability.EXACT
    assert cap(backend="sdpa") is Capability.EXACT
    assert cap(backend="flex", block_size=64) is Capability.EXACT
    # 没有块大小 / 带 KV cache ⇒ 只能退 dense，算降级而不是不支持
    assert cap(backend="flex") is Capability.DEGRADED
    assert cap(backend="flex", block_size=64,
               has_kv_cache=True) is Capability.DEGRADED
    # varlen 一档宁过严不过宽：带跨流边的表达式判不支持
    assert cap(backend="varlen") is Capability.UNSUPPORTED
    assert CompiledMask.supports(
        same_stream(), Scenario(phase="train",
                                backend="varlen")) is Capability.EXACT
    assert cap(backend="flash") is Capability.UNSUPPORTED
