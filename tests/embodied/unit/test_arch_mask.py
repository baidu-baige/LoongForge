# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 materialisation: two channels, four backends, capability negotiation (I1).

An ``AttentionPlan`` splits visibility into two channels: ``structural``, a
``[total, total]`` batch-shared bool matrix, and ``key_valid``, a ``[B, total]``
per-sample validity row. ``AttentionPlan`` plus ``CompiledMask`` form interface
I1, the only way Lane B may ask about visibility.

The central assertion is **flex == dense elementwise**: the BlockMask and the
dense bool materialised from one ``structural`` must select the same set of
(q, kv) pairs. If they diverge, training still converges while the equivalence
test sees tokens it should not, and the loss curve shows nothing.
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
try:  # flex needs torch>=2.5 and CUDA
    from torch.nn.attention.flex_attention import (  # noqa: F401
        create_block_mask, flex_attention)
except ImportError:  # pragma: no cover - depends on the torch version
    HAS_FLEX = False
needs_flex = pytest.mark.skipif(
    not HAS_FLEX, reason="FlexAttention needs CUDA and torch>=2.5")


@pytest.fixture
def exact_matmul():
    """Disable TF32 so the equivalence comparison runs at one precision.

    With TF32 on, the numeric difference between the two matmul paths is large
    enough to hide the semantic difference being looked for.
    """
    prior = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = prior


TOTAL = 128      # A multiple of block_size, so BlockMask has no tail block
VIDEO = 96       # Video tokens; the remaining 32 are action tokens


def layout(total: int = TOTAL) -> TokenLayout:
    """Two segments, 96 video tokens then 32 action ones, 32 tokens per frame."""
    frame = torch.arange(total, dtype=torch.long) // 32
    stream = (torch.arange(total, dtype=torch.long) >= VIDEO).long()
    return TokenLayout(
        segments=(SegmentRange("video", 0, VIDEO),
                  SegmentRange("action", VIDEO, total)),
        labels={"stream_id": stream, "frame": frame})


def plan(**extra) -> AttentionPlan:
    """A plan for "within a stream, plus video to action under frame causality"."""
    lay = layout()
    vis = same_stream() | (stream_edge("video", "action") & frame_causal(2))
    return AttentionPlan.from_visibility(vis, lay, **extra)


def test_structural_must_not_carry_batch_dim():
    """``structural`` is exactly 2-D and square, and matches ``layout.total``.

    INV-1: a batch dimension on the structural channel would make the whole
    mask per-sample, which kills BlockMask reuse and forces every backend down
    to dense.
    """
    lay = layout()
    with pytest.raises(ValueError, match="INV-1"):
        AttentionPlan(structural=torch.ones((2, TOTAL, TOTAL),
                                            dtype=torch.bool), layout=lay)
    with pytest.raises(ValueError, match="must be square"):
        AttentionPlan(structural=torch.ones((TOTAL, TOTAL - 1),
                                            dtype=torch.bool))
    with pytest.raises(ValueError, match="disagrees with layout.total"):
        AttentionPlan(structural=torch.ones((8, 8), dtype=torch.bool),
                      layout=lay)


def test_key_valid_must_be_two_dims():
    """``key_valid`` is ``[B, total]``, the per-sample channel.

    A 1-D tensor would be ambiguous between a batch row and a token row, and a
    wrong second dimension would broadcast padding onto the wrong tokens.
    """
    lay = layout()
    with pytest.raises(ValueError, match=r"\[B, total\]"):
        AttentionPlan(key_valid=torch.ones(TOTAL, dtype=torch.bool),
                      layout=lay)
    with pytest.raises(ValueError, match="!= layout.total"):
        AttentionPlan(key_valid=torch.ones((2, 8), dtype=torch.bool),
                      layout=lay)


def test_structural_none_is_the_single_stream_norm():
    """One stream with no causality means ``structural is None``, not all-True.

    Materialising a ``[L, L]`` all-True matrix would waste memory and block the
    attention kernels' unmasked fast path, so None is the normal case for a
    single-stream model.
    """
    lay = layout()
    empty = AttentionPlan.from_visibility(None, lay)
    assert empty.structural is None
    assert empty.materialize("dense") is None
    assert empty.total == lay.total


def test_dense_bool_broadcasts_when_key_valid_absent():
    """Without ``key_valid``, ``dense_bool`` is ``[1, 1, L, L]`` and broadcasts.

    Keeping the batch and head axes at size 1 means one allocation is shared by
    the whole batch instead of being expanded per sample.
    """
    dense = plan().dense_bool()
    assert dense.shape == (1, 1, TOTAL, TOTAL)
    assert dense.dtype == torch.bool


def test_two_channels_combine_without_touching_structural():
    """Combining the channels yields ``[B, 1, L, L]`` and leaves the first alone.

    The structural channel is the batch-shared, cacheable half; an in-place
    ``&`` with ``key_valid`` would poison it for every later sample.
    """
    lay = layout()
    key_valid = torch.ones((2, TOTAL), dtype=torch.bool)
    key_valid[1, -4:] = False
    both = plan(key_valid=key_valid)
    dense = both.dense_bool()
    assert dense.shape == (2, 1, TOTAL, TOTAL)
    structural = both.structural
    assert torch.equal(dense[0, 0], structural)
    assert not bool(dense[1, 0, :, -4:].any())
    # The structural channel was not modified in place: it is the batch-shared,
    # cacheable half
    assert torch.equal(both.structural, structural)


def test_sdpa_bias_is_additive_minus_inf():
    """The sdpa form is an additive float bias: 0.0 where visible, -inf elsewhere.

    ``scaled_dot_product_attention`` adds this to the scores, so the zero set
    must coincide exactly with the dense bool mask.
    """
    p = plan()
    bias = p.materialize("sdpa")
    dense = p.dense_bool()
    assert bias.dtype == torch.float32
    assert bias.shape == dense.shape
    assert torch.equal(bias == 0.0, dense)
    assert torch.isinf(bias[~dense]).all()


def test_unknown_backend_lists_registered_ones():
    """An unregistered backend name fails at materialisation time.

    ``KNOWN_BACKENDS`` is the closed set ``("dense", "sdpa", "flex",
    "varlen")``; a typo like "flash" must not fall through to a default.
    """
    with pytest.raises(ValueError, match="unknown backend"):
        plan().materialize("flash")
    assert KNOWN_BACKENDS == ("dense", "sdpa", "flex", "varlen")


def test_flex_requires_block_size_and_structural():
    """The flex path needs both a positive ``block_size`` and a structural mask.

    A BlockMask is defined in terms of blocks, and the per-sample channel alone
    carries no (q, kv) structure to build one from.
    """
    p = plan()
    with pytest.raises(ValueError, match="requires block_size"):
        p.materialize("flex")
    only_valid = AttentionPlan(key_valid=torch.ones((2, TOTAL),
                                                    dtype=torch.bool),
                               layout=layout())
    with pytest.raises(ValueError, match="structural channel is None"):
        only_valid.materialize("flex", block_size=64)


@needs_flex
def test_flex_block_mask_equals_dense_attention(exact_matmul):
    """flex and dense must select the same (q, kv) pairs for one visibility.

    The comparison is on the **attention output**, not on the BlockMask's
    block-level ``to_dense()``: the block-level form is all-ones inside a
    block, so an elementwise difference only shows once the mask is used by
    the kernel.
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
    """The per-sample channel survives into the BlockMask, giving ``[B, 1, L, L]``.

    The two channels are combined inside one ``mask_mod`` rather than ANDed
    into a ``[B, Q, K]`` tensor, so block-level skipping still works; this
    checks that the combination is also numerically right.
    """
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
    """A seeded ``[B, H, L, D]`` triple; both paths consume the same input."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    shape = (batch, 2, TOTAL, 16)
    return tuple(torch.randn(shape, generator=gen).to(device)
                 for _ in range(3))


def _flex_out(block_mask, device: torch.device, batch: int = 1):
    """Attention output of the FlexAttention path under ``block_mask``."""
    from torch.nn.attention.flex_attention import flex_attention
    q, k, v = _qkv(device, batch)
    return flex_attention(q, k, v, block_mask=block_mask)


def _dense_out(dense: torch.Tensor, device: torch.device, batch: int = 1):
    """Hand-written masked softmax reference.

    ``scaled_dot_product_attention`` is deliberately not the reference: on CUDA
    it uses TF32 matmuls, which differ from flex's triton kernel by about 5e-4
    and would hide the difference actually under test, "the two paths selected
    different (q, kv) pairs", whose magnitude is around 1e-1.
    """
    q, k, v = _qkv(device, batch)
    mask = dense.to(device)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def test_as_segments_requires_block_diagonal():
    """``as_segments`` only succeeds when the mask is block diagonal per segment.

    varlen attention can express nothing but per-segment blocks, so a mask with
    a cross-stream edge has to raise rather than return a segment table that
    silently drops that edge.
    """
    lay = layout()
    # same_stream() happens to be exactly block diagonal per segment, so varlen
    # can express it
    diag = same_stream().compile(lay, backend="varlen")
    assert diag.as_segments() == lay.segments
    # One cross-stream edge breaks that, and must raise instead of returning a
    # segment table
    crossed = (same_stream() | stream_edge("video", "action")).compile(
        lay, backend="varlen")
    with pytest.raises(ValueError, match="not block-diagonal by segment"):
        crossed.as_segments()


def test_compiled_mask_block_size_guard():
    """``as_block_mask`` rejects a non-positive ``block_size``.

    A zero block size would otherwise reach ``create_block_mask`` and fail deep
    inside the kernel launch.
    """
    compiled = same_stream().compile(layout(), backend="flex")
    with pytest.raises(ValueError, match="block_size must be positive"):
        compiled.as_block_mask(block_size=0)


def test_scenario_rejects_auto_backend():
    """A ``Scenario`` carries a resolved backend, never ``"auto"``.

    ``auto`` has to be resolved at build time; negotiating against it would
    make the capability answer depend on something only known at run time.
    """
    with pytest.raises(ValueError, match="must already have resolved it to a concrete backend"):
        Scenario(phase="train", backend="auto")


def test_supports_negotiation_table():
    """``CompiledMask.negotiate_capability`` grades each backend EXACT/DEGRADED/UNSUPPORTED.

    The grading is what lets the engine pick a backend without trying it:
    dense, sdpa and blocked flex are exact; flex without a block size or with a
    KV cache falls back to dense, which is a degradation rather than a refusal;
    varlen errs strict and refuses any cross-stream edge.
    """
    vis = same_stream() | stream_edge("video", "action")

    def cap(**kwargs) -> Capability:
        return CompiledMask.negotiate_capability(
            vis, Scenario(phase="train", **kwargs))

    assert cap(backend="dense") is Capability.EXACT
    assert cap(backend="sdpa") is Capability.EXACT
    assert cap(backend="flex", block_size=64) is Capability.EXACT
    # No block size, or a KV cache present, means falling back to dense: a
    # degradation, not a refusal
    assert cap(backend="flex") is Capability.DEGRADED
    assert cap(backend="flex", block_size=64,
               has_kv_cache=True) is Capability.DEGRADED
    # varlen errs on the strict side: an expression with a cross-stream edge is
    # reported unsupported
    assert cap(backend="varlen") is Capability.UNSUPPORTED
    assert CompiledMask.negotiate_capability(
        same_stream(), Scenario(phase="train",
                                backend="varlen")) is Capability.EXACT
    assert cap(backend="flash") is Capability.UNSUPPORTED
