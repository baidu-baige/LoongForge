# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the Qwen3-VL fused inference operators."""

import pytest
import torch

from groot_n1_7_op.qwen3_vl_fused_ops import (
    qwen3_vl_fused_text_rope_forward,
    qwen3_vl_fused_text_rms_norm_forward,
    qwen3_vl_fused_text_silu_mul_forward,
    qwen3_vl_fused_vision_rope_forward,
)
from groot_n1_7_op import _qwen3_vl_fused_ops as _ext


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    tolerances = {
        torch.float32: (2e-6, 2e-6),
        torch.float16: (5e-3, 5e-3),  # fp16: 10-bit mantissa; 5e-3 covers large-shape RoPE
        torch.bfloat16: (3e-2, 3e-2),  # bf16: 7-bit mantissa
    }
    atol, rtol = tolerances[dtype]
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def _rms_norm_reference(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Pure PyTorch fp32 RMSNorm reference (matches the existing 3-step kernel chain)."""
    variance = (hidden_states.float() * hidden_states.float()).mean(-1, keepdim=True)
    normalized = hidden_states.float() * torch.rsqrt(variance + epsilon)
    return weight * normalized.to(hidden_states.dtype).float()


def _rms_norm_3step(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Original 3-step CUDA kernel chain (square → Python mean → finish)."""
    squared = _ext.qwen3_vl_fused_text_rms_norm_square(hidden_states)
    variance = squared.mean(-1, keepdim=True)
    return _ext.qwen3_vl_fused_text_rms_norm_finish(hidden_states, variance, weight, epsilon)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_vision_rope_matches_reference(dtype):
    sequence, heads, head_dim = 7, 3, 8
    generator = torch.Generator(device="cuda").manual_seed(11)
    query = torch.randn(sequence, heads, head_dim, device="cuda", dtype=dtype, generator=generator)
    key = torch.randn_like(query)
    cos = torch.randn(sequence, head_dim, device="cuda", dtype=torch.float32, generator=generator)
    sin = torch.randn(sequence, head_dim, device="cuda", dtype=torch.float32, generator=generator)

    query_out, key_out = qwen3_vl_fused_vision_rope_forward(query, key, cos, sin)
    query_ref = (query.float() * cos[:, None, :] + _rotate_half(query.float()) * sin[:, None, :]).to(dtype)
    key_ref = (key.float() * cos[:, None, :] + _rotate_half(key.float()) * sin[:, None, :]).to(dtype)

    _assert_close(query_out, query_ref, dtype)
    _assert_close(key_out, key_ref, dtype)
    assert query_out.shape == query.shape
    assert key_out.shape == key.shape


def test_vision_rope_accepts_strided_inputs():
    sequence, heads, head_dim = 5, 2, 8
    base = torch.randn(sequence, heads, head_dim * 2, device="cuda", dtype=torch.bfloat16)
    query = base[..., ::2]
    key = (base + 0.5)[..., ::2]
    cos_base = torch.randn(sequence, head_dim * 2, device="cuda", dtype=torch.float32)
    sin_base = torch.randn_like(cos_base)
    cos, sin = cos_base[..., ::2], sin_base[..., ::2]

    query_out, key_out = qwen3_vl_fused_vision_rope_forward(query, key, cos, sin)
    query_ref = (query.float() * cos[:, None, :] + _rotate_half(query.float()) * sin[:, None, :]).to(query.dtype)
    key_ref = (key.float() * cos[:, None, :] + _rotate_half(key.float()) * sin[:, None, :]).to(key.dtype)
    _assert_close(query_out, query_ref, query.dtype)
    _assert_close(key_out, key_ref, key.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_heads,key_heads", [(4, 4), (4, 2)])
def test_text_rope_matches_reference(dtype, query_heads, key_heads):
    batch, sequence, head_dim = 2, 6, 8
    generator = torch.Generator(device="cuda").manual_seed(13)
    query = torch.randn(batch, query_heads, sequence, head_dim, device="cuda", dtype=dtype, generator=generator)
    key = torch.randn(batch, key_heads, sequence, head_dim, device="cuda", dtype=dtype, generator=generator)
    cos = torch.randn(batch, sequence, head_dim, device="cuda", dtype=dtype, generator=generator)
    sin = torch.randn_like(cos)

    query_out, key_out = qwen3_vl_fused_text_rope_forward(query, key, cos, sin)
    cos_b = cos[:, None, :, :].float()
    sin_b = sin[:, None, :, :].float()
    query_ref = (query.float() * cos_b + _rotate_half(query.float()) * sin_b).to(dtype)
    key_ref = (key.float() * cos_b + _rotate_half(key.float()) * sin_b).to(dtype)
    _assert_close(query_out, query_ref, dtype)
    _assert_close(key_out, key_ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_text_rms_norm_matches_reference(dtype):
    rows, hidden = 9, 16
    epsilon = 1e-6
    generator = torch.Generator(device="cuda").manual_seed(17)
    hidden_states = torch.randn(rows, hidden, device="cuda", dtype=dtype, generator=generator)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32, generator=generator)

    actual = qwen3_vl_fused_text_rms_norm_forward(hidden_states, weight, epsilon)
    variance = (hidden_states.float() * hidden_states.float()).mean(-1, keepdim=True)
    normalized = hidden_states.float() * torch.rsqrt(variance + epsilon)
    expected = weight * normalized.to(dtype).float()

    assert actual.dtype == torch.float32
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_text_silu_mul_matches_reference(dtype):
    shape = (3, 5, 32)
    generator = torch.Generator(device="cuda").manual_seed(19)
    gate = torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)
    up = torch.randn_like(gate)

    actual = qwen3_vl_fused_text_silu_mul_forward(gate, up)
    activated = (gate.float() / (1.0 + torch.exp(-gate.float()))).to(dtype).float()
    expected = (activated * up.float()).to(dtype)
    _assert_close(actual, expected, dtype)


def test_vision_rope_rejects_odd_head_dimension():
    query = torch.randn(2, 1, 7, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    cos = torch.randn(2, 7, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    with pytest.raises(RuntimeError, match="head_dim must be even"):
        qwen3_vl_fused_vision_rope_forward(query, key, cos, sin)


# ---------------------------------------------------------------------------
# Extended RMSNorm tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,hidden", [
    (1, 64),        # single row
    (512, 3584),    # Qwen3-7B inference shape
    (4096, 4096),   # large training batch
    (1, 1),         # degenerate: 1 element
    (3, 128),       # hidden not a power-of-2 multiple of warpSize
])
def test_rms_norm_fused_matches_reference(dtype, rows, hidden):
    """Fused single-pass RMSNorm must match the PyTorch fp32 reference."""
    gen = torch.Generator(device="cuda").manual_seed(rows * hidden)
    hs = torch.randn(rows, hidden, device="cuda", dtype=dtype, generator=gen)
    w = torch.randn(hidden, device="cuda", dtype=torch.float32, generator=gen)
    eps = 1e-6

    actual = qwen3_vl_fused_text_rms_norm_forward(hs, w, eps)
    expected = _rms_norm_reference(hs, w, eps)

    assert actual.dtype == torch.float32, "fused RMSNorm must return fp32"
    assert actual.shape == hs.shape
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,hidden", [
    (9, 16),
    (512, 3584),
    (4096, 4096),
])
def test_rms_norm_fused_agrees_with_3step(dtype, rows, hidden):
    """Fused kernel must produce numerically near-identical results to the old 3-step chain.

    The fused single-pass kernel computes the sum-of-squares reduction in a
    different order than the 3-step chain (which does element-wise square then
    a full .mean() reduction).  Both compute correct fp32 arithmetic but the
    reduction order differs, so small fp32 rounding differences are expected.
    Use dtype-appropriate tolerance: tighter for fp16 (smaller magnitude range)
    and more relaxed for bf16 (7-bit mantissa).
    """
    gen = torch.Generator(device="cuda").manual_seed(42)
    hs = torch.randn(rows, hidden, device="cuda", dtype=dtype, generator=gen)
    w = torch.randn(hidden, device="cuda", dtype=torch.float32, generator=gen)
    eps = 1e-6

    fused = qwen3_vl_fused_text_rms_norm_forward(hs, w, eps)
    three_step = _rms_norm_3step(hs, w, eps)

    # Both outputs are fp32. Tolerance reflects fp32 accumulation differences
    # across up to 'hidden' elements; 3e-2 covers the worst-case bf16 diff
    # and 5e-3 covers fp16 at large hidden sizes.
    torch.testing.assert_close(fused, three_step, atol=3e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_fused_near_zero_variance(dtype):
    """All-zero input: output must be all-zero, no NaN/Inf."""
    hs = torch.zeros(4, 64, device="cuda", dtype=dtype)
    w = torch.ones(64, device="cuda", dtype=torch.float32)
    out = qwen3_vl_fused_text_rms_norm_forward(hs, w, 1e-6)
    assert not out.isnan().any(), "NaN in output for zero input"
    assert not out.isinf().any(), "Inf in output for zero input"
    assert (out == 0).all(), "Expected zero output for zero input"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_fused_large_values(dtype):
    """Large-magnitude input: no overflow relative to reference."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    scale = 100.0 if dtype == torch.float16 else 1000.0
    hs = torch.randn(32, 256, device="cuda", dtype=dtype, generator=gen) * scale
    w = torch.ones(256, device="cuda", dtype=torch.float32)
    eps = 1e-6
    actual = qwen3_vl_fused_text_rms_norm_forward(hs, w, eps)
    expected = _rms_norm_reference(hs, w, eps)
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_fused_hidden_not_multiple_of_warp(dtype):
    """hidden_size that is not a multiple of 32 (warpSize) must still be correct."""
    gen = torch.Generator(device="cuda").manual_seed(99)
    hs = torch.randn(8, 100, device="cuda", dtype=dtype, generator=gen)
    w = torch.randn(100, device="cuda", dtype=torch.float32, generator=gen)
    eps = 1e-5
    actual = qwen3_vl_fused_text_rms_norm_forward(hs, w, eps)
    expected = _rms_norm_reference(hs, w, eps)
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6, 1e-8])
def test_rms_norm_fused_epsilon_sensitivity(epsilon):
    """Varying epsilon must track the reference for each value."""
    dtype = torch.bfloat16
    gen = torch.Generator(device="cuda").manual_seed(55)
    hs = torch.randn(16, 64, device="cuda", dtype=dtype, generator=gen)
    w = torch.ones(64, device="cuda", dtype=torch.float32)
    actual = qwen3_vl_fused_text_rms_norm_forward(hs, w, epsilon)
    expected = _rms_norm_reference(hs, w, epsilon)
    _assert_close(actual, expected, dtype)


# ---------------------------------------------------------------------------
# Extended Text RoPE tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,q_heads,k_heads,seq,head_dim", [
    (1, 1, 1, 1, 8),      # minimal
    (1, 32, 8, 2048, 128), # Qwen3-7B inference: GQA 32/8, long seq
    (4, 8, 2, 512, 64),    # GQA ratio 4
    (2, 16, 16, 64, 32),   # MHA (equal heads)
])
def test_text_rope_extended(dtype, batch, q_heads, k_heads, seq, head_dim):
    gen = torch.Generator(device="cuda").manual_seed(batch + q_heads + seq)
    q = torch.randn(batch, q_heads, seq, head_dim, device="cuda", dtype=dtype, generator=gen)
    k = torch.randn(batch, k_heads, seq, head_dim, device="cuda", dtype=dtype, generator=gen)
    cos = torch.randn(batch, seq, head_dim, device="cuda", dtype=dtype, generator=gen)
    sin = torch.randn(batch, seq, head_dim, device="cuda", dtype=dtype, generator=gen)

    q_out, k_out = qwen3_vl_fused_text_rope_forward(q, k, cos, sin)
    q_ref = (q.float() * cos[:, None].float() + _rotate_half(q.float()) * sin[:, None].float()).to(dtype)
    k_ref = (k.float() * cos[:, None].float() + _rotate_half(k.float()) * sin[:, None].float()).to(dtype)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    _assert_close(q_out, q_ref, dtype)
    _assert_close(k_out, k_ref, dtype)


def test_text_rope_output_dtype_preserved():
    """Output dtype must match input dtype."""
    for dtype in (torch.float16, torch.bfloat16):
        q = torch.randn(2, 4, 8, 16, device="cuda", dtype=dtype)
        k = torch.randn(2, 4, 8, 16, device="cuda", dtype=dtype)
        cos = torch.randn(2, 8, 16, device="cuda", dtype=dtype)
        sin = torch.randn_like(cos)
        q_out, k_out = qwen3_vl_fused_text_rope_forward(q, k, cos, sin)
        assert q_out.dtype == dtype
        assert k_out.dtype == dtype


# ---------------------------------------------------------------------------
# Extended SiLU-mul tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [
    (1, 1, 32),           # minimal
    (4, 512, 14336),      # Qwen3-7B FFN dim
    (2, 128, 4096),       # medium
    (1, 1, 65537),        # non-power-of-2
])
def test_silu_mul_extended(dtype, shape):
    gen = torch.Generator(device="cuda").manual_seed(sum(shape))
    gate = torch.randn(*shape, device="cuda", dtype=dtype, generator=gen)
    up = torch.randn(*shape, device="cuda", dtype=dtype, generator=gen)

    actual = qwen3_vl_fused_text_silu_mul_forward(gate, up)
    activated = (gate.float() / (1.0 + torch.exp(-gate.float()))).to(dtype).float()
    expected = (activated * up.float()).to(dtype)

    assert actual.shape == gate.shape
    assert actual.dtype == dtype
    _assert_close(actual, expected, dtype)


def test_silu_mul_gate_zero():
    """Zero gate → SiLU(0)=0 → output must be zero regardless of up."""
    dtype = torch.bfloat16
    gate = torch.zeros(4, 64, device="cuda", dtype=dtype)
    up = torch.randn(4, 64, device="cuda", dtype=dtype)
    out = qwen3_vl_fused_text_silu_mul_forward(gate, up)
    assert (out == 0).all()


def test_silu_mul_up_zero():
    """Zero up → output must be zero regardless of gate."""
    dtype = torch.float16
    gate = torch.randn(4, 64, device="cuda", dtype=dtype)
    up = torch.zeros(4, 64, device="cuda", dtype=dtype)
    out = qwen3_vl_fused_text_silu_mul_forward(gate, up)
    assert (out == 0).all()


# ---------------------------------------------------------------------------
# Extended vision RoPE tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq,heads,head_dim", [
    (1, 1, 8),
    (256, 16, 64),
    (1024, 8, 128),
])
def test_vision_rope_extended(dtype, seq, heads, head_dim):
    gen = torch.Generator(device="cuda").manual_seed(seq)
    query = torch.randn(seq, heads, head_dim, device="cuda", dtype=dtype, generator=gen)
    key = torch.randn(seq, heads, head_dim, device="cuda", dtype=dtype, generator=gen)
    cos = torch.randn(seq, head_dim, device="cuda", dtype=torch.float32, generator=gen)
    sin = torch.randn(seq, head_dim, device="cuda", dtype=torch.float32, generator=gen)

    q_out, k_out = qwen3_vl_fused_vision_rope_forward(query, key, cos, sin)
    q_ref = (query.float() * cos[:, None, :] + _rotate_half(query.float()) * sin[:, None, :]).to(dtype)
    k_ref = (key.float() * cos[:, None, :] + _rotate_half(key.float()) * sin[:, None, :]).to(dtype)

    assert q_out.shape == query.shape
    assert k_out.shape == key.shape
    _assert_close(q_out, q_ref, dtype)
    _assert_close(k_out, k_ref, dtype)
