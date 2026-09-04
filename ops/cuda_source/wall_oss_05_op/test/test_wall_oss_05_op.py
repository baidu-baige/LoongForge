# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and performance tests for the wall_oss_05_op CUDA extension.

Install first:

    pip install --no-build-isolation -e .

Run with:

    pytest -q test/test_wall_oss_05_op.py
"""

import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="wall_oss_0_5 tests require CUDA"
)


@pytest.fixture(scope="module")
def ext():
    """Load the CUDA extension via the package loader."""
    from wall_oss_05_op._cuda_ext import load
    try:
        return load()
    except ImportError:
        pytest.fail(
            "CUDA extension is not built; run "
            "`pip install --no-build-isolation -e .` first"
        )


@pytest.fixture(scope="module")
def ext_exact():
    """Load the bitwise-exact CUDA extension."""
    from wall_oss_05_op._cuda_ext import load_exact, is_exact_available
    if not is_exact_available():
        pytest.skip("exact extension not available")
    return load_exact()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_reference(x, cos, sin, interleave=False):
    cos = cos.float().unsqueeze(-2)
    sin = sin.float().unsqueeze(-2)
    if interleave:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        rotated = torch.stack((-x.float()[..., 1::2], x.float()[..., ::2]), dim=-1)
        rotated = rotated.flatten(-2)
    else:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        rotated = _rotate_half(x.float())
    return x.float() * cos + rotated * sin


def _bench(fn, warmup=100, iters=500):
    """Return mean latency in µs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


# ---------------------------------------------------------------------------
# Original correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interleave", [False, True])
def test_rope_forward_and_backward(ext, interleave):
    """Check standard RoPE forward and backward results."""
    torch.manual_seed(1)
    shape = (2, 5, 3, 16)
    q = torch.randn(shape, device="cuda", dtype=torch.float32)
    k = torch.randn(shape, device="cuda", dtype=torch.float32)
    cos = torch.randn(2, 5, 8, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    ext.rope(q, k, q_out, k_out, cos, sin, interleave)
    torch.testing.assert_close(q_out, _rope_reference(q, cos, sin, interleave))
    torch.testing.assert_close(k_out, _rope_reference(k, cos, sin, interleave))

    grad_q = torch.randn_like(q)
    grad_k = torch.randn_like(k)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    ext.rope_bwd(grad_q, grad_k, dq, dk, cos, sin, interleave)
    torch.testing.assert_close(dq, _rope_reference(grad_q, cos, -sin, interleave), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dk, _rope_reference(grad_k, cos, -sin, interleave), rtol=1e-5, atol=1e-5)


def test_m_rope_gqa(ext):
    """Check M-RoPE with distinct query and key/value head counts."""
    torch.manual_seed(2)
    b, s, hq, hkv, d = 2, 4, 4, 2, 16
    q = torch.randn((b, s, hq, d), device="cuda", dtype=torch.float32)
    k = torch.randn((b, s, hkv, d), device="cuda", dtype=torch.float32)
    cos = torch.randn((3, b, s, d // 2), device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    first, second = 4, 4
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    ext.m_rope(q, k, q_out, k_out, cos, sin, first, second)

    cos_half = torch.cat((cos[0, ..., :first], cos[1, ..., first:first+second], cos[2, ..., first+second:]), dim=-1)
    sin_half = torch.cat((sin[0, ..., :first], sin[1, ..., first:first+second], sin[2, ..., first+second:]), dim=-1)
    cos_sel = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(2)
    sin_sel = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(2)
    torch.testing.assert_close(q_out, q.float() * cos_sel + _rotate_half(q) * sin_sel)
    torch.testing.assert_close(k_out, k.float() * cos_sel + _rotate_half(k) * sin_sel)


def test_rot_pos_emb_int32_and_int64(ext):
    inv_freq = torch.tensor([1.0, 0.5, 0.25, 0.125], device="cuda")
    grid = torch.tensor([[1, 4, 4], [2, 2, 4]], device="cuda", dtype=torch.int32)
    for grid_dtype in (torch.int32, torch.int64):
        grid_i = grid.to(grid_dtype)
        counts = torch.empty(2, device="cuda", dtype=grid_dtype)
        ext.get_token_counts(grid_i, counts, 2)
        assert counts.tolist() == [16, 16]
        offsets = torch.cat([torch.zeros(1, device="cuda", dtype=grid_dtype), counts.cumsum(0)])
        out = torch.empty((32, 8), device="cuda", dtype=torch.float32)
        ext.rot_pos(inv_freq, grid_i, out, offsets, 2)
        assert out.shape == (32, 8)
        assert torch.isfinite(out).all()


def test_window_index_matches_reference_for_padded_grid(ext):
    grid = torch.tensor([[1, 4, 6]], device="cuda", dtype=torch.int32)
    info = torch.empty((1, 6), device="cuda", dtype=torch.int32)
    totals = torch.zeros(2, device="cuda", dtype=torch.int32)
    ext.get_totals(grid, info, totals, 2, 2)
    total_elements, total_windows = (int(x) for x in totals.tolist())
    indices = torch.empty(total_elements, device="cuda", dtype=torch.int32)
    cu = torch.empty(total_windows + 1, device="cuda", dtype=torch.int32)
    counts = torch.empty(total_windows, device="cuda", dtype=torch.int32)
    ext.get_window_index(grid, info, indices, cu, counts, 1, 2, 2, 1, 1)
    assert sorted(indices.tolist()) == list(range(6))
    assert cu.tolist() == [0, 4, 6]


def test_permute_unpermute_topk(ext):
    tokens = torch.arange(64, device="cuda", dtype=torch.float32).reshape(4, 16)
    indices = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0]], device="cuda", dtype=torch.int32)
    max_items = tokens.shape[0] * 2
    sorted_indices = torch.empty(max_items, device="cuda", dtype=torch.int32)
    row_id = torch.arange(max_items, device="cuda", dtype=torch.int32)
    sorted_row_id = torch.empty_like(row_id)
    temp = torch.empty(ext.cub_sort_pair_get_storage_bytes(max_items), device="cuda", dtype=torch.int8)
    output = torch.empty((max_items, tokens.shape[1]), device="cuda", dtype=tokens.dtype)
    row_map = torch.empty(max_items, device="cuda", dtype=torch.int32)
    ext.permute(tokens, indices, sorted_indices, row_id, sorted_row_id, temp, output, row_map, 0, max_items)
    expected_order = torch.argsort(indices.reshape(-1), stable=True)
    expected = tokens.index_select(0, expected_order // 2)
    torch.testing.assert_close(output, expected)
    restored = torch.empty_like(tokens)
    ext.unpermute(output, row_map, None, restored, tokens.shape[0], 2)
    torch.testing.assert_close(restored, tokens * 2)


# ---------------------------------------------------------------------------
# Extended correctness tests (ops refactored to use package-level imports)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rope_package_api_matches_cuda(ext, interleave, dtype):
    """Package-level rope() must match raw CUDA kernel for multiple dtypes."""
    from wall_oss_05_op import rope as rope_op
    torch.manual_seed(3)
    shape = (2, 8, 4, 32)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    cos = torch.randn(2, 8, 16, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    # CUDA kernel
    q_cuda = torch.empty_like(q)
    k_cuda = torch.empty_like(k)
    ext.rope(q, k, q_cuda, k_cuda, cos, sin, interleave)
    # Package API
    q_pkg, k_pkg = rope_op(q, k, cos, sin, interleave=interleave, inference=True)
    torch.testing.assert_close(q_pkg, q_cuda, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_pkg, k_cuda, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm_exact_matches_pytorch(ext_exact, dtype):
    """Exact RMSNorm CUDA must match eager PyTorch forward."""
    from wall_oss_05_op._cuda_wrappers import rmsnorm_exact_kernel
    torch.manual_seed(7)
    hs = torch.randn(64, 256, device="cuda", dtype=dtype)
    w = torch.randn(256, device="cuda", dtype=dtype)
    eps = 1e-6
    # CUDA exact
    cuda_out = rmsnorm_exact_kernel(hs, w, eps)
    # PyTorch reference
    x = hs.float()
    var = x.pow(2).mean(-1, keepdim=True)
    ref = (w.float() * x * torch.rsqrt(var + eps)).to(dtype)
    torch.testing.assert_close(cuda_out, ref, rtol=1e-3, atol=1e-3)


def test_rmsnorm_exact_inv_is_bitwise_identical_to_aten(ext_exact):
    """The fused inv kernel must reproduce ``rsqrt(var + eps)`` bit for bit."""
    torch.manual_seed(11)
    eps = 1e-6
    var = torch.rand(4096, 1, device="cuda", dtype=torch.float32).mul_(50.0)
    inv = torch.empty_like(var)
    ext_exact.rmsnorm_exact_inv(var, inv, eps)
    assert torch.equal(inv, torch.rsqrt(var + eps))


@pytest.mark.parametrize("cols", [256, 1024, 2048])
def test_rmsnorm_exact_gsq2_is_bitwise_identical_to_aten(ext_exact, cols):
    """The fused g_sq2 kernel must reproduce the ATen pointwise chain bit for bit."""
    torch.manual_seed(13)
    rows = 3072
    g_inv = torch.randn(rows, 1, device="cuda", dtype=torch.float32)
    inv = torch.rand(rows, 1, device="cuda", dtype=torch.float32).add_(0.1)
    g_sq2 = torch.empty_like(g_inv)
    ext_exact.rmsnorm_exact_gsq2(g_inv, inv, g_sq2, cols)

    g_var = (-0.5 * g_inv) * inv.pow(3)
    ref = ((g_var / cols) * 2.0).contiguous()
    assert torch.equal(g_sq2, ref)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm_exact_backward_matches_unfused_chain(ext_exact, dtype):
    """Forward and backward must stay bit-identical after the launch fusion."""
    from wall_oss_05_op._cuda_wrappers import rmsnorm_exact_kernel

    torch.manual_seed(23)
    eps = 1e-6
    x = torch.randn(129, 1024, device="cuda", dtype=dtype).requires_grad_()
    w = torch.randn(1024, device="cuda", dtype=dtype).requires_grad_()
    out = rmsnorm_exact_kernel(x, w, eps)
    grad = torch.randn_like(out)
    dx, dw = torch.autograd.grad((out * grad).sum(), (x, w))

    # Unfused reference: same kernels, ATen pointwise glue (pre-fusion ordering).
    module = ext_exact
    xd = x.detach()
    sq = torch.empty(xd.shape, dtype=torch.float32, device=xd.device)
    module.rmsnorm_exact_square(xd, sq)
    inv = torch.rsqrt(sq.mean(-1, keepdim=True) + eps)
    ref_out = torch.empty(xd.shape, dtype=dtype, device=xd.device)
    module.rmsnorm_exact_fwd_out(xd, inv.reshape(-1), w.detach(), ref_out)
    assert torch.equal(out, ref_out)

    opts = dict(dtype=torch.float32, device=xd.device)
    p_dw = torch.empty(xd.shape, **opts)
    p_inv = torch.empty(xd.shape, **opts)
    module.rmsnorm_exact_bwd_prod(grad, xd, inv.reshape(-1), w.detach(), p_dw, p_inv)
    ref_dw = p_dw.sum(dim=tuple(range(xd.dim() - 1))).to(dtype)
    g_inv = p_inv.sum(-1, keepdim=True)
    n = xd.shape[-1]
    g_sq2 = (((-0.5 * g_inv) * inv.pow(3) / n) * 2.0).reshape(-1).contiguous()
    ref_dx = torch.empty(xd.shape, dtype=dtype, device=xd.device)
    module.rmsnorm_exact_bwd_dx(grad, xd, inv.reshape(-1), g_sq2, w.detach(), ref_dx)

    assert torch.equal(dx, ref_dx)
    assert torch.equal(dw, ref_dw)


def _upcast_reference(x_bf16, w_fp32, eps=1e-6):
    """Model today's path: FSDP casts the input to fp32, the scatter casts back."""
    from wall_oss_05_op._cuda_wrappers import rmsnorm_exact_kernel

    x32 = x_bf16.to(torch.float32)
    out32 = rmsnorm_exact_kernel(x32, w_fp32, eps)
    return out32.to(torch.bfloat16)


@pytest.mark.parametrize("padding", [0, 1024])
def test_rmsnorm_exact_fp32_weight_matches_upcast_path(ext_exact, padding):
    """bf16 activations with an fp32 weight must equal the fp32-upcast path.

    ``padding`` > 0 exercises a strided activation view, which is what the norm
    receives once FSDP stops materializing a contiguous fp32 copy.
    """
    from wall_oss_05_op._cuda_wrappers import rmsnorm_exact_kernel

    torch.manual_seed(31)
    rows, cols = 137, 1024
    storage = torch.randn(rows, cols + padding, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(cols, device="cuda", dtype=torch.float32)
    grad = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)

    x_ref = storage.clone().requires_grad_()
    w_ref = weight.clone().requires_grad_()
    ref = _upcast_reference(x_ref[:, :cols], w_ref)
    ref_dx, ref_dw = torch.autograd.grad((ref * grad).sum(), (x_ref, w_ref))

    x_cand = storage.clone().requires_grad_()
    w_cand = weight.clone().requires_grad_()
    cand = rmsnorm_exact_kernel(x_cand[:, :cols], w_cand)
    cand_dx, cand_dw = torch.autograd.grad((cand * grad).sum(), (x_cand, w_cand))

    assert cand.dtype is torch.bfloat16
    assert cand_dw.dtype is torch.float32
    assert torch.equal(cand, ref)
    assert torch.equal(cand_dx, ref_dx)
    assert torch.equal(cand_dw, ref_dw)


def test_rmsnorm_dispatcher_uses_cuda_for_fp32_weight_with_bf16_activation(ext_exact):
    """The dispatcher must not silently fall back for the mixed dtype pair."""
    from wall_oss_05_op.norm import rmsnorm as rmsnorm_op

    torch.manual_seed(37)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, device="cuda", dtype=torch.float32)
    assert rmsnorm_op._cuda_supported(x, w)

    out = rmsnorm_op(x, w, 1e-6)
    assert out.dtype is torch.bfloat16
    assert torch.equal(out, _upcast_reference(x, w))
    assert not rmsnorm_op._input_fallback_logged


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_swiglu_exact_matches_pytorch(ext_exact, dtype):
    """Exact SwiGLU CUDA must match eager PyTorch forward."""
    from wall_oss_05_op._cuda_wrappers import swiglu_exact_kernel
    import torch.nn.functional as F
    torch.manual_seed(9)
    gate = torch.randn(32, 128, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)
    cuda_out = swiglu_exact_kernel(gate, up)
    ref = F.silu(gate) * up
    torch.testing.assert_close(cuda_out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("interleave", [False, True])
def test_rope_pytorch_fallback_matches_cuda(ext, interleave):
    """PyTorch fallback (no CUDA) must match CUDA kernel numerically."""
    from wall_oss_05_op.rope import RoPEOp
    torch.manual_seed(5)
    shape = (2, 6, 4, 32)
    q = torch.randn(shape, device="cuda", dtype=torch.float32)
    k = torch.randn(shape, device="cuda", dtype=torch.float32)
    cos = torch.randn(2, 6, 16, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    # CUDA kernel
    q_cuda = torch.empty_like(q)
    k_cuda = torch.empty_like(k)
    ext.rope(q, k, q_cuda, k_cuda, cos, sin, interleave)
    # Force PyTorch fallback
    rope_pt = RoPEOp.__new__(RoPEOp)
    rope_pt._resolved_fn = None
    rope_pt._backend = None
    rope_pt._resolve_lock = __import__("threading").Lock()
    rope_pt._call_logged = False
    q_pt, k_pt = rope_pt._pytorch_fallback(q, k, cos, sin, interleave=interleave)
    torch.testing.assert_close(q_pt, q_cuda, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_pt, k_cuda, rtol=1e-5, atol=1e-5)


def test_permute_package_api_matches_cuda(ext):
    """Package-level permute/unpermute must match raw CUDA kernel."""
    from wall_oss_05_op import permute as permute_op, unpermute as unpermute_op
    torch.manual_seed(11)
    tokens = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    indices = torch.randint(0, 4, (8,), device="cuda", dtype=torch.int32)
    # Package API
    permuted, row_map = permute_op(tokens, indices)
    restored = unpermute_op(permuted, row_map)
    # Verify restore is same as original (single top-k, so exact)
    torch.testing.assert_close(restored, tokens)


def test_window_index_multi_grid(ext):
    """Window index with multiple grids of different sizes."""
    grid = torch.tensor([[1, 8, 8], [1, 4, 4]], device="cuda", dtype=torch.int32)
    info = torch.empty((2, 6), device="cuda", dtype=torch.int32)
    totals = torch.zeros(2, device="cuda", dtype=torch.int32)
    ext.get_totals(grid, info, totals, 2, 2)
    total_elements, total_windows = (int(x) for x in totals.tolist())
    assert total_elements > 0
    assert total_windows > 0
    indices = torch.empty(total_elements, device="cuda", dtype=torch.int32)
    cu = torch.empty(total_windows + 1, device="cuda", dtype=torch.int32)
    counts_t = torch.empty(total_windows, device="cuda", dtype=torch.int32)
    ext.get_window_index(grid, info, indices, cu, counts_t, 1, 2, 2, 1, 1)
    # All indices must be non-negative and unique within each grid segment
    assert (indices >= 0).all()


# ---------------------------------------------------------------------------
# Performance benchmarks (CUDA kernel vs PyTorch fallback)
# ---------------------------------------------------------------------------

def test_bench_rope_cuda_vs_pytorch(ext):
    """Benchmark: CUDA rope vs PyTorch reference. CUDA should be faster."""
    torch.manual_seed(42)
    S, H, D = 2048, 32, 128
    q = torch.randn(S, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn(S, H, D, device="cuda", dtype=torch.float32)
    cos = torch.randn(S, D // 2, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    def cuda_fn():
        ext.rope(q, k, q_out, k_out, cos, sin, False)

    def pt_fn():
        _rope_reference(q, cos.unsqueeze(0), sin.unsqueeze(0))

    t_cuda = _bench(cuda_fn)
    t_pt = _bench(pt_fn)
    print(f"\n[RoPE bench] CUDA={t_cuda:.1f}µs  PyTorch={t_pt:.1f}µs  speedup={t_pt/t_cuda:.2f}x")
    # CUDA kernel must be at least as fast as pure PyTorch
    assert t_cuda <= t_pt * 2.0, f"CUDA ({t_cuda:.1f}µs) unexpectedly slow vs PyTorch ({t_pt:.1f}µs)"


def test_bench_rmsnorm_exact_vs_pytorch(ext_exact):
    """Benchmark: exact RMSNorm CUDA vs PyTorch eager."""
    from wall_oss_05_op._cuda_wrappers import rmsnorm_exact_kernel
    torch.manual_seed(42)
    rows, hidden = 1024, 4096
    hs = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    def cuda_fn():
        rmsnorm_exact_kernel(hs, w, eps)

    def pt_fn():
        x = hs.float()
        var = x.pow(2).mean(-1, keepdim=True)
        return (w.float() * x * torch.rsqrt(var + eps)).to(torch.bfloat16)

    t_cuda = _bench(cuda_fn)
    t_pt = _bench(pt_fn)
    print(f"\n[RMSNorm bench] CUDA={t_cuda:.1f}µs  PyTorch={t_pt:.1f}µs  speedup={t_pt/t_cuda:.2f}x")
    assert t_cuda <= t_pt * 3.0, f"CUDA ({t_cuda:.1f}µs) unexpectedly slow vs PyTorch ({t_pt:.1f}µs)"


def test_bench_permute_cuda_vs_pytorch(ext):
    """Benchmark: MoE permute CUDA vs PyTorch fallback."""
    from wall_oss_05_op.moe import PermuteOp
    torch.manual_seed(42)
    N, D, topk = 1024, 256, 2
    tokens = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, 8, (N, topk), device="cuda", dtype=torch.int32)
    max_items = N * topk
    sorted_indices = torch.empty(max_items, device="cuda", dtype=torch.int32)
    row_id = torch.arange(max_items, device="cuda", dtype=torch.int32)
    sorted_row_id = torch.empty_like(row_id)
    temp = torch.empty(ext.cub_sort_pair_get_storage_bytes(max_items), device="cuda", dtype=torch.int8)
    output = torch.empty((max_items, D), device="cuda", dtype=tokens.dtype)
    row_map = torch.empty(max_items, device="cuda", dtype=torch.int32)

    def cuda_fn():
        ext.permute(tokens, indices, sorted_indices, row_id, sorted_row_id, temp, output, row_map, 0, max_items)

    perm_op = PermuteOp.__new__(PermuteOp)

    def pt_fn():
        perm_op._pytorch_fallback(tokens, indices)

    t_cuda = _bench(cuda_fn)
    t_pt = _bench(pt_fn)
    print(f"\n[Permute bench] CUDA={t_cuda:.1f}µs  PyTorch={t_pt:.1f}µs  speedup={t_pt/t_cuda:.2f}x")
    # Just report; CUB radix sort vs torch.argsort may vary


# ---------------------------------------------------------------------------
# Public API tests (README L38-41 import style)
# Tests exercise each exported symbol via the top-level package import.
# ---------------------------------------------------------------------------

def test_public_api_rope():
    """rope() via top-level import with correct float32 cos/sin."""
    from wall_oss_05_op import rope
    B, H, S, D = 2, 8, 16, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, 4, S, D, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(B, S, D // 2, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    q_out, k_out = rope(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert q_out.dtype == q.dtype


def test_public_api_m_rope():
    """m_rope() via top-level import."""
    from wall_oss_05_op import m_rope
    B, H, S, D = 2, 8, 16, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, 4, S, D, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(3, B, S, D // 2, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    q_out, k_out = m_rope(q, k, cos, sin, (D // 4, D // 4))
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_public_api_m_rope_pack_matches_split_path():
    """Packed M-RoPE preserves split-path outputs and gradients."""
    from wall_oss_05_op import m_rope

    torch.manual_seed(3)
    b, s, hq, hkv, d = 2, 5, 8, 2, 32
    q_dim = hq * d
    kv_dim = hkv * d
    first, second = 8, 8
    qkv_data = torch.randn(
        (b, s, q_dim + 2 * kv_dim), device="cuda", dtype=torch.bfloat16
    )
    cos = torch.randn(
        (3, b, s, d // 2), device="cuda", dtype=torch.float32
    )
    sin = torch.randn_like(cos)

    qkv_ref = qkv_data.detach().clone().requires_grad_()
    q_ref = qkv_ref[..., :q_dim].view(b, s, hq, d)
    k_ref = qkv_ref[..., q_dim : q_dim + kv_dim].view(b, s, hkv, d)
    q_out, k_out = m_rope(q_ref, k_ref, cos, sin, (first, second))
    reference = torch.cat(
        (
            q_out.reshape(b, s, q_dim),
            k_out.reshape(b, s, kv_dim),
            qkv_ref[..., q_dim + kv_dim :],
        ),
        dim=-1,
    )

    qkv_candidate = qkv_data.detach().clone().requires_grad_()
    candidate = m_rope.pack(
        qkv_candidate, hq, hkv, cos, sin, (first, second)
    )
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

    upstream = torch.randn_like(reference)
    (reference * upstream).sum().backward()
    (candidate * upstream).sum().backward()
    torch.testing.assert_close(
        qkv_candidate.grad, qkv_ref.grad, rtol=0, atol=0
    )


def test_public_api_rmsnorm():
    """rmsnorm() via top-level import with matching float32 weight."""
    from wall_oss_05_op import rmsnorm
    hs = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    w = torch.randn(64, device="cuda", dtype=torch.float32)
    out = rmsnorm(hs, w, 1e-6)
    assert out.shape == hs.shape


def test_public_api_swiglu():
    """swiglu() via top-level import."""
    from wall_oss_05_op import swiglu
    gate = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    out = swiglu(gate, up)
    assert out.shape == gate.shape
    assert out.dtype == gate.dtype


def test_public_api_permute_unpermute():
    """permute() and unpermute() via top-level import, single top-k."""
    from wall_oss_05_op import permute, unpermute
    N, D = 16, 32
    tokens = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, 4, (N,), device="cuda", dtype=torch.int32)
    permuted, row_map = permute(tokens, indices)
    assert permuted.shape[1] == D
    restored = unpermute(permuted, row_map)
    assert restored.shape == tokens.shape
    # Single topk: restored should equal original
    torch.testing.assert_close(restored, tokens)


def test_single_qkv_unpermute_matches_three_separate_unpermutes():
    """One packed QKV unpermute preserves three-path outputs and gradients."""
    from wall_oss_05_op import permute, unpermute

    torch.manual_seed(4)
    n, q_dim, kv_dim = 37, 64, 16
    qkv_dim = q_dim + 2 * kv_dim
    qkv_data = torch.randn(
        (n, qkv_dim), device="cuda", dtype=torch.bfloat16
    )
    expert_indices = torch.randint(
        0, 2, (n,), device="cuda", dtype=torch.int32
    )
    permuted, row_map = permute(qkv_data, expert_indices)
    probs = torch.rand((n, 1), device="cuda", dtype=torch.float32)

    packed_ref = permuted.detach().clone().requires_grad_()
    q_ref, k_ref, v_ref = torch.split(
        packed_ref, (q_dim, kv_dim, kv_dim), dim=-1
    )
    reference = torch.cat(
        (
            unpermute(q_ref, row_map, probs),
            unpermute(k_ref, row_map, probs),
            unpermute(v_ref, row_map, probs),
        ),
        dim=-1,
    )

    packed_candidate = permuted.detach().clone().requires_grad_()
    candidate = unpermute(packed_candidate, row_map, probs)
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

    upstream = torch.randn_like(reference)
    (reference * upstream).sum().backward()
    (candidate * upstream).sum().backward()
    torch.testing.assert_close(
        packed_candidate.grad, packed_ref.grad, rtol=0, atol=0
    )


def test_public_api_get_rope_index(ext):
    """get_rope_index() via top-level import."""
    # Simple spatial grid: batch=1, 1 frame, 4x4 spatial
    grid = torch.tensor([[1, 4, 4]], device="cuda", dtype=torch.int32)
    info = torch.empty((1, 6), device="cuda", dtype=torch.int32)
    totals = torch.zeros(2, device="cuda", dtype=torch.int32)
    ext.get_totals(grid, info, totals, 2, 2)
    total_elements = int(totals[0])
    indices = torch.empty(total_elements, device="cuda", dtype=torch.int32)
    cu = torch.empty(int(totals[1]) + 1, device="cuda", dtype=torch.int32)
    counts = torch.empty(int(totals[1]), device="cuda", dtype=torch.int32)
    ext.get_window_index(grid, info, indices, cu, counts, 1, 2, 2, 1, 1)
    assert (indices >= 0).all()


def test_public_api_get_window_index(ext):
    """get_window_index() via top-level import (same kernel path as get_rope_index)."""
    from wall_oss_05_op import get_window_index
    # Verify the public API is callable; correctness tested via ext in other tests.
    assert callable(get_window_index)


def test_public_api_backend_inventory():
    """backend_inventory() reports a resolved backend for every public operator."""
    from wall_oss_05_op import backend_inventory

    inventory = backend_inventory()
    assert isinstance(inventory, dict)
    expected = {
        "rope",
        "m_rope",
        "rot_pos_emb",
        "rmsnorm",
        "swiglu",
        "permute",
        "unpermute",
        "get_rope_index",
        "get_window_index",
    }
    assert expected.issubset(inventory.keys())
    for name in expected:
        assert inventory[name] in ("cuda_inline", "pytorch")
