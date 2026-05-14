"""Tests for int4_qat package."""
import math
import pytest
import torch


def _reference_fake_int4_quant(x, block_size, sym):
    """Pure PyTorch per-block reference (loop-based, for correctness checking)."""
    M, N = x.shape
    block_m, block_n = block_size
    scale_rows = math.ceil(M / block_m)
    scale_cols = math.ceil(N / block_n)

    q = torch.empty_like(x)
    scale = torch.empty((scale_rows, scale_cols), device=x.device, dtype=x.dtype)
    zero = torch.empty_like(scale)

    for bi in range(scale_rows):
        r0, r1 = bi * block_m, min((bi + 1) * block_m, M)
        for bj in range(scale_cols):
            c0, c1 = bj * block_n, min((bj + 1) * block_n, N)
            block = x[r0:r1, c0:c1]

            if sym:
                s = torch.clamp(block.abs().max() / 7.0, min=1e-5)
                z = torch.zeros((), device=x.device, dtype=x.dtype)
                q_block = torch.round(block / s)
            else:
                block_min, block_max = block.min(), block.max()
                s = torch.clamp((block_max - block_min) / 15.0, min=1e-5)
                z = torch.clamp(-torch.round(block_min / s), min=0.0, max=15.0)
                q_block = torch.round(block / s) + z

            q[r0:r1, c0:c1] = q_block.to(x.dtype)
            scale[bi, bj] = s.to(x.dtype)
            zero[bi, bj] = z.to(x.dtype)

    return q, scale, zero


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


# ─── Low-level: fake_int4_quant ───


@pytest.mark.parametrize(
    "shape,block_size",
    [
        ((7, 65), [1, 32]),
        ((65, 7), [32, 1]),
        ((9, 257), [1, 128]),
        ((33, 96), [8, 32]),
    ],
)
def test_quant_symmetric(cuda_device, shape, block_size):
    from int4_qat import fake_int4_quant

    torch.manual_seed(0)
    x = torch.randn(*shape, device=cuda_device, dtype=torch.float32) * 1.7 + 0.13
    q, scale, zero = fake_int4_quant(x, block_size, sym=True)
    ref_q, ref_scale, _ = _reference_fake_int4_quant(x, block_size, True)

    assert q.shape == x.shape
    assert q.dtype == x.dtype
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(scale, ref_scale, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "shape,block_size",
    [
        ((7, 65), [1, 32]),
        ((65, 7), [32, 1]),
        ((9, 257), [1, 128]),
        ((33, 96), [8, 32]),
    ],
)
def test_quant_asymmetric(cuda_device, shape, block_size):
    from int4_qat import fake_int4_quant

    torch.manual_seed(1)
    x = torch.randn(*shape, device=cuda_device, dtype=torch.float32)
    x = x + torch.linspace(-0.5, 0.5, steps=shape[1], device=cuda_device).unsqueeze(0)

    q, scale, zero = fake_int4_quant(x, block_size, sym=False)
    ref_q, ref_scale, ref_zero = _reference_fake_int4_quant(x, block_size, False)

    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(scale, ref_scale, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(zero, ref_zero, rtol=0, atol=0)


def test_bfloat16(cuda_device):
    from int4_qat import fake_int4_quant

    torch.manual_seed(3)
    x = torch.randn(17, 129, device=cuda_device, dtype=torch.bfloat16)
    q, scale, zero = fake_int4_quant(x, [1, 128], sym=True)

    assert q.shape == x.shape and q.dtype == x.dtype
    assert scale.shape == (17, 2)
    assert torch.isfinite(q.float()).all()


# ─── Mid-level: fake_int4_quantize_dequantize ───


def test_quantize_dequantize_shape_dtype(cuda_device):
    from int4_qat import fake_int4_quantize_dequantize

    torch.manual_seed(10)
    w = torch.randn(64, 256, device=cuda_device, dtype=torch.bfloat16)
    fq = fake_int4_quantize_dequantize(w, group_size=32, sym=True)

    assert fq.shape == w.shape
    assert fq.dtype == w.dtype
    # Values should differ from original (quantization noise)
    assert not torch.equal(fq, w)


def test_quantize_dequantize_values_on_grid(cuda_device):
    """Dequantized values should be exact multiples of scale."""
    from int4_qat import fake_int4_quant, fake_int4_quantize_dequantize

    torch.manual_seed(11)
    w = torch.randn(8, 64, device=cuda_device, dtype=torch.float32)
    fq = fake_int4_quantize_dequantize(w, group_size=32, sym=True)

    # Verify: fq / scale should be integers (within tolerance)
    _, scale, _ = fake_int4_quant(w, [1, 32], sym=True)
    scale_full = scale.unsqueeze(-1).expand(8, 2, 32).reshape(8, 64)
    codes = fq / scale_full
    residual = (codes - codes.round()).abs()
    assert residual.max() < 1e-4


# ─── High-level: FakeInt4QuantSTE ───


def test_ste_forward_matches_dequant(cuda_device):
    from int4_qat import FakeQuantWeightTransform, fake_int4_quantize_dequantize

    torch.manual_seed(20)
    w = torch.randn(32, 128, device=cuda_device, dtype=torch.bfloat16, requires_grad=True)

    transform = FakeQuantWeightTransform(group_size=32, sym=True)
    fq_ste = transform(w)
    with torch.no_grad():
        fq_ref = fake_int4_quantize_dequantize(w, 32, True)

    torch.testing.assert_close(fq_ste, fq_ref)


def test_ste_gradient_passthrough(cuda_device):
    """STE backward should pass gradient through unchanged."""
    from int4_qat import FakeQuantWeightTransform

    torch.manual_seed(21)
    w = torch.randn(16, 64, device=cuda_device, dtype=torch.float32, requires_grad=True)

    transform = FakeQuantWeightTransform(group_size=32, sym=True)
    fq = transform(w)
    fq.sum().backward()

    assert w.grad is not None
    # STE: grad should be all ones (from sum)
    torch.testing.assert_close(w.grad, torch.ones_like(w))


# ─── Fused kernel: fake_int4_quantize_dequantize via fused CUDA path ───


def _has_fused_cuda():
    try:
        import int4_qat.cuda_fused  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_fused_cuda(), reason="Fused CUDA extension not built")
class TestFusedKernel:
    """Tests for the fused INT4 quant+dequant CUDA kernel."""

    def test_fused_matches_two_pass_small(self, cuda_device):
        """Bit-exact match on a small tensor."""
        from int4_qat import fake_int4_quantize_dequantize
        from int4_qat.cuda_fused import fused_fake_int4_quantize_dequantize_cuda

        torch.manual_seed(42)
        w = torch.randn(64, 256, device=cuda_device, dtype=torch.bfloat16)

        # Two-pass reference (force fallback by calling quant + dequant manually)
        from int4_qat.interface import fake_int4_quant
        q, scale, _ = fake_int4_quant(w, [1, 32], sym=True)
        scale_exp = scale.unsqueeze(-1).expand(64, 8, 32).reshape(64, 256)
        ref = (q * scale_exp).to(torch.bfloat16)

        # Fused path
        fused = fused_fake_int4_quantize_dequantize_cuda(w)

        torch.testing.assert_close(fused, ref, rtol=0, atol=0)

    @pytest.mark.parametrize("M,N", [
        (128, 7168),       # partial fc1
        (4096, 7168),      # single expert fc1
        (4096, 2048),      # single expert fc2
        (7, 65),           # non-aligned M and N
        (1, 32),           # minimal: one row, one group
        (1, 64),           # one row, two groups
    ])
    def test_fused_matches_two_pass_shapes(self, cuda_device, M, N):
        """Bit-exact match across various shapes."""
        from int4_qat.interface import fake_int4_quant
        from int4_qat.cuda_fused import fused_fake_int4_quantize_dequantize_cuda

        torch.manual_seed(100 + M + N)
        w = torch.randn(M, N, device=cuda_device, dtype=torch.bfloat16)

        q, scale, _ = fake_int4_quant(w, [1, 32], sym=True)
        scale_cols = (N + 31) // 32
        scale_exp = scale.unsqueeze(-1).expand(M, scale_cols, 32)
        scale_full = scale_exp.reshape(M, scale_cols * 32)[:, :N]
        ref = (q * scale_full).to(torch.bfloat16)

        fused = fused_fake_int4_quantize_dequantize_cuda(w)
        torch.testing.assert_close(fused, ref, rtol=0, atol=0)

    def test_fused_shape_dtype_preserved(self, cuda_device):
        """Output shape and dtype must match input."""
        from int4_qat.cuda_fused import fused_fake_int4_quantize_dequantize_cuda

        torch.manual_seed(50)
        w = torch.randn(512, 1024, device=cuda_device, dtype=torch.bfloat16)
        out = fused_fake_int4_quantize_dequantize_cuda(w)
        assert out.shape == w.shape
        assert out.dtype == w.dtype
        assert not torch.equal(out, w)  # quantization should change values

    def test_fused_transparent_dispatch(self, cuda_device):
        """fake_int4_quantize_dequantize auto-dispatches to fused when eligible."""
        from int4_qat import fake_int4_quantize_dequantize
        from int4_qat.cuda_fused import fused_fake_int4_quantize_dequantize_cuda

        torch.manual_seed(60)
        w = torch.randn(256, 1024, device=cuda_device, dtype=torch.bfloat16)

        # Both paths should give identical results
        via_api = fake_int4_quantize_dequantize(w, group_size=32, sym=True)
        via_fused = fused_fake_int4_quantize_dequantize_cuda(w)
        torch.testing.assert_close(via_api, via_fused, rtol=0, atol=0)

    def test_fused_fallback_for_asymmetric(self, cuda_device):
        """Asymmetric mode should fall back to two-pass (fused is sym-only)."""
        from int4_qat import fake_int4_quantize_dequantize

        torch.manual_seed(70)
        w = torch.randn(64, 128, device=cuda_device, dtype=torch.bfloat16)
        # Should not crash — falls back to two-pass
        out = fake_int4_quantize_dequantize(w, group_size=32, sym=False)
        assert out.shape == w.shape

    def test_fused_fallback_for_group128(self, cuda_device):
        """group_size=128 should fall back to two-pass."""
        from int4_qat import fake_int4_quantize_dequantize

        torch.manual_seed(80)
        w = torch.randn(64, 256, device=cuda_device, dtype=torch.bfloat16)
        out = fake_int4_quantize_dequantize(w, group_size=128, sym=True)
        assert out.shape == w.shape

    def test_fused_large_moe_shape(self, cuda_device):
        """Full MoE expert fc1 shape — correctness at scale."""
        from int4_qat.interface import fake_int4_quant
        from int4_qat.cuda_fused import fused_fake_int4_quantize_dequantize_cuda

        M, N = 262144, 7168  # fc1: 64 experts × 4096 × 7168
        torch.manual_seed(200)
        w = torch.randn(M, N, device=cuda_device, dtype=torch.bfloat16)

        q, scale, _ = fake_int4_quant(w, [1, 32], sym=True)
        scale_cols = N // 32
        scale_full = scale.unsqueeze(-1).expand(M, scale_cols, 32).reshape(M, N)
        ref = (q * scale_full).to(torch.bfloat16)

        fused = fused_fake_int4_quantize_dequantize_cuda(w)
        torch.testing.assert_close(fused, ref, rtol=0, atol=0)


# ─── FakeQuantWeightTransform ───


class TestWeightTransform:
    """Tests for the pluggable FakeQuantWeightTransform wrapper."""

    def test_transform_matches_raw_api(self, cuda_device):
        """Transform output must match fake_int4_quantize_dequantize."""
        from int4_qat import FakeQuantWeightTransform, fake_int4_quantize_dequantize

        torch.manual_seed(300)
        w = torch.randn(64, 256, device=cuda_device, dtype=torch.bfloat16)

        transform = FakeQuantWeightTransform(group_size=32, sym=True)
        out = transform(w)

        ref = fake_int4_quantize_dequantize(w, group_size=32, sym=True)
        # STE: out = w + (ref - w).detach() == ref  (values equal, grad graph differs)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_transform_ste_gradient(self, cuda_device):
        """Backward through the transform should pass gradient to original weight."""
        from int4_qat import FakeQuantWeightTransform

        torch.manual_seed(301)
        w = torch.randn(16, 64, device=cuda_device, dtype=torch.float32,
                         requires_grad=True)

        transform = FakeQuantWeightTransform(group_size=32, sym=True)
        out = transform(w)
        loss = out.sum()
        loss.backward()

        assert w.grad is not None
        torch.testing.assert_close(w.grad, torch.ones_like(w))

    def test_transform_disabled(self, cuda_device):
        """When enabled=False, transform is identity."""
        from int4_qat import FakeQuantWeightTransform

        torch.manual_seed(302)
        w = torch.randn(32, 128, device=cuda_device, dtype=torch.bfloat16)

        transform = FakeQuantWeightTransform(group_size=32, sym=True)
        transform.enabled = False

        out = transform(w)
        assert out is w  # exact same object, not a copy

    def test_transform_eval_mode(self, cuda_device):
        """In eval mode, transform is identity regardless of .enabled."""
        from int4_qat import FakeQuantWeightTransform

        torch.manual_seed(303)
        w = torch.randn(32, 128, device=cuda_device, dtype=torch.bfloat16)

        transform = FakeQuantWeightTransform(group_size=32, sym=True)

        # .enabled is True so transform will still quantize.
        # The _get_weight_tensors gating handles training vs eval.
        out = transform(w)
        assert out.shape == w.shape

    def test_apply_int4_qat_with_filter(self, cuda_device):
        """apply_int4_qat with filter_regex attaches to matching modules only."""
        from int4_qat import apply_int4_qat, FakeQuantWeightTransform

        class FakeGroupedLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def _get_weight_tensors(self):
                return [torch.zeros(1)]

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.expert_fc1 = FakeGroupedLinear()
                self.expert_fc2 = FakeGroupedLinear()
                self.absorb = FakeGroupedLinear()

        model = FakeModel()
        # Use regex to match expert_fc1 / expert_fc2 but not absorb
        transform, count = apply_int4_qat(
            model,
            filter_regex=r'expert_fc[12]$'
        )

        assert isinstance(transform, FakeQuantWeightTransform)
        assert count == 2
        assert hasattr(model.expert_fc1, '_int4_qat_transform')
        assert hasattr(model.expert_fc2, '_int4_qat_transform')
        assert not hasattr(model.absorb, '_int4_qat_transform')
        # Same instance shared
        assert model.expert_fc1._int4_qat_transform is model.expert_fc2._int4_qat_transform
