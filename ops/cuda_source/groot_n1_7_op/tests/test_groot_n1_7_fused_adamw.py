"""Correctness tests for the GR00T-N1.7 precision-compatible AdamW paths."""

import math

import pytest
import torch

from groot_n1_7_op.groot_fused_adamw import (
    capturable_grad_scaled_step,
    capturable_step,
    eager_step,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _reference_update(
        params, grads, exp_avgs, exp_avg_sqs, *, decay_factor, beta2,
        first_moment_weight, second_moment_weight, eps,
        bias_correction1, bias_correction2_sqrt, lr, grad_scale=1.0):
    for param, grad, exp_avg, exp_avg_sq in zip(params, grads, exp_avgs, exp_avg_sqs):
        p = param.float() * decay_factor
        g = grad.float() * grad_scale
        m = exp_avg.float() + first_moment_weight * (g - exp_avg.float())
        v = exp_avg_sq.float() * beta2 + second_moment_weight * g * g
        denominator = v.sqrt() / bias_correction2_sqrt + eps
        p = p + (-lr / bias_correction1) * (m / denominator)
        param.copy_(p)
        exp_avg.copy_(m)
        exp_avg_sq.copy_(v)


def _state(seed=23):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shapes = [(4097,), (17, 31)]
    params = [torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator) for shape in shapes]
    grads = [torch.randn_like(param) for param in params]
    exp_avgs = [torch.randn_like(param) for param in params]
    exp_avg_sqs = [torch.rand_like(param) for param in params]
    return params, grads, exp_avgs, exp_avg_sqs


def test_eager_step_matches_reference():
    params, grads, exp_avgs, exp_avg_sqs = _state()
    actual = [tensor.clone() for tensor in params]
    actual_m = [tensor.clone() for tensor in exp_avgs]
    actual_v = [tensor.clone() for tensor in exp_avg_sqs]
    expected = [tensor.clone() for tensor in params]
    expected_m = [tensor.clone() for tensor in exp_avgs]
    expected_v = [tensor.clone() for tensor in exp_avg_sqs]
    args = dict(
        decay_factor=0.999,
        beta2=0.91,
        first_moment_weight=0.13,
        second_moment_weight=0.09,
        eps=1e-5,
        bias_correction1=0.77,
        bias_correction2_sqrt=0.83,
        lr=0.002,
    )

    eager_step(actual, grads, actual_m, actual_v, **args)
    _reference_update(expected, grads, expected_m, expected_v, **args)
    for actual_tensor, expected_tensor in zip(actual + actual_m + actual_v, expected + expected_m + expected_v):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=2e-6, rtol=2e-6)


def _capturable_args():
    step = torch.tensor(2, device="cuda", dtype=torch.int64)
    bias1 = torch.tensor([0.0, 0.4, 0.7, 0.9], device="cuda", dtype=torch.float64)
    bias2 = torch.tensor([0.0, 0.6, 0.8, 0.95], device="cuda", dtype=torch.float64)
    return dict(
        lr=torch.tensor(0.003, device="cuda", dtype=torch.float64),
        step=step,
        bias_correction1=bias1,
        bias_correction2_sqrt=bias2,
        beta2=0.92,
        first_moment_weight=0.2,
        second_moment_weight=0.08,
        eps=1e-5,
        weight_decay=0.04,
    )


def test_capturable_step_matches_reference():
    params, grads, exp_avgs, exp_avg_sqs = _state(29)
    actual = [tensor.clone() for tensor in params]
    actual_m = [tensor.clone() for tensor in exp_avgs]
    actual_v = [tensor.clone() for tensor in exp_avg_sqs]
    expected = [tensor.clone() for tensor in params]
    expected_m = [tensor.clone() for tensor in exp_avgs]
    expected_v = [tensor.clone() for tensor in exp_avg_sqs]
    args = _capturable_args()

    capturable_step(actual, grads, actual_m, actual_v, **args)
    step = int(args["step"].item())
    lr = float(args["lr"].item())
    _reference_update(
        expected, grads, expected_m, expected_v,
        decay_factor=1.0 - lr * args["weight_decay"],
        beta2=args["beta2"],
        first_moment_weight=args["first_moment_weight"],
        second_moment_weight=args["second_moment_weight"],
        eps=args["eps"],
        bias_correction1=float(args["bias_correction1"][step].item()),
        bias_correction2_sqrt=float(args["bias_correction2_sqrt"][step].item()),
        lr=lr,
    )
    for actual_tensor, expected_tensor in zip(actual + actual_m + actual_v, expected + expected_m + expected_v):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=2e-6, rtol=2e-6)


def test_capturable_grad_scaled_step_matches_reference():
    params, grads, exp_avgs, exp_avg_sqs = _state(31)
    actual = [tensor.clone() for tensor in params]
    actual_m = [tensor.clone() for tensor in exp_avgs]
    actual_v = [tensor.clone() for tensor in exp_avg_sqs]
    expected = [tensor.clone() for tensor in params]
    expected_m = [tensor.clone() for tensor in exp_avgs]
    expected_v = [tensor.clone() for tensor in exp_avg_sqs]
    args = _capturable_args()
    args["grad_scale"] = torch.tensor(1.75, device="cuda", dtype=torch.float32)

    capturable_grad_scaled_step(actual, grads, actual_m, actual_v, **args)
    step = int(args["step"].item())
    lr = float(args["lr"].item())
    _reference_update(
        expected, grads, expected_m, expected_v,
        decay_factor=1.0 - lr * args["weight_decay"],
        beta2=args["beta2"],
        first_moment_weight=args["first_moment_weight"],
        second_moment_weight=args["second_moment_weight"],
        eps=args["eps"],
        bias_correction1=float(args["bias_correction1"][step].item()),
        bias_correction2_sqrt=float(args["bias_correction2_sqrt"][step].item()),
        lr=lr,
        grad_scale=float(args["grad_scale"].item()),
    )
    for actual_tensor, expected_tensor in zip(actual + actual_m + actual_v, expected + expected_m + expected_v):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=2e-6, rtol=2e-6)


def test_adamw_rejects_empty_tensor_list():
    with pytest.raises(RuntimeError, match="at least one parameter"):
        eager_step(
            [], [], [], [], decay_factor=1.0, beta2=0.9,
            first_moment_weight=0.1, second_moment_weight=0.1,
            eps=1e-8, bias_correction1=1.0,
            bias_correction2_sqrt=1.0, lr=1e-3,
        )


# ---------------------------------------------------------------------------
# Extended AdamW tests (numerical precision verification)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-8])
def test_eager_step_epsilon_sensitivity(eps):
    """rsqrtf-based denominator must match reference for various epsilon values."""
    params, grads, exp_avgs, exp_avg_sqs = _state(seed=77)
    actual = [t.clone() for t in params]
    actual_m = [t.clone() for t in exp_avgs]
    actual_v = [t.clone() for t in exp_avg_sqs]
    expected = [t.clone() for t in params]
    expected_m = [t.clone() for t in exp_avgs]
    expected_v = [t.clone() for t in exp_avg_sqs]
    args = dict(
        decay_factor=0.999, beta2=0.91, first_moment_weight=0.1,
        second_moment_weight=0.09, eps=eps,
        bias_correction1=0.9, bias_correction2_sqrt=0.95, lr=1e-3,
    )
    eager_step(actual, grads, actual_m, actual_v, **args)
    _reference_update(expected, grads, expected_m, expected_v, **args)
    for a, e in zip(actual + actual_m + actual_v, expected + expected_m + expected_v):
        torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-6)


def test_eager_step_near_zero_exp_avg_sq():
    """Very small exp_avg_sq: rsqrtf must not produce Inf (eps guards the denominator)."""
    shapes = [(128,), (64, 32)]
    gen = torch.Generator(device="cuda").manual_seed(13)
    params = [torch.randn(s, device="cuda", dtype=torch.float32, generator=gen) for s in shapes]
    grads = [torch.randn_like(p) for p in params]
    exp_avgs = [torch.zeros_like(p) for p in params]
    # exp_avg_sq very close to zero
    exp_avg_sqs = [torch.full_like(p, 1e-30) for p in params]
    args = dict(
        decay_factor=1.0, beta2=0.999, first_moment_weight=0.1,
        second_moment_weight=0.001, eps=1e-8,
        bias_correction1=1.0, bias_correction2_sqrt=1.0, lr=1e-3,
    )
    eager_step(
        [p.clone() for p in params], grads,
        [m.clone() for m in exp_avgs], [v.clone() for v in exp_avg_sqs], **args
    )
    # Just verify no exception and no NaN/Inf in result
    result = [p.clone() for p in params]
    eager_step(result, grads, exp_avgs, exp_avg_sqs, **args)
    for r in result:
        assert not r.isnan().any(), "NaN in AdamW output with near-zero exp_avg_sq"
        assert not r.isinf().any(), "Inf in AdamW output with near-zero exp_avg_sq"


def test_eager_step_large_tensors():
    """Multi-million-element tensor: rsqrtf path must remain numerically correct."""
    gen = torch.Generator(device="cuda").manual_seed(99)
    N = 1024 * 1024
    param = torch.randn(N, device="cuda", dtype=torch.float32, generator=gen)
    grad = torch.randn(N, device="cuda", dtype=torch.float32, generator=gen)
    exp_avg = torch.randn(N, device="cuda", dtype=torch.float32, generator=gen)
    exp_avg_sq = torch.rand(N, device="cuda", dtype=torch.float32, generator=gen).add_(0.01)
    args = dict(
        decay_factor=0.9999, beta2=0.999, first_moment_weight=0.1,
        second_moment_weight=0.001, eps=1e-8,
        bias_correction1=0.9, bias_correction2_sqrt=0.999, lr=1e-3,
    )
    actual_p = param.clone()
    actual_m = exp_avg.clone()
    actual_v = exp_avg_sq.clone()
    expected_p = param.clone()
    expected_m = exp_avg.clone()
    expected_v = exp_avg_sq.clone()
    eager_step([actual_p], [grad], [actual_m], [actual_v], **args)
    _reference_update([expected_p], [grad], [expected_m], [expected_v], **args)
    torch.testing.assert_close(actual_p, expected_p, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_m, expected_m, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_v, expected_v, atol=2e-6, rtol=2e-6)


def test_eager_step_multi_tensor_consistency():
    """AdamW applied to a list must match element-wise independent application."""
    gen = torch.Generator(device="cuda").manual_seed(41)
    shapes = [(7,), (3, 5), (128, 64), (4097,)]
    params = [torch.randn(s, device="cuda", dtype=torch.float32, generator=gen) for s in shapes]
    grads = [torch.randn_like(p) for p in params]
    exp_avgs = [torch.randn_like(p) for p in params]
    exp_avg_sqs = [torch.rand_like(p).add_(0.01) for p in params]
    args = dict(
        decay_factor=0.9995, beta2=0.95, first_moment_weight=0.15,
        second_moment_weight=0.05, eps=1e-6,
        bias_correction1=0.85, bias_correction2_sqrt=0.92, lr=2e-3,
    )
    # Joint multi-tensor call
    joint_p = [p.clone() for p in params]
    joint_m = [m.clone() for m in exp_avgs]
    joint_v = [v.clone() for v in exp_avg_sqs]
    eager_step(joint_p, grads, joint_m, joint_v, **args)

    # Element-wise single-tensor calls
    single_p = [p.clone() for p in params]
    single_m = [m.clone() for m in exp_avgs]
    single_v = [v.clone() for v in exp_avg_sqs]
    for i in range(len(params)):
        eager_step([single_p[i]], [grads[i]], [single_m[i]], [single_v[i]], **args)

    for a, e in zip(joint_p + joint_m + joint_v, single_p + single_m + single_v):
        torch.testing.assert_close(a, e, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("bias_correction2_sqrt", [0.1, 0.5, 0.9, 0.999])
def test_eager_step_bias_correction2_sensitivity(bias_correction2_sqrt):
    """Varying bias_correction2_sqrt: rsqrtf division must stay consistent with reference."""
    params, grads, exp_avgs, exp_avg_sqs = _state(seed=37)
    actual = [t.clone() for t in params]
    actual_m = [t.clone() for t in exp_avgs]
    actual_v = [t.clone() for t in exp_avg_sqs]
    expected = [t.clone() for t in params]
    expected_m = [t.clone() for t in exp_avgs]
    expected_v = [t.clone() for t in exp_avg_sqs]
    args = dict(
        decay_factor=1.0, beta2=0.9, first_moment_weight=0.1,
        second_moment_weight=0.1, eps=1e-8,
        bias_correction1=1.0, bias_correction2_sqrt=bias_correction2_sqrt, lr=1e-3,
    )
    eager_step(actual, grads, actual_m, actual_v, **args)
    _reference_update(expected, grads, expected_m, expected_v, **args)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-6)
