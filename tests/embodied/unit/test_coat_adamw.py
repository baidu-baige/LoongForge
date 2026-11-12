# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the COAT-style FP8 optimizer-state AdamW (needs a GPU and Triton)."""

from __future__ import annotations

import copy
import math

import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="COATAdamW needs a CUDA GPU")

from loongforge.embodied.optimizer.coat_adamw import FP8_MAX, QGROUP, CoatAdamW  # noqa: E402

LR, BETAS, EPS, WD = 2.5e-5, (0.9, 0.95), 1e-8, 0.01
EXPAND_MIN = 16


def _deq(q, scale, k, smm, signed):
    x = q.float() * scale[:, None]
    ax = x.abs()
    out = torch.where(ax > 0, torch.exp2(torch.log2(ax.clamp_min(1e-38)) / k[:, None]) * smm[:, None], torch.zeros_like(ax))
    return out * torch.where(x < 0, -1.0, 1.0) if signed else out


def _quant(x, signed):
    ax = x.abs()
    mx = ax.max(1).values + 1e-30
    mn = torch.where(ax > 0, ax, torch.full_like(ax, 3e38)).min(1).values
    mn = torch.where(mn >= 3e38, torch.zeros_like(mn), mn) + 1e-30
    smm = mx.sqrt() * mn.sqrt()
    logr = torch.log2(mx / mn).clamp_min(1e-3)
    k = (torch.floor(math.log2(FP8_MAX * FP8_MAX / 2) / logr * EXPAND_MIN) / EXPAND_MIN).clamp(1.0 / EXPAND_MIN, 64.0)
    scale = torch.exp2(k * torch.log2(mx / smm)) / FP8_MAX
    y = torch.where(ax > 0, torch.exp2(k[:, None] * torch.log2(ax.clamp_min(1e-38) / smm[:, None])), torch.zeros_like(ax))
    if signed:
        y = y * torch.where(x < 0, -1.0, 1.0)
    q = (y / scale[:, None]).clamp(-FP8_MAX if signed else 0.0, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale, k, smm


class _Reference:
    """The same update as the Triton kernel, written with plain PyTorch ops."""

    def __init__(self, n, device):
        self.n, self.g = n, (n + QGROUP - 1) // QGROUP
        zeros = lambda: torch.zeros(self.g, device=device)
        ones = lambda: torch.ones(self.g, device=device)
        q0 = torch.zeros(self.g, QGROUP, device=device, dtype=torch.float8_e4m3fn)
        self.m, self.ms, self.mk, self.msmm = q0.clone(), zeros(), ones(), ones()
        self.v, self.vs, self.vk, self.vsmm = q0.clone(), zeros(), ones(), ones()

    def step(self, p, grad, step):
        pad = self.g * QGROUP - self.n
        pf = torch.nn.functional.pad(p.float(), (0, pad)).view(self.g, QGROUP)
        gf = torch.nn.functional.pad(grad.float(), (0, pad)).view(self.g, QGROUP)
        m = _deq(self.m, self.ms, self.mk, self.msmm, True)
        v = _deq(self.v, self.vs, self.vk, self.vsmm, False)
        m = BETAS[0] * m + (1 - BETAS[0]) * gf
        v = BETAS[1] * v + (1 - BETAS[1]) * gf * gf
        denom = (v.sqrt() / math.sqrt(1 - BETAS[1] ** step) + EPS) * (1 - BETAS[0] ** step)
        pf = pf - LR * (m / denom + WD * pf)
        mask = (torch.arange(self.g * QGROUP, device=p.device) < self.n).view(self.g, QGROUP)
        self.m, self.ms, self.mk, self.msmm = _quant(m * mask, True)
        self.v, self.vs, self.vk, self.vsmm = _quant(v * mask, False)
        return pf.view(-1)[: self.n].to(p.dtype)


def _grad(n, dtype, seed, scale=1e-3):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, device="cuda", generator=gen) * scale).to(dtype)


def _opt(params):
    return CoatAdamW(params, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)


@pytest.mark.parametrize("n, dtype", [(130, torch.bfloat16), (1000, torch.bfloat16), (4096, torch.float32)])
def test_kernel_matches_the_plain_pytorch_reference(n, dtype):
    torch.manual_seed(0)
    p0 = (torch.randn(n, device="cuda") * 0.02).to(dtype)
    param = torch.nn.Parameter(p0.clone())
    opt, ref, p_ref = _opt([param]), _Reference(n, "cuda"), p0.clone()
    for step in range(1, 11):
        grad = _grad(n, dtype, step)
        param.grad = grad.clone()
        opt.step()
        p_ref = ref.step(p_ref, grad, step)
    tol = 2e-6 if dtype == torch.float32 else 2 ** -7 * float(p0.abs().max())
    assert (param.detach().float() - p_ref.float()).abs().max().item() <= tol
    st = opt.state[param]
    m_kernel = _deq(st["exp_avg"].view(torch.uint8).view(-1)[: ref.g * QGROUP].view(torch.float8_e4m3fn).view(ref.g, QGROUP)
                    if n % QGROUP == 0 else
                    torch.nn.functional.pad(st["exp_avg"].view(torch.uint8), (0, ref.g * QGROUP - n)).view(torch.float8_e4m3fn).view(ref.g, QGROUP),
                    st["exp_avg_scale"], st["exp_avg_exp"], st["exp_avg_smm"], True)
    m_ref = _deq(ref.m, ref.ms, ref.mk, ref.msmm, True)
    assert ((m_kernel - m_ref).abs().max() / m_ref.abs().max()).item() < 1e-4


def test_state_dict_roundtrip_keeps_fp8_dtypes_and_resumes_exactly():
    n = 5000
    a = torch.nn.Parameter((torch.randn(n, device="cuda") * 0.02).to(torch.bfloat16))
    opt_a = _opt([a])
    for step in range(1, 4):
        a.grad = _grad(n, torch.bfloat16, step)
        opt_a.step()
    b = torch.nn.Parameter(a.detach().clone())
    opt_b = _opt([b])
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))
    st = opt_b.state[b]
    assert st["exp_avg"].dtype == torch.float8_e4m3fn and st["exp_avg_sq"].dtype == torch.float8_e4m3fn
    assert st["exp_avg_scale"].dtype == torch.float32 and st["step"].device.type == "cpu"
    a.grad, b.grad = _grad(n, torch.bfloat16, 99), _grad(n, torch.bfloat16, 99)
    opt_a.step()
    opt_b.step()
    assert torch.equal(a, b)


def test_rejects_a_gradient_dtype_that_differs_from_the_parameter():
    param = torch.nn.Parameter(torch.zeros(256, device="cuda", dtype=torch.bfloat16))
    if not hasattr(param, "grad_dtype"):
        pytest.skip("this torch rejects a mismatched grad at assignment time")
    param.grad_dtype = None  # allow a grad dtype that differs from the parameter
    param.grad = torch.ones(256, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="grad dtype"):
        _opt([param]).step()


def test_all_zero_gradient_keeps_parameters_and_states_finite():
    param = torch.nn.Parameter(torch.zeros(3 * QGROUP, device="cuda", dtype=torch.float32))
    opt = _opt([param])
    for _ in range(3):
        param.grad = torch.zeros_like(param)
        opt.step()
    st = opt.state[param]
    assert torch.isfinite(param).all()
    for key in ("exp_avg_scale", "exp_avg_exp", "exp_avg_smm", "exp_avg_sq_scale", "exp_avg_sq_exp", "exp_avg_sq_smm"):
        assert torch.isfinite(st[key]).all(), key
    assert torch.isfinite(st["exp_avg"].float()).all() and torch.isfinite(st["exp_avg_sq"].float()).all()


def test_tracks_torch_adamw_on_fp32_parameters():
    n = 64 * QGROUP
    base = (torch.randn(n, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3)) * 0.02)
    drift = torch.randn(n, device="cuda", generator=torch.Generator(device="cuda").manual_seed(4))
    ours, theirs = torch.nn.Parameter(base.clone()), torch.nn.Parameter(base.clone())
    opt_a = _opt([ours])
    opt_b = torch.optim.AdamW([theirs], lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    for step in range(1, 101):
        grad = (_grad(n, torch.float32, step) + 0.3e-3 * drift)
        ours.grad, theirs.grad = grad.clone(), grad.clone()
        opt_a.step()
        opt_b.step()
    d_a, d_b = (ours - base).detach(), (theirs - base).detach()
    cosine = (d_a * d_b).sum() / (d_a.norm() * d_b.norm())
    assert cosine.item() > 0.999
    assert ((d_a - d_b).norm() / d_b.norm()).item() < 0.1
