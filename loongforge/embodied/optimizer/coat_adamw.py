# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# AdamW whose first and second moments are stored as FP8 (E4M3), after
# COAT (NVlabs/COAT, "Compressing Optimizer states and Activation for
# memory-efficient FP8 Training", Apache-2.0, coat/optimizer/fp8_adamw.py and
# kernels/fp8_adamw_expand_cuda_kernel.cu).
#
# What was taken from COAT:
#   * per-group (128 elements) FP8 storage with a dynamic range expansion:
#     before the cast, x -> sign(x) * (|x| / sqrt(min*max)) ** k with
#     k = floor(log2(448**2 / 2) / log2(max / min) * expand_min) / expand_min,
#     which stretches the group's range onto the full E4M3 range; the group keeps
#     three fp32 numbers (scale, k, sqrt(min*max)).
#   * dequantize, run the AdamW update in fp32 registers, re-quantize, all in one
#     kernel, one program per group.
# What differs from the COAT reference code:
#   * Triton instead of a CUDA extension (no nvcc build step).
#   * the step counter lives on the CPU: no GPU sync per parameter.
#   * the expansion exponent is clamped to [1/expand_min, 64] and an all-zero
#     group is handled explicitly (the CUDA code divides by log2(1) there).
#   * load_state_dict keeps the FP8 / fp32 dtypes of the saved state instead of
#     casting them to the parameter dtype.

from __future__ import annotations

import math
from itertools import chain

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
QGROUP = 128
# Triton kernels may only read module globals that are constexpr.
_FP8_MAX = tl.constexpr(448.0)
_LOG2_RATIO_UB = tl.constexpr(math.log2(448.0 * 448.0 / 2))
_QUANT_MIN_VAL = tl.constexpr(1e-30)
_BIG = tl.constexpr(3.0e38)


@triton.jit
def _coat_adamw_kernel(
    P, G,
    M, M_SCALE, M_EXP, M_SMM,
    V, V_SCALE, V_EXP, V_SMM,
    N, lr, wd, beta1, beta2, eps, bc1, bc2_sqrt,
    EXPAND_MIN: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    p = tl.load(P + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- dequantize the two moments ------------------------------------
    mq = tl.load(M + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.load(M_SCALE + pid)
    mk = tl.load(M_EXP + pid)
    msmm = tl.load(M_SMM + pid)
    x = mq * ms
    ax = tl.abs(x)
    m = tl.where(ax > 0.0, tl.where(x < 0.0, -1.0, 1.0) * tl.exp2(tl.log2(ax) / mk) * msmm, 0.0)

    vq = tl.load(V + offs, mask=mask, other=0.0).to(tl.float32)
    vs = tl.load(V_SCALE + pid)
    vk = tl.load(V_EXP + pid)
    vsmm = tl.load(V_SMM + pid)
    y = vq * vs
    v = tl.where(y > 0.0, tl.exp2(tl.log2(y) / vk) * vsmm, 0.0)

    # ---- AdamW update (fp32) -------------------------------------------
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    denom = (tl.sqrt(v) / bc2_sqrt + eps) * bc1
    p = p - lr * (m / denom + wd * p)
    tl.store(P + offs, p.to(P.dtype.element_ty), mask=mask)

    # ---- re-quantize m and v with dynamic range expansion --------------
    am = tl.abs(m)
    # first moment
    m_max = tl.max(tl.where(mask, am, 0.0), axis=0)
    m_min = tl.min(tl.where(mask & (am > 0.0), am, _BIG), axis=0)
    m_min = tl.where(m_min >= _BIG, 0.0, m_min)
    m_max = m_max + _QUANT_MIN_VAL
    m_min = m_min + _QUANT_MIN_VAL
    m_smm = tl.sqrt(m_max) * tl.sqrt(m_min)
    m_logr = tl.maximum(tl.log2(m_max / m_min), 1.0e-3)
    m_k = tl.floor(_LOG2_RATIO_UB / m_logr * EXPAND_MIN) / EXPAND_MIN
    m_k = tl.minimum(tl.maximum(m_k, 1.0 / EXPAND_MIN), 64.0)
    m_new_scale = tl.exp2(m_k * tl.log2(m_max / m_smm)) / _FP8_MAX
    m_y = tl.where(am > 0.0, tl.where(m < 0.0, -1.0, 1.0) * tl.exp2(m_k * tl.log2(am / m_smm)), 0.0)
    m_q = tl.minimum(tl.maximum(m_y / m_new_scale, -_FP8_MAX), _FP8_MAX)
    tl.store(M + offs, m_q.to(tl.float8e4nv), mask=mask)
    tl.store(M_SCALE + pid, m_new_scale)
    tl.store(M_EXP + pid, m_k)
    tl.store(M_SMM + pid, m_smm)

    # second moment (non-negative)
    v_max = tl.max(tl.where(mask, v, 0.0), axis=0)
    v_min = tl.min(tl.where(mask & (v > 0.0), v, _BIG), axis=0)
    v_min = tl.where(v_min >= _BIG, 0.0, v_min)
    v_max = v_max + _QUANT_MIN_VAL
    v_min = v_min + _QUANT_MIN_VAL
    v_smm = tl.sqrt(v_max) * tl.sqrt(v_min)
    v_logr = tl.maximum(tl.log2(v_max / v_min), 1.0e-3)
    v_k = tl.floor(_LOG2_RATIO_UB / v_logr * EXPAND_MIN) / EXPAND_MIN
    v_k = tl.minimum(tl.maximum(v_k, 1.0 / EXPAND_MIN), 64.0)
    v_new_scale = tl.exp2(v_k * tl.log2(v_max / v_smm)) / _FP8_MAX
    v_y = tl.where(v > 0.0, tl.exp2(v_k * tl.log2(v / v_smm)), 0.0)
    v_q = tl.minimum(tl.maximum(v_y / v_new_scale, 0.0), _FP8_MAX)
    tl.store(V + offs, v_q.to(tl.float8e4nv), mask=mask)
    tl.store(V_SCALE + pid, v_new_scale)
    tl.store(V_EXP + pid, v_k)
    tl.store(V_SMM + pid, v_smm)


class CoatAdamW(torch.optim.Optimizer):
    """AdamW with FP8 (E4M3) first/second moments (COAT dynamic range expansion)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        expand_min: int = 16,
        **_ignored,   # foreach / fused / ... passed by generic optimizer builders
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, expand_min=expand_min),
        )

    @staticmethod
    def _init_state(p: torch.Tensor) -> dict:
        groups = (p.numel() + QGROUP - 1) // QGROUP
        f32 = dict(device=p.device, dtype=torch.float32)
        return {
            "step": torch.tensor(0.0),
            "exp_avg": torch.zeros_like(p, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format),
            "exp_avg_scale": torch.zeros(groups, **f32),
            "exp_avg_exp": torch.ones(groups, **f32),
            "exp_avg_smm": torch.ones(groups, **f32),
            "exp_avg_sq": torch.zeros_like(p, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format),
            "exp_avg_sq_scale": torch.zeros(groups, **f32),
            "exp_avg_sq_exp": torch.ones(groups, **f32),
            "exp_avg_sq_smm": torch.ones(groups, **f32),
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.dtype != p.dtype:
                    raise RuntimeError(f"CoatAdamW needs grad dtype == param dtype, got {p.grad.dtype} vs {p.dtype}")
                if not p.is_contiguous() or not p.grad.is_contiguous():
                    raise RuntimeError("CoatAdamW needs contiguous parameters and gradients")
                st = self.state[p]
                if len(st) == 0:
                    st.update(self._init_state(p))
                st["step"] += 1
                step = int(st["step"].item())            # CPU tensor, no GPU sync
                n = p.numel()
                _coat_adamw_kernel[((n + QGROUP - 1) // QGROUP,)](
                    p, p.grad,
                    st["exp_avg"], st["exp_avg_scale"], st["exp_avg_exp"], st["exp_avg_smm"],
                    st["exp_avg_sq"], st["exp_avg_sq_scale"], st["exp_avg_sq_exp"], st["exp_avg_sq_smm"],
                    n, group["lr"], group["weight_decay"], beta1, beta2, group["eps"],
                    1.0 - beta1 ** step, math.sqrt(1.0 - beta2 ** step),
                    EXPAND_MIN=group["expand_min"], BLOCK=QGROUP, num_warps=4,
                )
        return loss

    @torch._disable_dynamo
    def load_state_dict(self, state_dict) -> None:
        """Like Optimizer.load_state_dict, but never casts FP8 / fp32 state to the param dtype."""
        saved_state = {k: v for k, v in state_dict["state"].items()}
        super().load_state_dict(state_dict)
        saved_ids = chain.from_iterable(g["params"] for g in state_dict["param_groups"])
        params = chain.from_iterable(g["params"] for g in self.param_groups)
        for saved_id, p in zip(saved_ids, params):
            if saved_id not in saved_state:
                continue
            new = {}
            for key, value in saved_state[saved_id].items():
                if key == "step":
                    new[key] = value.detach().to("cpu", torch.float32) if torch.is_tensor(value) else torch.tensor(float(value))
                elif torch.is_tensor(value):
                    new[key] = value.detach().to(device=p.device).clone()   # keeps float8_e4m3fn / float32
                else:
                    new[key] = value
            self.state[p] = new

    @staticmethod
    def state_bytes(p: torch.Tensor) -> int:
        """Bytes of optimizer state kept for a parameter of this size."""
        groups = (p.numel() + QGROUP - 1) // QGROUP
        return 2 * p.numel() + 2 * 3 * 4 * groups
