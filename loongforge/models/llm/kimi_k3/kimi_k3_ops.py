# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Megatron-Bridge under the Apache-2.0 License:
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/kimi/kimi_k3_ops.py

"""Kimi K3 numerical operators."""

from functools import cache

import torch
from torch import nn
from megatron.legacy.model.rms_norm import RMSNorm


def kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    *,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Kimi Delta Attention while preserving packed-sequence boundaries.

    The q/k L2 norm, the gate transform and sigmoid(beta) all happen inside the
    kernel, so callers must not apply them.
    """
    from fla.ops.kda import chunk_kda

    _drop_broken_hopper_autotune()
    output, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        transpose_state_layout=True,
        cu_seqlens=cu_seqlens,
    )
    return output


def situ_and_mul(
    inputs: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Apply K3's SiTU gated activation to a fused gate/up projection."""
    gate, linear = torch.chunk(inputs.float(), 2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        linear = linear_beta * torch.tanh(linear / linear_beta)
    return (gate * linear).to(inputs.dtype)


class SiTUAndMul(nn.Module):
    """Module wrapper used by MCore's custom-activation path."""

    def __init__(self, config) -> None:
        super().__init__()
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply SiTU to a fused gate/up projection."""
        return situ_and_mul(inputs, self.beta, self.linear_beta)


def attn_res_aggregate(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_norm: RMSNorm,
    score_proj: nn.Linear,
    output_norm: nn.Module | None = None,
) -> torch.Tensor:
    """Mix the AttnRes snapshots and the current prefix by learned attention.

    Scores come from normalized rows, but the returned mixture is over the
    un-normalized rows, matching the reference implementation.
    """
    rows = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
    rows_float = rows.float()
    normalized = rows_float * torch.rsqrt(rows_float.square().mean(dim=-1, keepdim=True) + score_norm.eps)
    score_weight = score_norm.weight.float() * score_proj.weight.squeeze(0).float()
    probabilities = torch.softmax((normalized * score_weight).sum(dim=-1), dim=-1)
    mixed = (probabilities.unsqueeze(-1) * rows_float).sum(dim=-2).to(rows.dtype)
    return mixed if output_norm is None else output_norm(mixed)


def sum_grads_across_tp(module: nn.Module) -> None:
    """Mark replicated parameters for TP gradient reduction."""
    for parameter in module.parameters():
        parameter.allreduce_gradients_across_tp_domain = True


@cache
def _drop_broken_hopper_autotune() -> None:
    """Drop an invalid Hopper autotune choice."""
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_kernel_wy_dqkg_fused
    from fla.utils import IS_NVIDIA_HOPPER

    if not IS_NVIDIA_HOPPER:
        return

    autotuner = chunk_kda_bwd_kernel_wy_dqkg_fused.fn
    autotuner.configs = [
        config for config in autotuner.configs
        if not (config.kwargs["BK"] == 32 and config.num_warps == 4)
    ]


__all__ = [
    "RMSNorm",
    "SiTUAndMul",
    "attn_res_aggregate",
    "kda",
    "situ_and_mul",
    "sum_grads_across_tp",
]
