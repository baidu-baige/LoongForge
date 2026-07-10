# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image modules for DiT"""

import math
from typing import Optional

import torch
import torch.nn as nn


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
    computation_device=None,
    align_dtype_to_timestep: bool = False,
):
    """Sinusoidal timestep embedding compatible with the diffusers layout."""
    if timesteps.dim() != 1:
        raise ValueError("timesteps must be a 1-D tensor")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device if computation_device is None else computation_device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    if align_dtype_to_timestep:
        emb = emb.to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TemporalTimesteps(nn.Module):
    """Wrap ``get_timestep_embedding`` as an ``nn.Module``."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        computation_device=None,
        scale: float = 1,
        align_dtype_to_timestep: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.computation_device = computation_device
        self.scale = scale
        self.align_dtype_to_timestep = align_dtype_to_timestep

    def forward(self, timesteps: torch.Tensor):
        """Return sinusoidal embeddings for ``timesteps``."""
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            computation_device=self.computation_device,
            scale=self.scale,
            align_dtype_to_timestep=self.align_dtype_to_timestep,
        )


class DiffusersCompatibleTimestepProj(nn.Module):
    """Two-layer MLP matching the diffusers Qwen-Image time projection."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim_in, dim_out)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor):
        """Run ``linear_1 -> SiLU -> linear_2`` on ``x``."""
        return self.linear_2(self.act(self.linear_1(x)))


class TimestepEmbeddings(nn.Module):
    """Compose ``TemporalTimesteps`` with the diffusers-style projector."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        computation_device=None,
        diffusers_compatible_format: bool = False,
        scale: float = 1,
        align_dtype_to_timestep: bool = False,
        use_additional_t_cond: bool = False,
    ):
        super().__init__()
        self.time_proj = TemporalTimesteps(
            num_channels=dim_in,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            computation_device=computation_device,
            scale=scale,
            align_dtype_to_timestep=align_dtype_to_timestep,
        )
        if diffusers_compatible_format:
            self.timestep_embedder = DiffusersCompatibleTimestepProj(dim_in, dim_out)
        else:
            self.timestep_embedder = nn.Sequential(
                nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
            )
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, dim_out)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype, addition_t_cond=None):
        """Return the projected timestep embedding, optionally with an additional token bank."""
        time_emb = self.time_proj(timestep).to(dtype)
        time_emb = self.timestep_embedder(time_emb)
        if addition_t_cond is not None:
            time_emb = time_emb + self.addition_t_embedding(addition_t_cond).to(dtype=dtype)
        return time_emb


class RMSNorm(nn.Module):
    """Simple RMSNorm implementation used when TE fused norms are not available."""

    def __init__(self, dim: int, eps: float, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((dim,))) if elementwise_affine else None

    def forward(self, hidden_states: torch.Tensor):
        """Apply RMSNorm to ``hidden_states`` along the last dimension."""
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states


class AdaLayerNorm(nn.Module):
    """AdaLN modulation used by Qwen-Image (single / dual / default variants)."""

    def __init__(self, dim: int, single: bool = False, dual: bool = False):
        super().__init__()
        self.single = single
        self.dual = dual
        self.linear = nn.Linear(dim, dim * [[6, 2][single], 9][dual])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        """Apply the AdaLN modulation, returning the shift/scale/gate slots."""
        emb = self.linear(torch.nn.functional.silu(emb))
        if self.single:
            scale, shift = emb.unsqueeze(1).chunk(2, dim=2)
            return self.norm(x) * (1 + scale) + shift
        if self.dual:
            chunks = emb.unsqueeze(1).chunk(9, dim=2)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = chunks
            norm_x = self.norm(x)
            x = norm_x * (1 + scale_msa) + shift_msa
            norm_x2 = norm_x * (1 + scale_msa2) + shift_msa2
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_x2, gate_msa2
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.unsqueeze(1).chunk(6, dim=2)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


def linear_out(module, x: torch.Tensor):
    """Call ``module(x)`` and merge Megatron's ``(out, bias)`` tuple result if present."""
    out = module(x)
    if isinstance(out, tuple):
        return out[0] if out[1] is None else out[0] + out[1]
    return out


class ApproximateGELU(nn.Module):
    """QuickGELU-style approximation ``x * sigmoid(1.702 * x)``."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
        linear_cls=None,
        column_linear_cls=None,
        config=None,
    ):
        super().__init__()
        linear_cls = linear_cls or nn.Linear
        column_linear_cls = column_linear_cls or linear_cls
        if column_linear_cls is nn.Linear:
            self.proj = column_linear_cls(dim_in, dim_out, bias=bias)
        else:
            self.proj = column_linear_cls(
                dim_in,
                dim_out,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=bias,
                skip_bias_add=False,
                is_expert=False,
                skip_weight_param_allocation=False,
                tp_comm_buffer_name="fc1",
            )

    def forward(self, x: torch.Tensor):
        """Project ``x`` and apply the approximate GELU non-linearity."""
        x = linear_out(self.proj, x)
        return x * torch.sigmoid(1.702 * x)


class QwenFeedForward(nn.Module):
    """Qwen-Image FFN block (approximate GELU + row-parallel down projection)."""

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        dropout: float = 0.0,
        linear_cls=None,
        column_linear_cls=None,
        row_linear_cls=None,
        config=None,
    ):
        super().__init__()
        inner_dim = int(dim * 4)
        dim_out = dim if dim_out is None else dim_out
        linear_cls = linear_cls or nn.Linear
        column_linear_cls = column_linear_cls or linear_cls
        row_linear_cls = row_linear_cls or linear_cls
        self.net = nn.ModuleList([])
        self.net.append(
            ApproximateGELU(
                dim, inner_dim, linear_cls=linear_cls, column_linear_cls=column_linear_cls, config=config
            )
        )
        self.net.append(nn.Dropout(dropout))
        if row_linear_cls is nn.Linear:
            self.net.append(row_linear_cls(inner_dim, dim_out))
        else:
            self.net.append(
                row_linear_cls(
                    inner_dim,
                    dim_out,
                    config=config,
                    init_method=config.init_method,
                    bias=True,
                    input_is_parallel=True,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name="fc2",
                )
            )

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """Run the FFN stack; unwrap Megatron ``(out, bias)`` tuples in-flight."""
        for module in self.net:
            hidden_states = module(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0] if hidden_states[1] is None else hidden_states[0] + hidden_states[1]
        return hidden_states
