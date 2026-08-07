# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Hugging Face Transformers under the Apache-2.0 License.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MiniCPM-V-4.6 GatedDeltaNet."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformer_engine.pytorch import Linear as TE_Linear

from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.huggingface import HuggingFaceModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import (
    deprecate_inference_params,
    nvtx_range_pop,
    nvtx_range_push,
)
from .context_parallel import (
    gather_from_context_parallel_region,
    scatter_to_context_parallel_region,
)
from .linear_utils import torch_linear_forward

try:
    from fla.modules import FusedRMSNormGated
    from fla.modules import fused_norm_gate as fla_fused_norm_gate
    from fla.modules.convolution import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    FusedRMSNormGated = None
    fla_fused_norm_gate = None
    causal_conv1d = None
    chunk_gated_delta_rule = None
    HAVE_FLA = False


def _pin_fla_gated_norm_autotune():
    """Select one FLA kernel configuration so reductions are reproducible."""
    if fla_fused_norm_gate is None:
        raise ImportError(
            "gated_norm_backend=fla_deterministic requires flash-linear-attention"
        )

    kernel_specs = {
        "layer_norm_gated_fwd_kernel": {"BT": 16},
        "layer_norm_gated_bwd_kernel": {"BT": 16},
        "layer_norm_gated_fwd_kernel1": {},
        "layer_norm_gated_bwd_kernel1": {},
    }
    for kernel_name, expected_kwargs in kernel_specs.items():
        autotuner = getattr(fla_fused_norm_gate, kernel_name).fn
        matching_configs = [
            config
            for config in autotuner.configs
            if config.kwargs == expected_kwargs
            and config.num_warps == 8
            and config.num_stages == 3
        ]
        if len(matching_configs) != 1:
            raise RuntimeError(
                f"Unable to select deterministic FLA config for {kernel_name}"
            )
        autotuner.configs = matching_configs
        autotuner.cache.clear()

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def _torch_l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    # HF training executes this reduction under CUDA autocast, which keeps the
    # elementwise square in the input dtype and accumulates the sum in FP32.
    inv_norm = torch.rsqrt(
        (x * x).sum(dim=dim, keepdim=True, dtype=torch.float32) + eps
    )
    return x * inv_norm


def _torch_causal_conv1d(
    x,
    weight,
    bias=None,
    activation=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
):
    def _conv_segment(segment):
        original_dtype = segment.dtype
        segment = segment.transpose(1, 2).contiguous().to(weight.dtype)
        segment = F.pad(segment, (weight.shape[-1] - 1, 0))
        out = F.conv1d(segment, weight.unsqueeze(1), bias, groups=weight.shape[0])
        if activation in ("silu", "swish"):
            out = F.silu(out)
        elif activation is not None:
            raise ValueError(f"Unsupported causal conv activation: {activation}")
        return out.transpose(1, 2).contiguous().to(original_dtype)

    cu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if cu is None:
        return (_conv_segment(x),)

    if x.shape[0] != 1:
        raise NotImplementedError("Packed fallback causal conv supports batch size 1 only.")

    cu = cu.detach().cpu().tolist()
    outputs = []
    for start, end in zip(cu[:-1], cu[1:]):
        outputs.append(_conv_segment(x[:, start:end, :]))
    return (torch.cat(outputs, dim=1),)


def _torch_module_causal_conv1d(
    x,
    weight,
    bias=None,
    activation=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
):
    """Match a padded ``nn.Conv1d`` followed by causal output cropping."""
    def _conv_segment(segment):
        original_dtype = segment.dtype
        sequence_length = segment.shape[1]
        segment = segment.transpose(1, 2).contiguous().to(weight.dtype)
        output = F.conv1d(
            segment,
            weight.unsqueeze(1),
            bias,
            padding=weight.shape[-1] - 1,
            groups=weight.shape[0],
        )[:, :, :sequence_length]
        if activation in ("silu", "swish"):
            output = F.silu(output)
        elif activation is not None:
            raise ValueError(f"Unsupported causal conv activation: {activation}")
        return output.transpose(1, 2).to(original_dtype)

    cu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if cu is None:
        return (_conv_segment(x),)
    if x.shape[0] != 1:
        raise NotImplementedError("Packed torch-module causal conv supports batch size 1 only.")
    boundaries = cu.detach().cpu().tolist()
    outputs = [
        _conv_segment(x[:, start:end, :])
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    return (torch.cat(outputs, dim=1),)


def _fused_causal_conv1d(
    x,
    weight,
    bias=None,
    activation=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
):
    """Call causal-conv1d with the batch-sequence-hidden layout used here."""
    if causal_conv1d_fn is None:
        raise ImportError("causal-conv1d is required for the fused causal conv backend")

    def _conv_segment(segment):
        output = causal_conv1d_fn(
            x=segment.transpose(1, 2).contiguous(),
            weight=weight,
            bias=bias,
            activation=activation,
        )
        return output.transpose(1, 2).contiguous()

    cu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if cu is None:
        return (_conv_segment(x),)
    if x.shape[0] != 1:
        raise NotImplementedError("Packed fused causal conv supports batch size 1 only.")
    boundaries = cu.detach().cpu().tolist()
    outputs = [
        _conv_segment(x[:, start:end, :])
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    return (torch.cat(outputs, dim=1),)


def _torch_chunk_gated_delta_rule_impl(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    del kwargs
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _torch_l2norm(query, dim=-1, eps=1e-6)
        key = _torch_l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0],
        core_attn_out.shape[1],
        -1,
        core_attn_out.shape[-1],
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _torch_chunk_gated_delta_rule(query, key, value, *args, **kwargs):
    """Run the HF torch fallback with its mixed-precision operator policy."""
    if query.is_cuda and query.dtype in (torch.bfloat16, torch.float16):
        with torch.autocast(device_type="cuda", dtype=query.dtype):
            return _torch_chunk_gated_delta_rule_impl(
                query, key, value, *args, **kwargs
            )
    return _torch_chunk_gated_delta_rule_impl(query, key, value, *args, **kwargs)


def _torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    **kwargs,
):
    """Deterministic FP32 recurrent gated-delta rule."""
    del kwargs
    cu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if cu is not None:
        if query.shape[0] != 1 or initial_state is not None:
            raise NotImplementedError(
                "Packed recurrent gated-delta rule supports batch size 1 without an initial state."
            )
        boundaries = cu.detach().cpu().tolist()
        outputs = []
        final_state = None
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            output, final_state = _torch_recurrent_gated_delta_rule(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1), final_state

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _torch_l2norm(query, dim=-1, eps=1e-6)
        key = _torch_l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        tensor.transpose(1, 2).contiguous().to(torch.float32)
        for tensor in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, key_head_dim = key.shape
    value_head_dim = value.shape[-1]
    query = query * (1 / (key_head_dim**0.5))
    recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            key_head_dim,
            value_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    output = torch.zeros(
        batch_size,
        num_heads,
        sequence_length,
        value_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    for index in range(sequence_length):
        query_t = query[:, :, index]
        key_t = key[:, :, index]
        value_t = value[:, :, index]
        decay_t = g[:, :, index].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, index].unsqueeze(-1)
        recurrent_state = recurrent_state * decay_t
        memory = (recurrent_state * key_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value_t - memory) * beta_t
        recurrent_state = recurrent_state + key_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, index] = (recurrent_state * query_t.unsqueeze(-1)).sum(dim=-2)

    final_state = recurrent_state if output_final_state else None
    output = output.transpose(1, 2).contiguous().to(initial_dtype)
    return output, final_state


class Qwen3NextRMSNormGated(nn.Module):
    """
    The RMSNorm layer with gating, used in the Qwen3-Next model.
    Args:
        hidden_size (int): The dimension size of the hidden layer
        eps (float, optional): Numerical stability parameter, default to 1e-6
        **kwargs: Other optional parameters
    """
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        """forward"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class Qwen3NextRMSNorm(torch.nn.Module):
    """
    Zero-Centered RMSNorm for Qwen3-Next.
    Uses (1 + weight) scaling to match HuggingFace implementation exactly.
    This eliminates the need for +1/-1 offset during weight conversion.

    Interface matches TENorm for compatibility with Megatron-Core build_module.
    """
    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.config = config
        self.eps = eps
        # Initialize weight to zeros (Zero-Centered), matching HuggingFace Qwen3NextRMSNorm
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size))
        setattr(self.weight, 'sequence_parallel', self.config.sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, hidden_states):
        """forward"""
        output = self._norm(hidden_states.float())
        # Zero-Centered: use (1 + weight) instead of weight
        # This matches HuggingFace's Qwen3NextRMSNorm exactly
        output = output * (1.0 + self.weight.float())
        return output.type_as(hidden_states)


@dataclass
class GatedDeltaNetSubmodules:
    """
    Contains the module specs for the input linear, output norm, and output linear layers.
    """
    in_proj_qkvz: Union[ModuleSpec, type] = IdentityOp
    in_proj_ba: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class MiniCPMV46GatedDeltaNet(HuggingFaceModule):
    """Gated Delta Net (GDN) layer class
    GDN layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaNetSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        projection_split_mode: str = "merged",
        gated_delta_rule_backend: str = "auto",
        gated_norm_backend: str = "auto",
        causal_conv_backend: str = "auto",
        linear_backend: str = "transformer_engine",
        A_init_range: Tuple[float, float] = (0, 16),
        pg_collection: ProcessGroupCollection = None,
        **kwargs
    ):
        """
        Args:
            config: The config of the model.
            submodules: Contains the module specs for the input and output linear layers.
            layer_number: The layer number of this GDN layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization in the kernel of the gated delta rule.
            A_init_range: The initialization range for the attention weights.
            pg_collection: The required process groups to use for tensor model parallel and context
                parallel.
        """
        super().__init__(config)

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        self.projection_split_mode = projection_split_mode
        self.gated_delta_rule_backend = gated_delta_rule_backend
        if gated_norm_backend not in ("auto", "torch", "fla", "fla_deterministic"):
            raise ValueError(
                "gated_norm_backend must be one of: auto, torch, fla, "
                "fla_deterministic"
            )
        if gated_norm_backend in ("fla", "fla_deterministic") and FusedRMSNormGated is None:
            raise ImportError(
                f"gated_norm_backend={gated_norm_backend} requires "
                "flash-linear-attention"
            )
        if gated_norm_backend == "fla_deterministic":
            _pin_fla_gated_norm_autotune()
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet"
        self.pg_collection = pg_collection
        self.tp_size = self.pg_collection.tp.size()
        self.cp_size = self.pg_collection.cp.size()
        if linear_backend not in ("transformer_engine", "auto", "torch"):
            raise ValueError(
                "linear_backend must be one of: transformer_engine, auto, torch"
            )
        if linear_backend == "torch" and self.tp_size != 1:
            raise ValueError("The torch linear backend requires tensor parallel size 1")
        self.use_torch_linear = linear_backend == "torch" or (
            linear_backend == "auto" and self.tp_size == 1
        )
        self.sequence_parallel = config.sequence_parallel

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        assert self.activation in ["silu", "swish"], f"Only silu and swish are supported, but got {self.activation}"
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads

        # Conv1d for QKV
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        # weight shape: [conv_dim, 1, d_conv]
        # bias shape: [conv_dim]
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim,
            padding=self.conv_kernel_dim - 1,
        )

        self.in_proj_qkvz_dim = self.qk_dim * 2 + self.v_dim * 2
        self.in_proj_ba_dim = self.num_value_heads * 2
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_qkvz_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN layer "
                f"in_proj_qkvz output tensor ({self.in_proj_qkvz_dim}) must be a multiple of {fp8_align_size}."
            )
            assert self.in_proj_ba_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN layer "
                f"in_proj_ba output tensor ({self.in_proj_ba_dim}) must be a multiple of {fp8_align_size}."
            )
        self.in_proj_qkvz = TE_Linear(self.hidden_size, self.in_proj_qkvz_dim, bias=False)
        self.in_proj_ba = TE_Linear(self.hidden_size, self.in_proj_ba_dim, bias=False)
        self.in_proj_qkvz.canonical_split_sizes = {
            "in_proj_qkv": self.conv_dim,
            "in_proj_z": self.v_dim,
        }
        self.in_proj_ba.canonical_split_sizes = {
            "in_proj_b": self.num_value_heads,
            "in_proj_a": self.num_value_heads,
        }

        # dt_bias parameter
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        A = torch.empty(self.num_value_heads).uniform_(self.A_init_range[0], self.A_init_range[1])
        self.A_log = nn.Parameter(torch.log(A))
        if self.tp_size > 1:
            setattr(self.dt_bias, "average_gradients_across_tp_domain", True)
            setattr(self.A_log, "average_gradients_across_tp_domain", True)

        use_fused_gated_norm = gated_norm_backend in ("fla", "fla_deterministic") or (
            gated_norm_backend == "auto" and FusedRMSNormGated is not None
        )
        self.out_norm = (
            FusedRMSNormGated(
                self.value_head_dim,
                eps=self.config.layernorm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=self.config.params_dtype,
            )
            if use_fused_gated_norm
            else Qwen3NextRMSNormGated(
                self.value_head_dim, eps=self.config.layernorm_epsilon
            )
        )

        self.out_proj = TE_Linear(self.v_dim, self.hidden_size, bias=False)
        if gated_delta_rule_backend == "auto":
            self.chunk_gated_delta_rule = chunk_gated_delta_rule or _torch_chunk_gated_delta_rule
        elif gated_delta_rule_backend == "torch_chunk":
            self.chunk_gated_delta_rule = _torch_chunk_gated_delta_rule
        elif gated_delta_rule_backend == "torch_recurrent":
            self.chunk_gated_delta_rule = _torch_recurrent_gated_delta_rule
        else:
            raise ValueError(
                "gated_delta_rule_backend must be one of: auto, torch_chunk, torch_recurrent"
            )
        if causal_conv_backend == "auto":
            self.causal_conv1d = causal_conv1d or _torch_causal_conv1d
        elif causal_conv_backend == "torch":
            self.causal_conv1d = _torch_causal_conv1d
        elif causal_conv_backend == "torch_module":
            self.causal_conv1d = _torch_module_causal_conv1d
        elif causal_conv_backend == "causal_conv1d":
            if causal_conv1d_fn is None:
                raise ImportError(
                    "causal_conv_backend=causal_conv1d requires causal-conv1d"
                )
            self.causal_conv1d = _fused_causal_conv1d
        else:
            raise ValueError(
                "causal_conv_backend must be one of: auto, torch, torch_module, causal_conv1d"
            )

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`.
        """

        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_key_heads,
            2 * self.key_head_dim + 2 * self.value_head_dim * self.num_value_heads // self.num_key_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_key_heads, 2 * self.num_value_heads // self.num_key_heads
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.key_head_dim,
            self.key_head_dim,
            (self.num_value_heads // self.num_key_heads * self.value_head_dim),
            (self.num_value_heads // self.num_key_heads * self.value_head_dim),
        ]
        split_arg_list_ba = [self.num_value_heads // self.num_key_heads, self.num_value_heads // self.num_key_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.value_head_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.value_head_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_value_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_value_heads)
        return query, key, value, z, b, a

    def _split_qwen35_qkvz_weights(self):
        in_proj_qkvz = getattr(self.in_proj_qkvz, "to_wrap", self.in_proj_qkvz)
        weight = in_proj_qkvz.weight
        grouped = weight.reshape(
            self.num_key_heads,
            2 * self.key_head_dim + 2 * self.value_head_dim * self.num_value_heads // self.num_key_heads,
            self.hidden_size,
        )
        values_per_key_group = self.num_value_heads // self.num_key_heads * self.value_head_dim
        query = grouped[:, : self.key_head_dim, :].reshape(-1, self.hidden_size)
        key = grouped[:, self.key_head_dim : 2 * self.key_head_dim, :].reshape(-1, self.hidden_size)
        value = grouped[
            :,
            2 * self.key_head_dim : 2 * self.key_head_dim + values_per_key_group,
            :,
        ].reshape(-1, self.hidden_size)
        z = grouped[:, 2 * self.key_head_dim + values_per_key_group :, :].reshape(-1, self.hidden_size)
        return torch.cat((query, key, value), dim=0), z

    def _split_qwen35_ba_weights(self):
        in_proj_ba = getattr(self.in_proj_ba, "to_wrap", self.in_proj_ba)
        weight = in_proj_ba.weight
        values_per_key_group = self.num_value_heads // self.num_key_heads
        grouped = weight.reshape(self.num_key_heads, 2 * values_per_key_group, self.hidden_size)
        b = grouped[:, :values_per_key_group, :].reshape(-1, self.hidden_size)
        a = grouped[:, values_per_key_group:, :].reshape(-1, self.hidden_size)
        return b, a

    def apply_mask_to_padding_states(self, hidden_states, attention_mask):
        """
        Tunes out the hidden states for padding tokens according to the attention mask
        """
        # NOTE: attention mask is a 2D boolean tensor
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        return hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        Perform a forward pass through the GDN module.
        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.
        Return:
            (Tuple[Tensor, Tensor]) GDN output and bias.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if self.sequence_parallel and self.tp_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        if self.cp_size > 1:
            hidden_states = gather_from_context_parallel_region(
                hidden_states, packed_seq_params, self.pg_collection.cp
            )

        cu_seqlens = None if packed_seq_params is None else packed_seq_params.cu_seqlens_q
        cu_seqlens_cpu = None if packed_seq_params is None else packed_seq_params.cu_seqlens_cpu
        if (
            self.use_torch_linear
            and cu_seqlens is not None
            and cu_seqlens.numel() == 2
        ):
            cu_seqlens = None
            cu_seqlens_cpu = None
        hidden_states = hidden_states.transpose(0, 1).contiguous() # [S, B, D] -> [B, S, D]
        if attention_mask is not None:
            if attention_mask.dim() >= 3 and attention_mask.shape[2] > 1: # [B, 1, S, S]
                attention_mask = (~attention_mask).sum(dim=(1, 2)) > 0 # [B, S]
            else:
                attention_mask = ~(attention_mask.squeeze(1).squeeze(1)) # [B, S]
        hidden_states = self.apply_mask_to_padding_states(hidden_states, attention_mask)

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN does not support inference for now.")

        if self.projection_split_mode == "qwen3_5":
            qkv_weight, z_weight = self._split_qwen35_qkvz_weights()
            b_weight, a_weight = self._split_qwen35_ba_weights()
            qkv = F.linear(hidden_states, qkv_weight)
            if hasattr(self.in_proj_qkvz, "adapter_output"):
                adapter_output = self.in_proj_qkvz.adapter_output("qkv", hidden_states)
                if adapter_output is not None:
                    qkv = qkv + adapter_output.reshape(qkv.shape)

            z = F.linear(hidden_states, z_weight)
            if hasattr(self.in_proj_qkvz, "adapter_output"):
                adapter_output = self.in_proj_qkvz.adapter_output("z", hidden_states)
                if adapter_output is not None:
                    z = z + adapter_output.reshape(z.shape)

            beta = F.linear(hidden_states, b_weight)
            if hasattr(self.in_proj_ba, "adapter_output"):
                adapter_output = self.in_proj_ba.adapter_output("b", hidden_states)
                if adapter_output is not None:
                    beta = beta + adapter_output.reshape(beta.shape)

            alpha = F.linear(hidden_states, a_weight)
            if hasattr(self.in_proj_ba, "adapter_output"):
                adapter_output = self.in_proj_ba.adapter_output("a", hidden_states)
                if adapter_output is not None:
                    alpha = alpha + adapter_output.reshape(alpha.shape)
            z = z.reshape(hidden_states.shape[0], hidden_states.shape[1], -1, self.value_head_dim)
        else:
            projected_states_qkvz = self.in_proj_qkvz(hidden_states)
            projected_states_ba = self.in_proj_ba(hidden_states)

            query, key, value, z, beta, alpha = self.fix_query_key_value_ordering(
                projected_states_qkvz,
                projected_states_ba
            )
            query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))
            qkv = torch.cat((query, key, value), dim=-1)

        nvtx_range_push(suffix="conv1d")
        qkv = self.causal_conv1d(
            x=qkv,
            weight=self.conv1d.weight.squeeze(1),  # d, 1, w -> d, w
            bias=self.conv1d.bias,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )[0]
        nvtx_range_pop(suffix="conv1d")

        # Split qkv into query, key, and value
        query, key, value = torch.split(
            qkv,
            [
                self.qk_dim,
                self.qk_dim,
                self.v_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.key_head_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.key_head_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.value_head_dim)

        if self.num_value_heads // self.num_key_heads > 1:
            query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
            key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)

        # Calculate g and beta
        nvtx_range_push(suffix="g_and_beta")
        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)  # In fp32
        beta = beta.sigmoid()
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            cu_seqlens=cu_seqlens,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        # RMSNorm
        nvtx_range_push(suffix="gated_norm")
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        norm_out = self.out_norm(core_attn_out, z)
        nvtx_range_pop(suffix="gated_norm")

        norm_out = norm_out.reshape(z_shape_og)
        norm_out = norm_out.reshape(norm_out.shape[0], norm_out.shape[1], -1)

        # Output projection
        nvtx_range_push(suffix="out_proj")
        if self.use_torch_linear:
            out, _ = torch_linear_forward(self.out_proj, norm_out)
        else:
            out = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        out = out.transpose(0, 1).contiguous() # [B, S, D] -> [S, B, D]

        if self.cp_size > 1:
            out = scatter_to_context_parallel_region(
                out, packed_seq_params, self.pg_collection.cp
            )
        if self.sequence_parallel and self.tp_size > 1:
            out = reduce_scatter_to_sequence_parallel_region(out) / self.tp_size
        return out, None

    def backward_dw(self):
        """Execute weight gradient computation for all linear layers."""
        if self.use_torch_linear:
            return
        self._backward_in_proj_qkvz()
        self._backward_in_proj_ba()
        self._backward_out_proj()

    def _backward_in_proj_qkvz(self):
        """Computes weight gradients of in_proj_qkvz layer."""
        self.in_proj_qkvz.backward_dw()

    def _backward_in_proj_ba(self):
        """Computes weight gradients of in_proj_ba layer."""
        self.in_proj_ba.backward_dw()

    def _backward_out_proj(self):
        """Computes weight gradients of out_proj layer."""
        self.out_proj.backward_dw()
