# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright 2026 AIAK Training Omni, All rights reserved.
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
#
# Some of this code was adopted from https://github.com/huggingface/transformers

"""GatedDeltaNet"""
import os
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import (
    deprecate_inference_params, 
    nvtx_range_pop, 
    nvtx_range_push,
)
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region, 
    scatter_to_sequence_parallel_region
)

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.modules import FusedRMSNormGated

    HAVE_FLA = True
except ImportError:
    chunk_gated_delta_rule = None
    FusedRMSNormGated = None

    HAVE_FLA = False

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


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
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


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


class GatedDeltaNet(MegatronModule):
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
        assert config.context_parallel_size == 1, "GatedDeltaNet currently does not support context parallelism."

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet"
        self.pg_collection = pg_collection
        self.tp_size = self.pg_collection.tp.size()
        self.sequence_parallel = config.sequence_parallel

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.unpacking_hidden_states_in_gdn = int(os.environ.get('UNPACKING_HIDDEN_STATES_IN_GDN', 1))

        # Input projection (hidden_states -> q, k, v, gate, beta, alpha)
        # TODO: for now, output gate is forced for GDN.
        # We may remove this restriction in the future.
        self.in_proj_qkvz_dim = self.qk_dim * 2 + self.v_dim * 2
        self.in_proj_ba_dim = self.num_value_heads * 2
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN layer "
                "input projection output tensor must be a multiple of 16."
            )
        self.in_proj_qkvz = nn.Linear(self.hidden_size, self.in_proj_qkvz_dim, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, self.in_proj_ba_dim, bias=False)

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
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        
        # dt_bias parameter
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.num_value_heads,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )

        # A_log parameter
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_value_heads,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        
        self.out_norm = ( 
            Qwen3NextRMSNormGated(
                self.value_head_dim, eps=self.config.layernorm_epsilon
            ) if FusedRMSNormGated is None 
            else FusedRMSNormGated(
                self.value_head_dim,
                eps=self.config.layernorm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=self.config.params_dtype,
            )
        )
        self.out_proj = nn.Linear(self.v_dim, self.hidden_size, bias=False)

        # TODO: support CP

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                # conv1d.weight
                if self.conv_init is not None:
                    nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
                # dt_bias
                torch.ones(
                    self.num_value_heads,
                    out=self.dt_bias.data,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                )
                # A_log
                A = torch.empty(
                    self.num_value_heads,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(torch.log(A))

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
    
    def apply_mask_to_padding_states(self, hidden_states, attention_mask):
        """
        Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
        """
        # NOTE: attention mask is a 2D boolean tensor
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        return hidden_states

    def unpacking_hidden_states(self, hidden_states, packed_seq_params):
        """
        Unpacks the hidden states from the packed format.
        """
        assert packed_seq_params.qkv_format == "thd", "Only support THD format."
        num_samples = len(packed_seq_params.cu_seqlens_q) - 1
        max_length = packed_seq_params.max_seqlen_q
        cu_seqlens_q = packed_seq_params.cu_seqlens_q
        
        new_hidden_states = hidden_states.new_zeros(
            (num_samples, max_length, hidden_states.shape[-1])
        )
        attention_mask = hidden_states.new_zeros(
            (num_samples, max_length), dtype=torch.bool
        )
        for i in range(num_samples):
            start, end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
            new_hidden_states[i, :end - start] = hidden_states[start:end, 0]
            attention_mask[i, :end - start] = True
        
        return new_hidden_states, attention_mask

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
        seq_length, batch, _ = hidden_states.shape
        
        if packed_seq_params is not None and self.unpacking_hidden_states_in_gdn:
            hidden_states, attention_mask = self.unpacking_hidden_states(
                hidden_states, packed_seq_params
            )
        else:
            hidden_states = hidden_states.transpose(0, 1)
            if attention_mask is not None:
                if attention_mask.shape[2] > 1:
                    attention_mask = attention_mask.sum(dim=(1, 3)) > 0
                else:
                    attention_mask = ~(attention_mask.squeeze(1).squeeze(1))
        hidden_states = self.apply_mask_to_padding_states(hidden_states, attention_mask)
        
        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN does not support inference for now.")

        # Input projection
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        
        query, key, value, z, beta, alpha = self.fix_query_key_value_ordering(
            projected_states_qkvz,
            projected_states_ba
        )
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        qkv = mixed_qkv.transpose(1, 2).contiguous()
        
        nvtx_range_push(suffix="conv1d")
        if (causal_conv1d_fn is None):
            qkv = self.act_fn(self.conv1d(qkv)[..., :hidden_states.shape[1]])
        else:
            assert self.activation in ["silu", "swish"]
            qkv = causal_conv1d_fn(
                x=qkv,
                weight=self.conv1d.weight.squeeze(1),  # d, 1, w -> d, w
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        nvtx_range_pop(suffix="conv1d")

        # Split qkv into query, key, and value
        qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
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
        if not HAVE_FLA:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        nvtx_range_pop(suffix="gated_delta_rule")
                    
        # RMSNorm
        nvtx_range_push(suffix="gated_norm")
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        norm_out = self.out_norm(core_attn_out, z)
        nvtx_range_pop(suffix="gated_norm")

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out = norm_out.reshape(z_shape_og)
        norm_out = norm_out.reshape(norm_out.shape[0], norm_out.shape[1], -1)
        
        # Output projection
        nvtx_range_push(suffix="out_proj")
        out = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")
        
        if packed_seq_params is not None:
            out = out[attention_mask][:, None]
            out = torch.concat([out, out.new_zeros(seq_length - out.shape[0], 1, out.shape[2])])
        else:
            out = out.transpose(0, 1).contiguous()

        if self.sequence_parallel and self.tp_size > 1:
            out = scatter_to_sequence_parallel_region(out)
        return out, None


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    # pylint: disable=line-too-long
    '''
    Reference: https://github.com/huggingface/transformers/blob/144c8ce2809a2e21914017652700e1ecb450501e/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L470-L547
    '''

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
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
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # chunk decay
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
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
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
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state