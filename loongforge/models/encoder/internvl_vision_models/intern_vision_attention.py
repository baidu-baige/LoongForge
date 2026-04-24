# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

""" intern vision attention module """

import torch

from megatron.core.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import Attention, SelfAttentionSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.process_groups_config import ProcessGroupCollection

try:
    import transformer_engine

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


class InternViTRMSNorm(MegatronModule):
    """ InternViTRMSNorm for InternViT-6B qk_layernorm """

    def __init__(
        self,
        config,
        hidden_size: int,
        eps: float = 1e-6,
        sequence_parallel: bool = False,
        compute_var: bool = False,
    ):
        """Custom RMSNorm for InternViT.

        Args:
            config (TransformerConfig): Config.
            hidden_size (int): Input hidden size.
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
            compute_var (bool): Indicator to compute statistic manually.
        """
        super().__init__(config=config)
        self.config = config
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self._compute_var = compute_var

        assert not sequence_parallel, "Sequence parallelism is not supported with InternViT."

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x, var):
        if var is None:
            var = x.pow(2).mean(-1, keepdim=True)

        return x * torch.rsqrt(var + self.eps)

    def forward(self, x):
        """Run RMSNorm with an option to compute custom statistic."""
        var = None
        if self._compute_var:
            unpadded_hidden_size = self.config.hidden_size  # 3200
            max_dim = x.shape[-1]  # 128

            x = x.reshape(x.size(0), x.size(1), -1)
            var = self._gather_var(x.float().pow(2), max_dim) / unpadded_hidden_size

        output = self._norm(x.float(), var).type_as(x)
        output = output * self.weight

        if self._compute_var:
            output = output.reshape(output.size(0), output.size(1), -1, max_dim)

        return output

    def _gather_var(self, input_, max_dim):
        """Compute statistic across the non-dummy heads."""
        world_size = get_tensor_model_parallel_world_size()

        # Size and dimension.
        last_dim = input_.dim() - 1
        rank = get_tensor_model_parallel_rank()

        num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        valid_ranks = (self.config.original_num_attention_heads - 1) // num_attention_heads_per_partition

        residual_heads = self.config.original_num_attention_heads % num_attention_heads_per_partition
        if residual_heads == 0:
            residual_heads = num_attention_heads_per_partition
        max_dim = max_dim * residual_heads

        if rank < valid_ranks:  # Ranks without any dummy attention heads.
            var = input_.sum(-1, keepdim=True)
        elif rank == valid_ranks:  # The only rank which may contain 'residual_heads' dummy attention heads.
            var = input_[..., :max_dim].sum(-1, keepdim=True)
        else:
            var = input_.sum(-1, keepdim=True) * 0.0  # All heads in these ranks are dummy heads: Zero-out.

        tensor_list = [torch.empty_like(var) for _ in range(world_size)]
        tensor_list[rank] = var
        torch.distributed.all_gather(tensor_list, var, group=get_tensor_model_parallel_group())

        output = torch.cat(tensor_list, dim=last_dim).contiguous()

        return output.sum(-1, keepdim=True)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata={}):
        """ overwrite sharded_state_dict """
        # in InternVitSelfAttention the q_layernorm and k_layernorm weights
        # are tensor-parallel so must be converted to sharded tensors
        if 'q_layernorm' in prefix or 'k_layernorm' in prefix:
            state_dict = self.state_dict(prefix='', keep_vars=True)
            return make_sharded_tensors_for_checkpoint(
                state_dict, prefix, {'weight': 0}, sharded_offsets
            )
        else:
            return super().sharded_state_dict(prefix, sharded_offsets, metadata)


# Override a few things that are special in InternViT and not supported by the SelfAttention class.
class InternSelfAttention(Attention):
    """
    SelfAttention for InternVision
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        self.config = config
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear and self.config.add_qkv_bias,  # `or` modified to `and`
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        qk_layernorm_hidden_size = (
            self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        )  # 512 for internvit-6b and TP8

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=qk_layernorm_hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
                compute_var=True,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=qk_layernorm_hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
                compute_var=True,
            )
        else:
            self.k_layernorm = None


    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        #hidden_states = hidden_states.transpose(0, 1).contiguous()
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        if self.config.sequence_parallel:
            mixed_qkv = mixed_qkv.transpose(0, 1).contiguous()
        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            ((self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2) *
             self.hidden_size_per_attention_head),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition *
             self.hidden_size_per_attention_head),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:
            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(
                mixed_qkv,
                3,
                split_arg_list,
            )
        else:
            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(
                mixed_qkv,
                split_arg_list,
                dim=3,
            )

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ):
        """
        Perform a forward pass through the attention module.
        """
        # hidden_states: [sq, b, h]
        if self.config.flash_decode:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.

        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        if (
            self.config.flash_decode
            and inference_params is not None
            and inference_params.decode_mode
        ):
            assert self.layer_number in inference_params.key_value_memory_dict
            assert inference_params.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[
                self.layer_number
            ]
            output = self.flash_decoding(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        query, key, value, rotary_pos_emb, attn_mask_type, _ = self._adjust_key_value_for_inference(
            inference_params,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            assert self.apply_rotary_fn is not None, "apply_rotary_fn must be defined"
            query = self.apply_rotary_fn(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            key = self.apply_rotary_fn(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = self.apply_rotary_fn(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================
        if self.config.sequence_parallel:
            core_attn_out = core_attn_out.transpose(0, 1).contiguous()

        output, bias = self.linear_proj(core_attn_out)

        return output, bias


class InternViTTEDotProductAttention(TEDotProductAttention):
    """Adjusted Attention for InternViT"""

    def forward(self, *args, **kwargs):
        """Regular TEDotProductAttention + zero-out dummy attention heads."""
        out = super().forward(*args, **kwargs)

        if self.config.original_num_attention_heads is None:
            return out

        # This makes sure the dummy attention heads are zeroed out.
        mask = torch.ones_like(out, dtype=out.dtype, device=out.device)
        rank = get_tensor_model_parallel_rank()
        hidden_dim = out.shape[-1]
        
        world_size = get_tensor_model_parallel_world_size()
        num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        valid_ranks = (self.config.original_num_attention_heads - 1) // num_attention_heads_per_partition
        max_dim = hidden_dim // num_attention_heads_per_partition
        residual_heads = self.config.original_num_attention_heads % num_attention_heads_per_partition
        if residual_heads == 0:
            residual_heads = num_attention_heads_per_partition
        max_dim = max_dim * residual_heads
        # valid_ranks = 6

        if rank == valid_ranks:
            mask[..., max_dim:] *= 0.0
        elif rank > valid_ranks:
            mask *= 0.0
        out = out * mask

        return out
