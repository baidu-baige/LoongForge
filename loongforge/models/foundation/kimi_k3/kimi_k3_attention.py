# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Megatron-Bridge under the Apache-2.0 License:
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/kimi/kimi_k3_layers.py

"""Kimi K3 KDA and no-RoPE MLA attention implementations."""

import copy

import torch
from einops import rearrange
from fla.modules import FusedRMSNormGated, ShortConvolution
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELinear,
    TERowParallelLinear,
)
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)
from torch import nn

from .kimi_k3_ops import KimiRMSNorm, kda, sum_grads_across_tp


def _linear(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output, bias = module(inputs)
    if bias is not None:
        raise ValueError(
            "Kimi K3 requires bias-free projections, "
            f"got bias from {type(module).__name__}"
        )
    return output


class KimiK3ShortConvolution(ShortConvolution):
    """KDA short convolution, TP-sharded along the channel dimension.

    The weight starts in FP32 for checkpoint compatibility and follows the model
    dtype when ``Float16Module`` wraps it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight.data = self.weight.data.float()
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """Return the TP-sharded convolution state."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0},
            sharded_offsets,
            dp_cp_group=metadata["dp_cp_group"],
        )


class KimiK3Attention(MegatronModule):
    """Select KDA or Kimi's no-RoPE MLA by global layer number."""

    def __init__(
        self,
        config,
        layer_number: int,
        cp_comm_type: str | None = None,
        pg_collection=None,
        pp_layer_offset: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        del pp_layer_offset, name
        self.cp_comm_type = cp_comm_type

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "cp"]
            )
        elif not hasattr(pg_collection, "tp") or not hasattr(pg_collection, "cp"):
            raise ValueError("KimiK3Attention requires TP and CP process groups")
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.tp_size = self.tp_group.size()
        self.cp_group = pg_collection.cp
        self.cp_size = self.cp_group.size()
        self.sequence_parallel = config.sequence_parallel
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False

        self.layer_idx = layer_number - 1
        self.is_kda = layer_number in config.kimi_kda_layers
        if self.is_kda and self.cp_size > 1:
            raise ValueError("Kimi K3 KDA context parallelism is not supported yet")
        if self.is_kda:
            self._init_kda(config)
        else:
            self._init_mla(config)

    def _duplicated_linear(self, input_size: int, output_size: int) -> TELinear:
        return TELinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _column_linear(self, input_size: int, output_size: int) -> TEColumnParallelLinear:
        return TEColumnParallelLinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def _row_linear(self, input_size: int, output_size: int) -> TERowParallelLinear:
        return TERowParallelLinear(
            input_size,
            output_size,
            config=self.linear_config,
            init_method=self.config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )

    def _init_kda(self, config) -> None:
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.kimi_linear_num_heads
        if self.num_heads % self.tp_size:
            raise ValueError(
                f"KDA heads {self.num_heads} must be divisible by TP size {self.tp_size}"
            )
        self.local_num_heads = self.num_heads // self.tp_size
        self.head_dim = config.kimi_linear_head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.local_projection_size = self.local_num_heads * self.head_dim

        self.q_proj = self._column_linear(hidden_size, self.projection_size)
        self.k_proj = self._column_linear(hidden_size, self.projection_size)
        self.v_proj = self._column_linear(hidden_size, self.projection_size)
        conv_kwargs = {
            "hidden_size": self.local_projection_size,
            "kernel_size": config.kimi_linear_conv_kernel_size,
            "activation": "silu",
            "device": device,
            "dtype": dtype,
        }
        self.q_conv1d = KimiK3ShortConvolution(**conv_kwargs)
        self.k_conv1d = KimiK3ShortConvolution(**conv_kwargs)
        self.v_conv1d = KimiK3ShortConvolution(**conv_kwargs)

        self.f_a_proj = self._duplicated_linear(hidden_size, self.head_dim)
        self.f_b_proj = self._column_linear(self.head_dim, self.projection_size)
        self.b_proj = self._column_linear(hidden_size, self.num_heads)
        self.g_proj = self._column_linear(hidden_size, self.projection_size)

        # Stored in FP32 for checkpoint compatibility, then cast with the model.
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32, device=device)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32, device=device)
        )
        set_tensor_model_parallel_attributes(self.A_log, True, 0, 1)
        set_tensor_model_parallel_attributes(self.dt_bias, True, 0, 1)

        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.layernorm_epsilon,
            activation="sigmoid",
            device=device,
            dtype=dtype,
        )
        sum_grads_across_tp(self.o_norm)
        self.o_proj = self._row_linear(self.projection_size, hidden_size)
        self.gate_lower_bound = config.kimi_kda_gate_lower_bound

    def _init_mla(self, config) -> None:
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()
        dtype = config.params_dtype
        self.num_heads = config.num_attention_heads
        if self.num_heads % self.tp_size:
            raise ValueError(
                f"MLA heads {self.num_heads} must be divisible by TP size {self.tp_size}"
            )
        self.local_num_heads = self.num_heads // self.tp_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_head_dim
        self.qk_extra_head_dim = config.qk_pos_emb_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_extra_head_dim

        self.q_a_proj = self._duplicated_linear(hidden_size, self.q_lora_rank)
        self.q_a_layernorm = KimiRMSNorm(
            self.q_lora_rank, config.layernorm_epsilon, device=device, dtype=dtype
        )
        self.q_b_proj = self._column_linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim
        )
        self.kv_a_proj_with_mqa = self._duplicated_linear(
            hidden_size, self.kv_lora_rank + self.qk_extra_head_dim
        )
        self.kv_a_layernorm = KimiRMSNorm(
            self.kv_lora_rank, config.layernorm_epsilon, device=device, dtype=dtype
        )
        self.kv_b_proj = self._column_linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        self.g_proj = self._column_linear(
            hidden_size, self.num_heads * self.v_head_dim
        )
        self.o_proj = self._row_linear(
            self.num_heads * self.v_head_dim, hidden_size
        )

        self.core_attention = TEDotProductAttention(
            config=self.config,
            layer_number=self.layer_idx + 1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=self.q_head_dim**-0.5,
            k_channels=self.q_head_dim,
            v_channels=self.v_head_dim,
            cp_comm_type=self.cp_comm_type,
            pg_collection=self.pg_collection,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """Return attention state with explicit KDA TP sharding."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded_state_dict = super().sharded_state_dict(
            prefix, sharded_offsets, metadata
        )
        if not self.is_kda:
            return sharded_state_dict

        sharded_state_dict.update(
            make_sharded_tensors_for_checkpoint(
                {"A_log": self.A_log, "dt_bias": self.dt_bias},
                prefix,
                {"A_log": 0, "dt_bias": 0},
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return sharded_state_dict

    def _forward_kda(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        x = hidden_states.transpose(0, 1)
        cu_seqlens = (
            packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
        )
        if packed_seq_params is not None and cu_seqlens is None:
            raise ValueError("Packed KDA input requires cu_seqlens_q")

        conv_kwargs = {"output_final_state": False, "cu_seqlens": cu_seqlens}
        q, _ = self.q_conv1d(x=_linear(self.q_proj, x), **conv_kwargs)
        k, _ = self.k_conv1d(x=_linear(self.k_proj, x), **conv_kwargs)
        v, _ = self.v_conv1d(x=_linear(self.v_proj, x), **conv_kwargs)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.local_num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.local_num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.local_num_heads)
        forget_gate = rearrange(
            _linear(self.f_b_proj, _linear(self.f_a_proj, x)),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        beta = _linear(self.b_proj, x).float().sigmoid()
        output = kda(
            q,
            k,
            v,
            forget_gate,
            beta,
            self.A_log,
            self.dt_bias,
            self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
        )
        gate = rearrange(
            _linear(self.g_proj, x),
            "b s (h d) -> b s h d",
            h=self.local_num_heads,
        )
        output = self.o_norm(
            output.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim)
        )
        output = output.view(*gate.shape).flatten(-2)
        return _linear(self.o_proj, output.to(hidden_states.dtype)).transpose(0, 1)

    def _forward_mla(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        x = hidden_states
        query = _linear(self.q_b_proj, self.q_a_layernorm(_linear(self.q_a_proj, x)))
        query = query.view(*query.shape[:-1], self.local_num_heads, self.q_head_dim)

        compressed_kv = _linear(self.kv_a_proj_with_mqa, x)
        kv_latent, key_extra = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_extra_head_dim],
            dim=-1,
        )
        key_value = _linear(self.kv_b_proj, self.kv_a_layernorm(kv_latent))
        key_value = key_value.view(
            *key_value.shape[:-1],
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_nope, value = torch.split(
            key_value,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        value = value.contiguous()
        key_extra = copy_to_tensor_model_parallel_region(key_extra, group=self.tp_group)
        key_extra = key_extra.unsqueeze(-2).expand(*key_nope.shape[:-1], -1)
        key = torch.cat((key_nope, key_extra), dim=-1)

        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if is_thd:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        output = self.core_attention(
            query,
            key,
            value,
            None,
            packed_seq_params=packed_seq_params,
            attn_mask_type=AttnMaskType.causal,
        )
        if is_thd:
            output = output.reshape(output.size(0), 1, -1)
        output = output * torch.sigmoid(_linear(self.g_proj, x))
        return _linear(self.o_proj, output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Run the attention implementation selected for this layer."""
        del (
            attention_mask,
            key_value_states,
            inference_context,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            sequence_len_offset,
            kwargs,
        )
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=self.tp_group,
            )
        output = (
            self._forward_kda(hidden_states, packed_seq_params)
            if self.is_kda
            else self._forward_mla(hidden_states, packed_seq_params)
        )
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output, None


__all__ = ["KimiK3Attention", "KimiK3ShortConvolution"]
