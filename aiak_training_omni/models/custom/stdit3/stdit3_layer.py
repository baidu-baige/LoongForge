# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""tranformer layer."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
from einops import rearrange

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import BaseTransformerLayer, get_transformer_layer_offset
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import make_viewless_tensor

from aiak_training_omni.models.custom.transformer.vision.stdit_model_embedding import t2i_modulate
from aiak_training_omni.models.custom.transformer.vision.stdit_transformer_config import StditTransformerConfig


@dataclass
class STDiT3BlockSubmodules:
    """Submodules for a stdit3 block."""
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class STDiT3LayerSubmodules:
    """Submodules for a transformer layer."""
    spatial_block: Union[ModuleSpec, type] = IdentityOp
    temporal_block: Union[ModuleSpec, type] = IdentityOp
    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class LlamaRMSNorm(torch.nn.Module):
    """RMS normalization layer from Llama model."""
    def __init__(self, config: StditTransformerConfig, hidden_size, **kwargs):
        """
        copy from transformers/src/transformers/models/llama/modeling_llama.py
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        """Apply RMS normalization to hidden states."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class STDiT3Block(MegatronModule):
    """A single transformer block."""
    def __init__(
        self,
        config: StditTransformerConfig,
        submodules: STDiT3BlockSubmodules,
        attention_shape: str,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config)

        self.submodules_config = submodules
        self.attention_shape = attention_shape
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.scale_shift_table = torch.nn.Parameter(torch.randn(6, config.hidden_size) / config.hidden_size**0.5)

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        ## [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention, config=self.config, layer_number=layer_number,
        )
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        ## [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention, config=self.config, layer_number=layer_number)

        ## [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # ## [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # ## [Module 8: MLP block]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        hidden_states,
        T,
        S,
        attention_mask=None,
        attn_mask_type=None,
        context=None,
        context_mask=None,
        timestep=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ):
        """ hidden_states: [ts, b, h] """
        TS, B, H = hidden_states.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[: , None] + timestep.reshape(6, B, -1)
        ).chunk(6)

        # Spatial Self Attention.
        residual = hidden_states
        hidden_states = t2i_modulate(self.input_layernorm(hidden_states), shift_msa, scale_msa)
        hidden_states = rearrange(hidden_states, f"(T S) B C -> {self.attention_shape}", T=T, S=S).contiguous()

        attention_output, attention_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.padding if attention_mask.any() else AttnMaskType.no_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        attention_output = rearrange(attention_output, f"{self.attention_shape} -> (T S) B C",
                T=T, S=S).contiguous()

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                (attention_output, attention_bias), gate_msa, residual, self.hidden_dropout
            )

        # Cross attention.
        residual = hidden_states
        attention_output_with_bias = self.cross_attention(
            hidden_states,
            attention_mask=context_mask,
            attn_mask_type=AttnMaskType.padding if context_mask[0].any() or context_mask[1].any()
                    else AttnMaskType.no_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, None, residual, self.hidden_dropout
            )

        # MLP.
        residual = hidden_states
        pre_mlp_layernorm_output = t2i_modulate(self.pre_mlp_layernorm(hidden_states), shift_mlp, scale_mlp)
        mlp_output, mlp_bias = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                (mlp_output, mlp_bias), gate_mlp, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output, context

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """sharded state dict"""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict


class STDiT3Layer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: StditTransformerConfig,
        submodules: STDiT3LayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config)
        self.spatial_block = build_module(
            submodules.spatial_block,
            config=config,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            attention_shape="S (B T) C",
        )
        self.temporal_block = build_module(
            submodules.temporal_block,
            config=config,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            attention_shape="T (B S) C",
        )

    def forward(
        self,
        hidden_states,
        T,
        S,
        context=None,
        rotary_pos_emb=None,
        s_attn_mask=None,
        t_attn_mask=None,
        attention_mask=None,
        **kwargs
    ):
        """ hidden_states: [TS, B, H] """
        hidden_states, context = self.spatial_block(
            hidden_states,
            T,
            S,
            attention_mask=s_attn_mask,
            context=context,
            rotary_pos_emb=None,
            **kwargs
        )
        hidden_states, context = self.temporal_block(
            hidden_states,
            T,
            S,
            attention_mask=t_attn_mask,
            context=context,
            rotary_pos_emb=rotary_pos_emb,
            **kwargs
        )
        return hidden_states, context

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """sharded state dict"""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
