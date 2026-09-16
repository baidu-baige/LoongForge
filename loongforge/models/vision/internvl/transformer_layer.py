# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""tranformer layer"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor


class LayerScale(nn.Module):
    """ LayerScale """

    def __init__(self, embed_dim, initializer_factor=1.0, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(initializer_factor * torch.ones(embed_dim))

    def forward(self, x):
        """ forward """
        return x.mul_(self.weight) if self.inplace else x * self.weight


@dataclass
class TransformerLayerInternVisionSubmodules(TransformerLayerSubmodules):
    """
    TransformerLayerInternVisionSubmodules for Intern Vision model:
    Args:
        mlp (Union[ModuleSpec, type], optional): Specification or type of the MLP module. Defaults to IdentityOp.
        mlp_dense (Union[ModuleSpec, type], optional): Specification or type of the dense MLP module.
            Defaults to IdentityOp.
        moe_mlp (Union[ModuleSpec, type], optional): Specification or type of the MLP module for MoE.
            Defaults to IdentityOp.
    """
    post_attention_layerscale: nn.Parameter = LayerScale
    post_mlp_layerscale: nn.Parameter = LayerScale


class TransformerLayerIntern(TransformerLayer):
    """A single transformer layer for Intern Vision model.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerInternVisionSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        **kwargs,
    ):
        """
            Initializes the TransformerLayer module.
        
        Args:
            config (TransformerConfig): The configuration of the model.
            submodules (TransformerLayerDeepSeekSubmodules): The submodules of the model.
            layer_number (int, optional): The number of the current layer. Defaults to 1.
            hidden_dropout (float, optional): The dropout probability for the hidden state. Defaults to None.
            **kwargs (dict, optional): Additional keyword arguments passed to the parent class. Defaults to {}.
        
        Raises:
            ValueError: If the `submodules` argument is not a valid type.
        """
        super(TransformerLayerIntern, self).__init__(config=config,
                                                     submodules=submodules,
                                                     layer_number=layer_number,
                                                     hidden_dropout=hidden_dropout,
                                                     **kwargs)

        self.post_attention_layerscale = build_module(submodules.post_attention_layerscale, config.hidden_size,
                                                      config.initializer_factor)

        self.post_mlp_layerscale = build_module(submodules.post_mlp_layerscale, config.hidden_size,
                                                config.initializer_factor)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        **kwargs,
    ):
        """ intern vit transforemr layer forward """
        # hidden_states: [s, b, h]
        # Residual connection.
        residual = hidden_states
        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, attention_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output
            )

        attention_output_with_bias = self.post_attention_layerscale((
            attention_output + attention_bias) if attention_bias is not None else attention_output), None

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output, mlp_bias = self.mlp(pre_mlp_layernorm_output)

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output
            )

        mlp_output_with_bias = self.post_mlp_layerscale((mlp_output +
                                                         mlp_bias) if mlp_bias is not None else mlp_output), None

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
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

        # CUDA graph requires returned values to be Tensors
        if self.config.external_cuda_graph and self.training:
            return output

        return output, context
