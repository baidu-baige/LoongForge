"""tranformer layer for aiak."""

from dataclasses import dataclass
from typing import Union

from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from megatron.core.utils import make_viewless_tensor


@dataclass
class TransformerLayerCogvlmSubmodules(TransformerLayerSubmodules):
    """
    TransformerLayerCogvlmSubmodules:
    Args:
        mlp (Union[ModuleSpec, type], optional): Specification or type of the MLP module. Defaults to IdentityOp.
        mlp_dense (Union[ModuleSpec, type], optional): Specification or type of the dense MLP module.
            Defaults to IdentityOp.
        moe_mlp (Union[ModuleSpec, type], optional): Specification or type of the MLP module for MoE.
            Defaults to IdentityOp.
    """

    post_self_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    post_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    post_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp


class TransformerLayerCogvlm(TransformerLayer):
    """A single transformer layer for CogVLM.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerCogvlmSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        **kwargs,
    ):
        """
            Initializes the TransformerLayerDeepSeek module.

        Args:
            config (TransformerConfig): The configuration of the model.
            submodules (TransformerLayerDeepSeekSubmodules): The submodules of the model.
            layer_number (int, optional): The number of the current layer. Defaults to 1.
            hidden_dropout (float, optional): The dropout probability for the hidden state. Defaults to None.
            **kwargs (dict, optional): Additional keyword arguments passed to the parent class. Defaults to {}.

        Raises:
            ValueError: If the `submodules` argument is not a valid type.
        """
        super(TransformerLayerCogvlm, self).__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            **kwargs,
        )

        self.post_self_attn_layernorm = build_module(
            submodules.post_self_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.post_cross_attn_layernorm = build_module(
            submodules.post_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.post_mlp_layernorm = build_module(
            submodules.post_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        attn_mask_type=None,
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
        """
        Override the forward method of the TransformerLayer class.
        """
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )

        if self.post_self_attn_layernorm is not None and not isinstance(
            self.post_self_attn_layernorm, IdentityOp
        ):
            attention_output_with_bias = (
                self.post_self_attn_layernorm(
                    attention_output_with_bias[0] + attention_output_with_bias[1]
                ),
                None,
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            attn_mask_type=attn_mask_type,
            key_value_states=context,
            inference_params=inference_params,
            **kwargs,
        )

        if self.post_cross_attn_layernorm is not None and not isinstance(
            self.post_cross_attn_layernorm, IdentityOp
        ):
            attention_output_with_bias = (
                self.post_cross_attn_layernorm(
                    attention_output_with_bias[0] + attention_output_with_bias[1]
                ),
                None,
            )

        if (
            isinstance(attention_output_with_bias, dict)
            and "context" in attention_output_with_bias
        ):
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, **kwargs)

        if self.post_mlp_layernorm is not None and not isinstance(
            self.post_mlp_layernorm, IdentityOp
        ):
            mlp_output_with_bias = (
                self.post_mlp_layernorm(
                    mlp_output_with_bias[0] + mlp_output_with_bias[1]
                ),
                None,
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(
                self.training, self.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, self.hidden_dropout)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )

        return output, context
