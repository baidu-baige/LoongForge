"""mtp tranformer layer for deepseek"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Union, Literal, Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context

from aiak_training_omni.models.foundation import DeepseekConfig


@dataclass
class MultiTokenPredLayerDeepSeekSubmodules(TransformerLayerSubmodules):
    """
    MultiTokenPredLayerDeepSeekSubmodules:
    Args:
        eh_proj (Union[ModuleSpec, type], optional): Specification or type of the proj layer. Defaults to IdentityOp.
        hnorm (Union[ModuleSpec, type], optional): Specification or type of the norm layer on hidden states.
            Defaults to IdentityOp.
        enorm (Union[ModuleSpec, type], optional): Specification or type of the norm layer on residual embedding.
            Defaults to IdentityOp.
        output_layernorm (Union[ModuleSpec, type], optional): Specification or type of the output head pre-layernorm.
            Defaults to IdentityOp.
    """

    eh_proj: Union[ModuleSpec, type] = IdentityOp
    hnorm: Union[ModuleSpec, type] = IdentityOp
    enorm: Union[ModuleSpec, type] = IdentityOp
    output_layernorm: Union[ModuleSpec, type] = IdentityOp


class MultiTokenPredLayerDeepSeek(TransformerLayer):
    """
    A MultiTokenPredLayerDeepSeek inherit from TransformerLayerDeepSeek. But invokes
        the following changes:
            1. Adds an additional projection layer between the input embeddings and the hidden states.
            2. Adds an additional normalization layer before the projection layer.
            3. Adds an additional normalization layer after the projection layer.
            4. Adds an additional output projection layer. i.e. output_head.
    """

    def __init__(
        self,
        config: DeepseekConfig,
        submodules: MultiTokenPredLayerDeepSeekSubmodules,
        # Params for MTP embedding layer
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal[
            "learned_absolute", "rope"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        parallel_output: bool = True,
        # Params for MTP layer
        share_mtp_embeddings_and_output_weights: bool = True,
        # Params for TransformerLayer
        layer_number: int = 1,
        hidden_dropout: float = None,
        # Params for parallel
        pre_process: bool = True,
        post_process: bool = True,
        **kwargs,
    ):

        # Build the TransformerLayer in FP8 context
        with get_fp8_context(config, is_init=True):
            # The TransformerLayer constructor
            super(MultiTokenPredLayerDeepSeek, self).__init__(
                config=config,
                submodules=submodules,
                layer_number=layer_number,
                hidden_dropout=hidden_dropout,
                **kwargs,
            )

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.share_mtp_embeddings_and_output_weights = (
            share_mtp_embeddings_and_output_weights
        )
        self.position_embedding_type = position_embedding_type

        self.pre_process = pre_process
        self.post_process = post_process

        # Whether use FP8 mixed precision in TransformerLayer forward
        self.use_inner_fp8_context = (
            self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        )

        # The embedding section
        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=position_embedding_type,
            # TODO: add `skip_weight_param_allocation` after megatron-core update
        )

        if position_embedding_type == "rope" and not self.config.multi_latent_attention:
            # not used for mla
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )

        # Setup the projection layer
        self.eh_proj = build_module(
            submodules.eh_proj,
            self.config.hidden_size * 2,
            self.config.hidden_size,
            parallel_mode="duplicated",
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="eh",
            skip_weight_param_allocation=False,
        )

        # Setup the embedding norm
        self.enorm = build_module(
            submodules.enorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # Setup the hidden_states norm
        self.hnorm = build_module(
            submodules.hnorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # Setup the layernorm before the output head layer, notably,
        # the layernorm is differ from the main model and not shared.
        if submodules.output_layernorm != IdentityOp:
            self.output_layernorm = build_module(
                submodules.output_layernorm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.output_layernorm = None

        if self.config.defer_embedding_wgrad_compute:
            self.embedding_activation_buffer = []
            self.grad_output_buffer = []
        else:
            self.embedding_activation_buffer = None
            self.grad_output_buffer = None

        # Setup the output head layer
        self.output_head = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            self.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not parallel_output,
            skip_weight_param_allocation=self.share_mtp_embeddings_and_output_weights,
            embedding_activation_buffer=self.embedding_activation_buffer,
            grad_output_buffer=self.grad_output_buffer,
        )

    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        """Computes the language model loss (Cross entropy across vocabulary)

        Args:
            labels (Tensor): The labels of dimension [batch size, seq length]
            logits (Tensor): The final logits returned by the output layer of the transformer model

        Returns:
            Tensor: Loss tensor of dimensions [batch size, sequence_length]
        """
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        if self.config.cross_entropy_loss_fusion:
            loss = fused_vocab_parallel_cross_entropy(logits, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss

    def forward(
        self,
        hidden_states,
        attention_mask,
        input_ids=None,
        decoder_input=None,
        attn_mask_type=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        position_ids=None,
        labels=None,
        inference_params=None,
        packed_seq_params=None,
        embed_weight=None,
        output_weight=None,
    ):
        """
        Forward function of MultiTokenPredLayerDeepSeek
        """
        # Main model's embedding layer and MTP modules are in same PP stage.
        if self.pre_process and self.post_process:
            # Use the pass in `decoder_input` from the main model directly.
            assert decoder_input is not None
        else:
            decoder_input = self.embedding(
                input_ids=input_ids,
                position_ids=position_ids,
                # TODO: weight=embed_weight, currently not supported
            )

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if (
            self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
        ):
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params,
                self.decoder,
                decoder_input,
                self.config,
                packed_seq_params,
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
            )

        with torch.cuda.nvtx.range("MTP_layer_forward"):
            # Prepare the input embeddings and hidden states for the MTP layer
            enorm_output = self.enorm(decoder_input)
            hnorm_output = self.hnorm(hidden_states)
            hidden_states = torch.cat([enorm_output, hnorm_output], dim=-1)
            hidden_states, _ = self.eh_proj(hidden_states)

            # Only the transformer layer is warpped in FP8 context
            inner_fp8_context = (
                get_fp8_context(self.config)
                if self.use_inner_fp8_context
                else nullcontext()
            )
            with inner_fp8_context:
                # Forward pass through the MTP layer's TransformerLayer
                hidden_states, _ = super(MultiTokenPredLayerDeepSeek, self).forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    attn_mask_type=attn_mask_type,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                )

            # Output Head block: output_layernorm and output_head
            if self.output_layernorm is not None:
                output_layernorm_output = self.output_layernorm(hidden_states)
            else:
                output_layernorm_output = hidden_states

            logits, _ = self.output_head(output_layernorm_output, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return hidden_states, loss

    def _submodule_eh_proj_dw(self):
        """backward_dw for attention in eh_proj"""
        self.eh_proj.backward_dw()
