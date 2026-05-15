# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Wan transformer block."""

from contextlib import nullcontext

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import te_checkpoint
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.transformer.transformer_block import TransformerBlock


class WanTransformerBlock(TransformerBlock):
    """"TransformerBlock for wan"""
    def _checkpointed_forward(
        self,
        hidden_states,
        attention_mask,
        context,
        context_mask,
        rotary_pos_emb,
        attention_bias,
        packed_seq_params,
        use_inner_quantization_context,
        **kwargs,
    ):
        timestep_mod = kwargs.pop("timestep_mod", None)
        if timestep_mod is None:
            return super()._checkpointed_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_inner_quantization_context=use_inner_quantization_context,
                **kwargs,
            )

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states, attention_mask, context, context_mask, rotary_pos_emb, timestep_mod
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)

                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                            timestep_mod=timestep_mod,
                            **kwargs,
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    timestep_mod,
                )
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                timestep_mod,
            )

        if self.config.enable_chunkpipe:
            for layer_idx in range(0, self.num_layers_per_pipeline_rank):
                hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
            return hidden_states

        if self.config.recompute_method == "uniform":
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self._recompute_num_layers)
                )
                layer_idx += self._recompute_num_layers
        elif self.config.recompute_method == "block":
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self._recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        timestep_mod,
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

