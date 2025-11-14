""" Fine-grained schedule for the model chunk."""

import contextlib
import weakref
from typing import Any, Callable, Optional, Tuple, Union, Dict

import torch
from torch import Tensor, LongTensor

from megatron.core import InferenceParams, tensor_parallel
from megatron.training import get_args
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.combined_1f1b import (
    AbstractSchedulePlan,
    ScheduleNode,
    get_com_stream,
    get_comp_stream,
    make_viewless,
)
from megatron.core.transformer import transformer_layer
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllPerBatchState, MoEFlexPerBatchState

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context


def weak_method(method):
    """ weak_method is used to avoid circular references in the schedule graph."""
    method_ref = weakref.WeakMethod(method)
    del method

    def wrapped_func(*args, **kwarg):
        # nonlocal object_ref
        return method_ref()(*args, **kwarg)

    return wrapped_func


class PreProcessNode(ScheduleNode):
    """ Preprocess node for the model chunk schedule plan."""
    def __init__(self, model, model_chunk_state, event, stream):
        super().__init__(weak_method(self.forward_impl), stream, event)
        self.model = model
        self.model_chunk_state = model_chunk_state
    
    def forward_impl(self):
        """ Preprocess the input for the model chunk schedule plan."""
        model = self.model
        input_ids = self.model_chunk_state.input_ids
        position_ids = self.model_chunk_state.position_ids
        inference_params = self.model_chunk_state.inference_params
        packed_seq_params = self.model_chunk_state.packed_seq_params
        image_inputs = self.model_chunk_state.image_inputs
        video_inputs = self.model_chunk_state.video_inputs
        audio_inputs = self.model_chunk_state.audio_inputs
        decoder_input = self.model_chunk_state.decoder_input
        
        has_encoder_model = hasattr(model, "encoder_model")
        combined_embeddings = None
        # if the model chunk has encoder model, we should first preprocess the encoder info
        if has_encoder_model:
            use_inference_kv_cache = (
                inference_params is not None
                and "image_tokens_count" in inference_params.key_value_memory_dict
            )
            if use_inference_kv_cache:
                vision_embeddings = None
            if model.add_encoder:
                combined_embeddings, decode_input = model.encoder_model(
                    input_ids=input_ids,
                    image_inputs=image_inputs,
                    video_inputs=video_inputs,
                    inference_params=inference_params,
                )

            decoder_input = combined_embeddings

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif model.foundation_model.pre_process:
            decoder_input = model.foundation_model.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            decoder_input = model.foundation_model.decoder.input_tensor

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if (
            rotary_pos_emb is None
            and model.foundation_model.position_embedding_type == "rope"
            and not model.foundation_model.config.multi_latent_attention
            and model.foundation_model.config.rotary_emb_func != "Qwen2VLRotaryEmbedding"
        ):
            rotary_seq_len = model.foundation_model.rotary_pos_emb.get_rotary_seq_len(
                inference_params,
                model.foundation_model.decoder,
                decoder_input,
                model.foundation_model.config,
                packed_seq_params,
            )
            rotary_pos_emb = model.foundation_model.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
            )
        else:
            rotary_pos_emb = (
                model.foundation_model.rotary_pos_emb(
                    position_ids,
                    packed_seq=packed_seq_params,
                )
                .transpose(0, 2)
                .contiguous()
            )

        #(gpt_model.config.enable_cuda_graph or gpt_model.config.flash_decode)
        if (
            (model.foundation_model.config.enable_cuda_graph)
            and rotary_pos_cos is not None
            and inference_params
        ):
            sequence_len_offset = torch.tensor(
                [inference_params.sequence_len_offset] * inference_params.current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None
        
        # saved for later use
        self.model_chunk_state.rotary_pos_emb = rotary_pos_emb
        self.model_chunk_state.rotary_pos_cos = rotary_pos_cos
        self.model_chunk_state.rotary_pos_sin = rotary_pos_sin
        self.model_chunk_state.sequence_len_offset = sequence_len_offset
        return decoder_input
        

class PostProcessNode(ScheduleNode):
    """ Postprocess node for the model chunk schedule plan."""
    def __init__(self, model, model_chunk_state, event, stream):
        super().__init__(weak_method(self.forward_impl), stream, event)
        self.model = model
        self.model_chunk_state = model_chunk_state

    def forward_impl(self, hidden_states):
        """ Postprocess the output of the model chunk schedule plan."""
        model = self.model
        labels = self.model_chunk_state.labels
        runtime_gather_output = self.model_chunk_state.runtime_gather_output
        # Final layer norm.
        if model.decoder.final_layernorm is not None:
            hidden_states = model.decoder.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = transformer_layer.make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )
        output_weight = None
        if model.share_embeddings_and_output_weights:
            output_weight = model.shared_embedding_or_output_weight()
        logits, _ = model.output_layer(
            hidden_states, weight=output_weight
            #hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        if labels is None:
            # [s b h] => [b s h]
            return float16_to_fp32(logits.transpose(0, 1).contiguous())
        loss = float16_to_fp32(
            model.compute_language_model_loss(labels, logits))

        return loss


class MtpPostProcessNode(ScheduleNode):
    """ MTP Post Process Node for the model chunk schedule plan."""

    def __init__(self, model, model_chunk_state, event, stream):
        super().__init__(weak_method(self.forward_impl), stream, event)
        self.model = model
        self.model_chunk_state = model_chunk_state
        self.layer_idx = 0

    def forward_impl(self, hidden_states, decoder_loss):
        """ Calculate the MTP loss."""
        model = self.model
        layer = self.model.mtp_layers[0]
        labels = self.model_chunk_state.labels_for_mtp
        layer_idx = self.layer_idx
        if not model.share_embeddings_and_output_weights and \
                model.share_mtp_embeddings_and_output_weights:
            output_weight = model.output_layer.weight.detach()
            output_weight.zero_out_wgrad = True

        # Output Head block: output_layernorm and output_head
        if layer.output_layernorm is not None:
            output_layernorm_output = layer.output_layernorm(hidden_states)
        else:
            output_layernorm_output = hidden_states

        logits, _ = layer.output_head(
            output_layernorm_output, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        mtp_loss = layer.compute_language_model_loss(labels, logits)
        mask = torch.ones_like(mtp_loss)
        mask[:, -(layer_idx + 1):] = 0.0
        mtp_loss = mtp_loss * mask

        assert mtp_loss is not None and decoder_loss is not None, \
            "mtp_loss and decoder_loss should not be None"
        total_loss = decoder_loss + mtp_loss * model.config.mtp_loss_coef / \
            model.config.num_nextn_predict_layers  # [b, s]

        self.model_chunk_state.rotary_pos_emb_mtp = None
        self.model_chunk_state.labels_for_mtp = None

        return total_loss

class TransformerLayerNode(ScheduleNode):
    """ Base class for transformer layer nodes in the schedule plan."""
    def __init__(self, chunk_state, common_state, layer, stream, event, free_inputs=False):
        super().__init__(
            weak_method(self.forward_impl),
            stream,
            event,
            weak_method(self.backward_impl),
            free_inputs=free_inputs,
        )
        # layer state
        self.common_state = common_state
        # model chunk state
        self.chunk_state = chunk_state
        self.layer = layer
        self.detached = tuple()
        self.before_detached = tuple()

        # Add fp8 context for the base class, so the subclass(DenseLinear and GroupedLinear) can use it
        use_fp8_context = self.layer.config.fp8 and self.layer.config.fp8_recipe != Fp8Recipe.delayed
        self.fp8_context = (
            get_fp8_context(self.layer.config, self.layer.layer_number - 1) if use_fp8_context
            else contextlib.nullcontext()
        )

    def detach(self, t):
        """ Detach the tensor and store it for later use."""
        detached = make_viewless(t).detach()
        detached.requires_grad = t.requires_grad
        self.before_detached = self.before_detached + (t,)
        self.detached = self.detached + (detached,)
        return detached

    def backward_impl(self, outputs, output_grad):
        """Implements the backward pass for the transformer layer node."""
        detached_grad = tuple([e.grad for e in self.detached])
        grads = output_grad + detached_grad
        self.default_backward_func(outputs + self.before_detached, grads)
        self.before_detached = None
        self.detached = None
        # return grads for record stream
        return grads


class MoeAttnNode(TransformerLayerNode):
    """ Attention node for the model chunk schedule plan."""

    def __init__(self, chunk_state, common_state, layer, stream, event, free_inputs=False, layer_idx=None,
                 is_mtp=False, model=None):
        super().__init__(
            chunk_state=chunk_state,
            common_state=common_state,
            layer=layer,
            stream=stream,
            event=event,
            free_inputs=free_inputs,
        )
        self.is_mtp = is_mtp
        self.model = model
        self.layer_idx = layer_idx

        self.use_recompute = layer.config.recompute_granularity == 'full' \
            and layer.config.recompute_method in ['block', 'uniform']
        if layer_idx is not None:
            self.use_recompute = self.use_recompute and layer_idx < layer.config.recompute_num_layers

    def submodule_mtp_attn_forward(self, hidden_states):
        """ Forward preprocess for the MTP attention submodule."""
        layer = self.layer
        model = self.model

        decoder_input = self.chunk_state.decoder_input
        position_ids = self.chunk_state.position_ids
        ori_input_ids = self.chunk_state.input_ids.detach()
        ori_labels = self.chunk_state.labels_for_mtp.detach()

        inference_params = self.chunk_state.inference_params
        packed_seq_params = self.chunk_state.packed_seq_params

        if self.layer_idx == 0:
            # Final layer norm.
            if model.post_process and model.decoder.final_layernorm is not None:
                hidden_states = model.decoder.final_layernorm(hidden_states)
                # TENorm produces a "viewed" tensor. This will result in schedule.py's
                # deallocate_output_tensor() throwing an error, so a viewless tensor is
                # created to prevent this.
                hidden_states = transformer_layer.make_viewless_tensor(
                    inp=hidden_states, requires_grad=True, keep_graph=True
                )

        # Shift right by `mtp_depth` and pad back to regular length
        mtp_input_ids = torch.nn.functional.pad(
            ori_input_ids[:, self.layer_idx + 1:],  # [b, s-mtp_depth]
            (0, self.layer_idx + 1), "constant", 0,  # [b, s]
        ).contiguous()

        mtp_labels = torch.nn.functional.pad(
            ori_labels[:, self.layer_idx + 1:],  # [b, s-mtp_depth]
            (0, self.layer_idx + 1), "constant", 0,  # [b, s]
        ).contiguous()

        if model.pre_process and model.post_process:
            decoder_input = torch.nn.functional.pad(
                decoder_input[self.layer_idx + 1:, ...],  # [s-mtp_depth, b, h]
                (0, 0, 0, 0, 0, self.layer_idx + 1), "constant", 0,  # [s, b, h]
            ).contiguous()

        # Main model's embedding layer and MTP modules are in same PP stage.
        if layer.pre_process and layer.post_process:
            # Use the pass in `decoder_input` from the main model directly.
            assert decoder_input is not None
        else:
            decoder_input = layer.embedding(
                input_ids=mtp_input_ids,
                position_ids=position_ids,
                # TODO: weight=embed_weight, currently not supported
            )

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if layer.position_embedding_type == 'rope' and not layer.config.multi_latent_attention:
            rotary_seq_len = layer.rotary_pos_emb.get_rotary_seq_len(
                inference_params,
                layer.decoder,
                decoder_input,
                layer.config,
                packed_seq_params,
            )
            rotary_pos_emb = layer.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == 'thd',
            )
        self.chunk_state.rotary_pos_emb_mtp = rotary_pos_emb
        self.chunk_state.labels_for_mtp = mtp_labels

        # Prepare the input embeddings and hidden states for the MTP layer
        enorm_output = layer.enorm(decoder_input)
        hnorm_output = layer.hnorm(hidden_states)
        hidden_states = torch.cat([enorm_output, hnorm_output], dim=-1)
        hidden_states, _ = layer.eh_proj(hidden_states)

        return hidden_states
    
    def _get_fp8_context(self, recompute=False):
        """Get the FP8 context for the current layer."""
        use_fp8 = self.layer.config.fp8 and self.layer.config.fp8_recipe != Fp8Recipe.delayed
        if not use_fp8:
            return contextlib.nullcontext()
        if recompute:
            return get_fp8_context(self.layer.config, self.layer.layer_number - 1)
        return self.fp8_context
    
    def forward_impl(self, hidden_states):
        """ Forward pass for the attention node."""
       
        if self.is_mtp:
            hidden_states = self.submodule_mtp_attn_forward(hidden_states)
        
        attention_mask = self.chunk_state.attention_mask
        attn_mask_type = self.chunk_state.attn_mask_type
        context = self.chunk_state.context
        context_mask = self.chunk_state.context_mask
        rotary_pos_emb = self.chunk_state.rotary_pos_emb \
            if not self.is_mtp else self.chunk_state.rotary_pos_emb_mtp
        rotary_pos_cos = self.chunk_state.rotary_pos_cos
        rotary_pos_sin = self.chunk_state.rotary_pos_sin
        attention_bias = self.chunk_state.attention_bias
        inference_params = self.chunk_state.inference_params
        packed_seq_params = self.chunk_state.packed_seq_params
        sequence_len_offset = self.chunk_state.sequence_len_offset

        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state), self.fp8_context:
            if self.use_recompute and not self.is_mtp:
                
                def custom_forward(*inputs):
                    hidden_states = inputs[0]
                    attention_mask = inputs[1]
                    attn_mask_type = inputs[2]
                    
                    from megatron.core.transformer.enums import AttnMaskType
                    if attn_mask_type is not None:
                        attn_mask_type = AttnMaskType(attn_mask_type.item())
                    
                    inference_params = inputs[3]
                    rotary_pos_emb = inputs[4]
                    rotary_pos_cos = inputs[5]
                    rotary_pos_sin = inputs[6]
                    attention_bias = inputs[7]
                    sequence_len_offset = inputs[8]

                    with self._get_fp8_context(recompute=self.use_recompute):
                        output_ = self.layer._submodule_attention_forward(
                            hidden_states,
                            attention_mask=attention_mask,
                            attn_mask_type=attn_mask_type,
                            inference_params=inference_params,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )
                    return output_

                if attn_mask_type is not None:
                    _attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
                else:
                    _attn_mask_type = None

                hidden_states = tensor_parallel.checkpoint(
                        custom_forward, False, hidden_states,
                        attention_mask, _attn_mask_type, inference_params, rotary_pos_emb, rotary_pos_cos,
                        rotary_pos_sin, attention_bias, sequence_len_offset)
            else:
                hidden_states = self.layer._submodule_attention_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    attn_mask_type=attn_mask_type,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )
        return hidden_states

    def dw(self):
        """ Backward pass for the attention node."""
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_attention_router_compound_dw()
            if self.is_mtp:
                # for mtp, we need to backward_dw eh_proj
                self.layer._submodule_eh_proj_dw()


class MoePostAttnNode(TransformerLayerNode):
    """ Post Attention node for the model chunk schedule plan."""

    def __init__(self, chunk_state, common_state, layer, stream, event, free_inputs=False, layer_idx=None,
                 is_mtp=False):
        super().__init__(
            chunk_state=chunk_state,
            common_state=common_state,
            layer=layer,
            stream=stream,
            event=event,
            free_inputs=free_inputs,
        )
        self.layer_idx = layer_idx
        self.is_mtp = is_mtp
        self.use_recompute = layer.config.recompute_granularity == 'full' \
            and layer.config.recompute_method in ['block', 'uniform']
        if layer_idx is not None:
            self.use_recompute = self.use_recompute and layer_idx < layer.config.recompute_num_layers

    def _get_fp8_context(self, recompute=False):
        """Get the FP8 context for the current layer."""
        use_fp8 = self.layer.config.fp8 and self.layer.config.fp8_recipe != Fp8Recipe.delayed
        if not use_fp8:
            return contextlib.nullcontext()
        if recompute:
            return get_fp8_context(self.layer.config, self.layer.layer_number - 1)
        return self.fp8_context

    def forward_impl(self, hidden_states):
        """ Forward pass for the post attention node."""
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state), self.fp8_context:
            if self.use_recompute and not self.is_mtp:
                def custom_forward(*inputs):
                    hidden_states = inputs[0]
                    with self._get_fp8_context(recompute=self.use_recompute):
                        return self.layer._submodule_post_attn_forward(hidden_states)

                (pre_mlp_layernorm_output, tokens_per_expert, permutated_local_input_tokens,
                 probs) = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
            else:
                (pre_mlp_layernorm_output, tokens_per_expert, permutated_local_input_tokens,
                 probs) = self.layer._submodule_post_attn_forward(hidden_states)

        # detached here
        if not self.layer.is_deepep_dispatcher():
            self.common_state.tokens_per_expert = tokens_per_expert
        self.common_state.probs = self.detach(probs)
        self.common_state.residual = self.detach(hidden_states)
        #self.common_state.pre_mlp_layernorm_output = self.detach(pre_mlp_layernorm_output)
        self.common_state.pre_mlp_layernorm_output = pre_mlp_layernorm_output
        return permutated_local_input_tokens


class MoeDispatchNode(TransformerLayerNode):
    """ Dispatch node for the model chunk schedule plan."""

    def forward_impl(self, permutated_local_input_tokens):
        """ Forward pass for the dispatch node."""
        token_dispatcher = self.layer.mlp.token_dispatcher
        probs = self.common_state.probs
        with token_dispatcher.per_batch_state_context(self.common_state), self.fp8_context:
            if self.layer.is_deepep_dispatcher():
                dispatched_input, probs = self.layer._submodule_dispatch_forward(permutated_local_input_tokens, probs)
            else:
                dispatched_input = self.layer._submodule_dispatch_forward(permutated_local_input_tokens)
        if self.layer.is_deepep_dispatcher():
            self.common_state.probs = self.detach(probs)
        return dispatched_input


class MoeMlPNode(TransformerLayerNode):
    """ MLP node for the model chunk schedule plan."""

    def __init__(self, chunk_state, common_state, layer, stream, event, free_inputs=False, layer_idx=None,
                 is_mtp=False):
        super().__init__(
            chunk_state=chunk_state,
            common_state=common_state,
            layer=layer,
            stream=stream,
            event=event,
            free_inputs=free_inputs,
        )
        self.is_mtp = is_mtp
        self.use_recompute = (layer.config.recompute_granularity == 'full' and \
            layer.config.recompute_method in ['block', 'uniform']) or layer.config.moe_layer_recompute
        if layer_idx is not None:
            self.use_recompute = self.use_recompute and layer_idx < layer.config.recompute_num_layers

    def _get_fp8_context(self, recompute=False):
        """Get the FP8 context for the current layer."""
        use_fp8 = self.layer.config.fp8 and self.layer.config.fp8_recipe != Fp8Recipe.delayed
        if not use_fp8:
            return contextlib.nullcontext()
        if recompute:
            return get_fp8_context(self.layer.config, self.layer.layer_number - 1)
        return self.fp8_context
    
    def forward_impl(self, dispatched_input):
        """ Forward pass for the MLP node."""
        pre_mlp_layernorm_output = self.common_state.pre_mlp_layernorm_output.detach()
        probs = self.common_state.probs
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state), self.fp8_context:
            dispatched_input, probs, tokens_per_expert = self.layer._submodule_moe_preprocess(
                dispatched_input, probs,
                self.common_state.tokens_per_expert if not self.layer.is_deepep_dispatcher() else None
            )
            if self.use_recompute and not self.is_mtp:
                def custom_forward(dispatched_input, pre_mlp_layernorm_output, probs, tokens_per_expert):
                    with self._get_fp8_context(recompute=self.use_recompute):
                        expert_output, shared_expert_output, mlp_bias = self.layer._submodule_moe_forward(
                            dispatched_input, pre_mlp_layernorm_output, tokens_per_expert
                        )
                        assert mlp_bias is None
                    return expert_output, shared_expert_output, probs

                expert_output, shared_expert_output, probs = tensor_parallel.checkpoint(
                    custom_forward, False, dispatched_input, pre_mlp_layernorm_output, probs, tokens_per_expert)
            else:    
                expert_output, shared_expert_output, mlp_bias = self.layer._submodule_moe_forward(
                    dispatched_input, pre_mlp_layernorm_output, tokens_per_expert
                )
                assert mlp_bias is None
            expert_output = self.layer._submodule_moe_postprocess(expert_output)
        if self.layer.is_deepep_dispatcher():
            self.common_state.probs = None
        else:
            self.common_state.probs = self.detach(probs)
        # pre_mlp_layernorm_output  used
        self.common_state.pre_mlp_layernorm_output = None
        # detach shared_expert_output
        if self.layer.mlp.use_shared_expert:
            self.common_state.shared_output = self.detach(shared_expert_output)
        # else:
        #     self.common_state.shared_output = None
        return expert_output

    def dw(self):
        """ Backward pass for the MLP node."""
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_mlp_dw()


class MoeCombineNode(TransformerLayerNode):
    """ Combine node for the model chunk schedule plan."""

    def forward_impl(self, permutated_local_input_tokens):
        """ Forward pass for the combine node."""
        token_dispatcher = self.layer.mlp.token_dispatcher
        probs = self.common_state.probs
        with token_dispatcher.per_batch_state_context(self.common_state), self.fp8_context:
            # release tensor not used by backward
            combined_output = self.layer._submodule_combine_forward(
                permutated_local_input_tokens,
                probs
            )

        return combined_output


class MoePostCombineNode(TransformerLayerNode):
    """ Combine node for the model chunk schedule plan."""

    def forward_impl(self, combined_output):
        """ Forward pass for the combine node."""
        token_dispatcher = self.layer.mlp.token_dispatcher
        residual = self.common_state.residual
        if self.layer.mlp.use_shared_expert:
            shared_output = self.common_state.shared_output
        else:
            shared_output = None
        probs = self.common_state.probs
        with token_dispatcher.per_batch_state_context(self.common_state), self.fp8_context:
            # release tensor not used by backward
            output = self.layer._submodule_post_combine_forward(
                combined_output,
                shared_output,
                None,
                probs,
                residual
            )
        # release tensor not used by backward
        if self.layer.mlp.use_shared_expert:
            shared_output.untyped_storage().resize_(0)
            self.common_state.shared_output = None
        # shared_output.untyped_storage().resize_(0)
        # shared_output\residual\prob used
        
        # self.common_state.shared_output = None
        self.common_state.residual = None
        self.common_state.probs = None
        return output


class DenseAttnNode(TransformerLayerNode):
    """ Dense attention node for the model chunk schedule plan."""
    
    def __init__(self, chunk_state, common_state, layer, stream, event, free_inputs=False, layer_idx=None):
        super().__init__(
            chunk_state=chunk_state,
            common_state=common_state,
            layer=layer,
            stream=stream,
            event=event,
            free_inputs=free_inputs,
        )
        self.use_recompute = layer.config.recompute_granularity == 'full' \
            and layer.config.recompute_method in ['block', 'uniform']
        if layer_idx is not None:
            self.use_recompute = self.use_recompute and layer_idx < layer.config.recompute_num_layers
    
    def _get_fp8_context(self, recompute=False):
        """Get the FP8 context for the current layer."""
        use_fp8 = self.layer.config.fp8 and self.layer.config.fp8_recipe != Fp8Recipe.delayed
        if not use_fp8:
            return contextlib.nullcontext()
        if recompute:
            return get_fp8_context(self.layer.config, self.layer.layer_number - 1)
        return self.fp8_context
    
    def forward_impl(self, hidden_states):
        """ Forward pass for the dense attention node."""
        attention_mask = self.chunk_state.attention_mask
        attn_mask_type = self.chunk_state.attn_mask_type
        context = self.chunk_state.context
        context_mask = self.chunk_state.context_mask
        rotary_pos_emb = self.chunk_state.rotary_pos_emb
        rotary_pos_cos = self.chunk_state.rotary_pos_cos
        rotary_pos_sin = self.chunk_state.rotary_pos_sin
        attention_bias = self.chunk_state.attention_bias
        inference_params = self.chunk_state.inference_params
        packed_seq_params = self.chunk_state.packed_seq_params
        sequence_len_offset = self.chunk_state.sequence_len_offset
        with self.fp8_context:
            if self.use_recompute:
                def custom_forward(*inputs):
                    hidden_states = inputs[0]
                    attention_mask = inputs[1]
                    attn_mask_type = inputs[2]

                    from megatron.core.transformer.enums import AttnMaskType
                    if attn_mask_type is not None:
                        attn_mask_type = AttnMaskType(attn_mask_type.item())

                    inference_params = inputs[3]
                    rotary_pos_emb = inputs[4]
                    rotary_pos_cos = inputs[5]
                    rotary_pos_sin = inputs[6]
                    attention_bias = inputs[7]
                    sequence_len_offset = inputs[8]

                    with self._get_fp8_context(recompute=self.use_recompute):
                        hidden_states = self.layer._submodule_attention_forward(
                            hidden_states,
                            attention_mask=attention_mask,
                            attn_mask_type=attn_mask_type,
                            inference_params=inference_params,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )
                    return hidden_states

                if attn_mask_type is not None:
                    _attn_mask_type = torch.tensor(
                        [attn_mask_type.value], dtype=torch.int, device=hidden_states.device)
                else:
                    _attn_mask_type = None

                hidden_states = tensor_parallel.checkpoint(
                    custom_forward, False, hidden_states,
                    attention_mask, _attn_mask_type, inference_params, rotary_pos_emb, rotary_pos_cos,
                    rotary_pos_sin, attention_bias, sequence_len_offset)
            else:
                hidden_states = self.layer._submodule_attention_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    attn_mask_type=attn_mask_type,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )
        return hidden_states

    def dw(self):
        """ Backward pass for the dense attention node."""
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_attention_router_compound_dw()

class FakeScheduleNode:
    """ Fake schedule node for the model chunk schedule plan."""
    def forward(self, inputs):
        """ Forward pass for the fake schedule node."""
        return inputs

    def backward(self, outgrads):
        """ Backward pass for the fake schedule node."""
        return outgrads


class DenseMlpNode(TransformerLayerNode):
    """ Dense MLP node for the model chunk schedule plan."""

    def __init__(self, chunk_state, common_state, layer, stream, event, free_inputs=False, layer_idx=None):
        super().__init__(
            chunk_state=chunk_state,
            common_state=common_state,
            layer=layer,
            stream=stream,
            event=event,
            free_inputs=free_inputs,
        )
        self.use_recompute = layer.config.recompute_granularity == 'full' \
            and layer.config.recompute_method in ['block', 'uniform']
        if layer_idx is not None:
            self.use_recompute = self.use_recompute and layer_idx < layer.config.recompute_num_layers
    
    def _get_fp8_context(self, recompute=False):
        """Get the FP8 context for the current layer."""
        use_fp8 = self.layer.config.fp8 and self.layer.config.fp8_recipe != Fp8Recipe.delayed
        if not use_fp8:
            return contextlib.nullcontext()
        if recompute:
            return get_fp8_context(self.layer.config, self.layer.layer_number - 1)
        return self.fp8_context
    
    def forward_impl(self, hidden_states):
        """ Forward pass for the dense MLP node."""

        with self.fp8_context:
            if self.use_recompute:
                def custom_forward(hidden_states):
                    with self._get_fp8_context(recompute=self.use_recompute):
                        output =  self.layer._submodule_dense_forward(hidden_states)
                    return output
                output = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
            else:
                output = self.layer._submodule_dense_forward(hidden_states)

        return output

    def dw(self):
        """ Backward pass for the dense MLP node."""
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            self.layer._submodule_mlp_dw()

def build_non_moe_layer_plan(layer, event, chunk_state, comp_stream, com_stream):
    """ Build a non-MoE layer schedule plan."""
    common_state = TransformerLayerState()
    attn = DenseAttnNode(chunk_state, common_state, layer, comp_stream, event)
    attn.name = "attn"
    post_attn = FakeScheduleNode()
    dispatch = FakeScheduleNode()
    mlp = DenseMlpNode(chunk_state, common_state, layer, comp_stream, event)
    mlp.name = "mlp"
    combine = FakeScheduleNode()
    post_combine = FakeScheduleNode()
    return TransformerLayerSchedulePlan(attn, post_attn, dispatch, mlp, combine, post_combine)


def build_layer_schedule_plan(layer, event, chunk_state, comp_stream, com_stream, layer_idx=None):
    """ Build a schedule plan for a transformer layer."""
    if not isinstance(layer.mlp, MoELayer):
        return build_non_moe_layer_plan(layer, event, chunk_state, comp_stream, com_stream)
    if layer.is_deepep_dispatcher():
        common_state = MoEFlexPerBatchState()
    else:
        common_state = TransformerLayerState()
    attn = MoeAttnNode(chunk_state, common_state, layer, comp_stream, event, layer_idx=layer_idx)
    attn.name = "attn"
    post_attn = MoePostAttnNode(chunk_state, common_state, layer, comp_stream, event, layer_idx=layer_idx)
    post_attn.name = "post_attn"
    dispatch = MoeDispatchNode(chunk_state, common_state, layer, com_stream, event,
                               False if layer.is_deepep_dispatcher() else True)
    dispatch.name = "dispatch"
    mlp = MoeMlPNode(chunk_state, common_state, layer, comp_stream, event, True, layer_idx=layer_idx)
    mlp.name = "mlp"
    combine = MoeCombineNode(chunk_state, common_state, layer, com_stream, event, True)
    combine.name = "combine"
    post_combine = MoePostCombineNode(chunk_state, common_state, layer, comp_stream, event)
    post_combine.name = "post_combine"
    return TransformerLayerSchedulePlan(attn, post_attn, dispatch, mlp, combine, post_combine)


def build_mtp_layer_schedule_plan(layer, event, chunk_state, comp_stream, com_stream, layer_idx=None, model=None):
    """ Build a schedule plan for an MTP layer."""
    if layer.is_deepep_dispatcher():
        common_state = MoEFlexPerBatchState()
    else:
        common_state = TransformerLayerState()
    
    attn = MoeAttnNode(chunk_state, common_state, layer, comp_stream, event, layer_idx=layer_idx, is_mtp=True, 
                        model=model)
    attn.name = "attn"
    post_attn = MoePostAttnNode(chunk_state, common_state, layer, comp_stream, event, layer_idx=layer_idx, is_mtp=True)
    post_attn.name = "post_attn"
    dispatch = MoeDispatchNode(chunk_state, common_state, layer, com_stream, event,
                               False if layer.is_deepep_dispatcher() else True)
    dispatch.name = "dispatch"
    mlp = MoeMlPNode(chunk_state, common_state, layer, comp_stream, event, True, layer_idx=layer_idx, is_mtp=True)
    mlp.name = "mlp"
    combine = MoeCombineNode(chunk_state, common_state, layer, com_stream, event, True)
    combine.name = "combine"
    post_combine = MoePostCombineNode(chunk_state, common_state, layer, comp_stream, event)
    post_combine.name = "post_combine"
    return TransformerLayerSchedulePlan(attn, post_attn, dispatch, mlp, combine, post_combine)


class TransformerLayerState(MoEAlltoAllPerBatchState):
    """ State for the transformer layer schedule plan."""
    pass


class ModelChunkSate:
    """ Model chunk state for the model chunk schedule plan."""
    pass


class TransformerLayerSchedulePlan:
    """ Schedule plan for a transformer layer."""
    def __init__(self, attn, post_attn, dispatch, mlp, combine, post_combine):
        self.attn = attn
        self.post_attn = post_attn
        self.dispatch = dispatch
        self.mlp = mlp
        self.combine = combine
        self.post_combine = post_combine


class ModelChunkSchedulePlan(AbstractSchedulePlan):
    """ Schedule plan for the model chunk."""
    def __init__(self):
        super().__init__()
        self._pre_process = None
        self._post_process = None
        self._mtp_post_process = None
        self._mtp_layer = []
        self._model_chunk_state = ModelChunkSate()
        self._transformer_layers = []
        self._event = torch.cuda.Event()

    @classmethod
    def forward_backward(
        cls,
        f_schedule_plan,
        b_schedule_plan,
        grad=None,
        f_context=None,
        b_context=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """ Forward and backward pass for the model chunk schedule plan."""
        return schedule_chunk_1f1b(
            f_schedule_plan,
            b_schedule_plan,
            b_grad=grad,
            f_context=f_context,
            b_context=b_context,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
        )

    @property
    def event(self):
        """ Event for the model chunk schedule plan."""
        return self._event

    def record_current_stream(self):
        """ Record the current stream for the model chunk schedule plan."""
        stream = torch.cuda.current_stream()
        self.event.record(stream)

    def wait_current_stream(self):
        """ Wait for the current stream for the model chunk schedule plan."""
        stream = torch.cuda.current_stream()
        self.event.wait(stream)

    @property
    def pre_process(self):
        """ Preprocess node for the model chunk schedule plan."""
        return self._pre_process

    @pre_process.setter
    def pre_process(self, value):
        """ Set the preprocess node for the model chunk schedule plan."""
        self._pre_process = value

    @property
    def post_process(self):
        """ Postprocess node for the model chunk schedule plan."""
        return self._post_process

    @post_process.setter
    def post_process(self, value):
        """ Set the postprocess node for the model chunk schedule plan."""
        self._post_process = value

    @property
    def mtp_post_process(self):
        """ Postprocess node for the model chunk schedule plan."""
        return self._mtp_post_process

    @mtp_post_process.setter
    def mtp_post_process(self, value):
        """ Set the postprocess node for the model chunk schedule plan."""
        self._mtp_post_process = value

    def get_layer(self, i):
        """ Get the transformer layer at index i."""
        assert i < self.num_layers()
        return self._transformer_layers[i]
    
    def get_mtp_layer(self, i):
        """ Get the MTP layer at index i."""
        if len(self._mtp_layer) == 0:
            return None
        assert i < len(self._mtp_layer)
        return self._mtp_layer[i]

    def num_layers(self):
        """ Get the number of transformer layers in the model chunk schedule plan."""
        return len(self._transformer_layers)

    def add_layer(self, layer):
        """ Add a transformer layer to the model chunk schedule plan."""
        self._transformer_layers.append(layer)
    
    def add_mtp_layer(self, layer):
        """ Add an MTP layer to the model chunk schedule plan."""
        self._mtp_layer.append(layer)

    @property
    def state(self):
        """ Get the model chunk state for the model chunk schedule plan."""
        return self._model_chunk_state


def schedule_layer_1f1b(
    f_layer,
    b_layer,
    f_input=None,
    b_grad=None,
    f_context=None,
    b_context=None,
    post_forward=None,
    post_backward=None,
    f_schedule_plan=None,
    b_schedule_plan=None,
    is_last_layer=False,

):
    """ Schedule a layer for 1f1b."""
    f_context = f_context if f_context is not None else contextlib.nullcontext()
    b_context = b_context if b_context is not None else contextlib.nullcontext()

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.post_combine.backward(b_grad)
            b_grad = b_layer.combine.backward(b_grad)

    if f_layer is not None:
        with f_context:
            f_input = f_layer.attn.forward(f_input)
            f_input = f_layer.post_attn.forward(f_input)

    if f_layer is not None:
        with f_context:
            f_input = f_layer.dispatch.forward(f_input)

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.mlp.backward(b_grad)
            
    if b_layer is not None:
        with b_context:
            b_grad = b_layer.dispatch.backward(b_grad)
            b_layer.mlp.dw()

    if f_layer is not None:
        with f_context:
            f_input = f_layer.mlp.forward(f_input)

    if f_layer is not None:
        with f_context:
            f_input = f_layer.combine.forward(f_input)

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.post_attn.backward(b_grad)

    if f_layer is not None:
        with f_context:
            f_input = f_layer.post_combine.forward(f_input)

    if is_last_layer:
        if f_schedule_plan is not None and post_forward is not None:
            with f_context:
                f_schedule_plan.wait_current_stream()
                post_forward(f_input)

    if b_layer is not None:
        with b_context:
            b_grad = b_layer.attn.backward(b_grad)

    if is_last_layer:
        if b_schedule_plan is not None and post_backward is not None:
            with b_context:
                b_schedule_plan.wait_current_stream()
                post_backward(b_grad)

    if b_layer is not None:
        with b_context:
            b_layer.attn.dw()

    return f_input, b_grad


def schedule_chunk_1f1b(
    f_schedule_plan,
    b_schedule_plan,
    b_grad=None,
    f_context=None,
    b_context=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
):
    """ Schedule a chunk for 1f1b."""
    f_context = f_context if f_context is not None else contextlib.nullcontext()
    b_context = b_context if b_context is not None else contextlib.nullcontext()

    f_input = None
    mtp_b_grad = None
    if f_schedule_plan:
        # pp output send/receive sync
        if pre_forward is not None:
            with f_context:  # virtual pipeline parallel context
                pre_forward()
        f_schedule_plan.record_current_stream()
        f_input = f_schedule_plan.pre_process.forward()

    if b_schedule_plan:
        b_schedule_plan.record_current_stream()

        assert b_grad is not None

        if pre_backward is not None:
            with b_context:
                pre_backward()
            b_schedule_plan.record_current_stream()

        # MTP postprocess backward
        if b_schedule_plan.mtp_post_process is not None:
            with b_context:
                (mtp_b_grad, b_grad) = b_schedule_plan.mtp_post_process.backward(b_grad)
        
        # MTP layer backward
        if b_schedule_plan.get_mtp_layer(0) is not None:
            with b_context:
                b_layer = b_schedule_plan.get_mtp_layer(0)
                torch.cuda.nvtx.range_push(f"mtp_layer_b")
                _, mtp_b_grad = schedule_layer_1f1b(
                    None, b_layer, b_grad=mtp_b_grad
                )
                torch.cuda.nvtx.range_pop()

        if b_schedule_plan.post_process is not None:
            with b_context:  # virtual pipeline parallel context
                b_grad = b_schedule_plan.post_process.backward(b_grad)

    f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
    b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
    overlaped_layers = min(f_num_layers, b_num_layers)
    equal_layers = (f_num_layers == b_num_layers)
    b_grad = (b_grad + mtp_b_grad) if mtp_b_grad is not None else b_grad

    for i in range(overlaped_layers):
        f_layer = f_schedule_plan.get_layer(i)
        b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
        torch.cuda.nvtx.range_push(f"layer_{i}f-layer_{b_num_layers - 1 - i}b")
        f_input, b_grad = schedule_layer_1f1b(
            f_layer,
            b_layer,
            f_input=f_input,
            b_grad=b_grad,
            f_context=f_context,
            b_context=b_context,
            post_forward=post_forward,
            post_backward=post_backward,
            f_schedule_plan=f_schedule_plan,
            b_schedule_plan=b_schedule_plan,
            is_last_layer=(i == overlaped_layers - 1 and equal_layers),
        )
        torch.cuda.nvtx.range_pop()

    with b_context:
        for i in range(overlaped_layers, b_num_layers):
            b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
            torch.cuda.nvtx.range_push(f"layer_{b_num_layers - 1 - i}b")
            _, b_grad = schedule_layer_1f1b(
                None, b_layer, b_grad=b_grad
            )
            torch.cuda.nvtx.range_pop()

    with f_context:
        for i in range(overlaped_layers, f_num_layers):
            f_layer = f_schedule_plan.get_layer(i)
            torch.cuda.nvtx.range_push(f"layer_{i}f")
            f_input, _ = schedule_layer_1f1b(f_layer, None, f_input=f_input)
            torch.cuda.nvtx.range_pop()

    if not equal_layers:
        # output pp send receive, overlapped with attn backward
        if f_schedule_plan is not None and post_forward is not None:
            with f_context:
                # post_forward()/send_forward_recv_forward() is running in the communication stream,
                # so the p2p comm could be overlapped with the attn backward
                with torch.cuda.stream(get_com_stream()):
                    f_schedule_plan.wait_current_stream()
                    post_forward(f_input)

        # pp grad send / receive, overlapped with attn dw of cur micro-batch
        # and forward attn of next micro-batch
        if b_schedule_plan is not None and post_backward is not None:
            with b_context:
                b_schedule_plan.wait_current_stream()
                post_backward(b_grad)
    
    ori_f_input = f_input
    # post process forward
    with f_context:
        if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
            f_input = f_schedule_plan.post_process.forward(ori_f_input)
    
    # MTP layer forward
    with f_context:
        if f_schedule_plan is not None and f_schedule_plan.get_mtp_layer(0) is not None:
            f_layer = f_schedule_plan.get_mtp_layer(0)
            torch.cuda.nvtx.range_push(f"mtp_layer_f")
            f_mtp_input, _ = schedule_layer_1f1b(f_layer, None, f_input=ori_f_input)
            torch.cuda.nvtx.range_pop()
    
    # MTP post process forward
    with f_context:
        if f_schedule_plan is not None and f_schedule_plan.mtp_post_process is not None:
            f_input = f_schedule_plan.mtp_post_process.forward((f_mtp_input, f_input))
    
    # pre process backward
    with b_context:
        if b_schedule_plan is not None:
            b_schedule_plan.pre_process.backward(b_grad)

    if f_schedule_plan:
        f_schedule_plan.wait_current_stream()
    if b_schedule_plan:
        b_schedule_plan.wait_current_stream()

    return f_input


def build_model_chunk_schedule_plan(
    model,
    input_ids: Optional[LongTensor],
    position_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    attn_mask_type: Optional[AttnMaskType] = None,
    labels: Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    image_inputs: Optional[Dict[str, torch.Tensor]] = None,
    video_inputs: Optional[Dict[str, torch.Tensor]] = None,
    audio_inputs: Optional[Dict[str, torch.Tensor]] = None,
    decoder_input: Tensor = None,
    runtime_gather_output: Optional[bool] = None,
    num_nextn_predict_layers: Any = None,
    extra_block_kwargs: Any = None,
):
    """Builds a schedule plan for a model chunk.

    This function creates a schedule plan for a model chunk, including
    preprocessing, transformer layers, and postprocessing.

    Args:
        model: The model to build a schedule plan for.
        input_ids: Input token IDs.
        position_ids: Position IDs.
        attention_mask: Attention mask.
        attn_mask_type: Attention mask type.
        labels: Labels for loss computation.
        inference_params: Parameters for inference.
        packed_seq_params: Parameters for packed sequences.
        image_inputs: Image inputs.
        video_inputs: Video inputs.
        audio_inputs: Audio inputs.
        decoder_input: Decoder input tensor.
        runtime_gather_output: Whether to gather output at runtime.
        num_nextn_predict_layers: Number of nextn predict layers.
        extra_block_kwargs: Additional keyword arguments for blocks.

    Returns:
        The model chunk schedule plan.
    """
    comp_stream = get_comp_stream()
    com_stream = get_com_stream()
    model_chunk_schedule_plan = ModelChunkSchedulePlan()
    event = model_chunk_schedule_plan.event
    state = model_chunk_schedule_plan.state

    # save for later use
    state.input_ids = input_ids
    state.position_ids = position_ids
    state.attention_mask = attention_mask
    state.attn_mask_type = attn_mask_type
    state.labels = labels
    state.labels_for_mtp = labels[:] if labels is not None else None
    state.inference_params = inference_params
    state.packed_seq_params = packed_seq_params
    state.image_inputs = image_inputs
    state.video_inputs = video_inputs
    state.audio_inputs = audio_inputs
    state.decoder_input = decoder_input
    state.runtime_gather_output = runtime_gather_output
    state.extra_block_kwargs = extra_block_kwargs
    state.rotary_pos_emb_mtp = None
    state.context = None
    state.context_mask = None
    state.attention_bias = None

    # build preprocess
    model_chunk_schedule_plan.pre_process = PreProcessNode(model, state, event, comp_stream)
    model_chunk_schedule_plan.pre_process.name = "pre_process"

    # if has foundation, use foundation as model
    if hasattr(model, "foundation_model"):
        model = model.foundation_model
    # build for layers
    for layer_idx in range(model.decoder.num_layers_per_pipeline_rank):
        layer = model.decoder._get_layer(layer_idx)
        layer_plan = build_layer_schedule_plan(layer, event, state, comp_stream, com_stream, layer_idx=layer_idx)
        model_chunk_schedule_plan.add_layer(layer_plan)
    
    # build post process
    if model.post_process:
        model_chunk_schedule_plan.post_process = PostProcessNode(model, state, event, comp_stream)
        model_chunk_schedule_plan.post_process.name = "post_process"

    # MTP layer construct
    if num_nextn_predict_layers is not None and model.mtp_layers is not None:
        assert (num_nextn_predict_layers == 1), 'A2A overlap only support one MTP layer now'
        for layer_idx in range(num_nextn_predict_layers):
            layer = model.mtp_layers[layer_idx]
            mtp_layer_plan = build_mtp_layer_schedule_plan(layer, event, state, comp_stream, com_stream,
                                                           layer_idx, model)
            model_chunk_schedule_plan.add_mtp_layer(mtp_layer_plan)
        model_chunk_schedule_plan.mtp_post_process = MtpPostProcessNode(model, state, event, comp_stream)
        model_chunk_schedule_plan.mtp_post_process.name = "mtp_post_process"

    return model_chunk_schedule_plan