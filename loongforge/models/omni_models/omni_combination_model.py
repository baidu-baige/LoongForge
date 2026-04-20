# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate the execution flow of encoder -> foundation -> decoder"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from .omni_encoder_model import OmniEncoderModel
from .omni_decoder_model import OmniDecoderModel
from .utils import get_inputs_on_this_cp_rank
from transformers.models.auto.modeling_auto import AutoModel
from loongforge.models.common import BaseMegatronModule, BaseModelConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from loongforge.train.initialize import (
    mpu,
    change_parallel_state, 
    get_encoder_dp_size,
    get_num_micro_batches_per_decoder_dp,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_init_chunk_handler,
)
from megatron.training import print_rank_0

class OmniCombinationModel(BaseMegatronModule):
    """Omni multimodal combination model"""
    def __init__(
        self,
        config: BaseModelConfig,
        language_vocab_size: int,
        language_max_sequence_length: int,
        allow_missing_adapter_checkpoint: bool = False,
        parallel_output: bool = True,
        language_position_embedding_type: str = "rope",
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 1000000,
        language_rope_scaling: bool = False,
        language_rope_scaling_factor: float = 8.0,
        language_rotary_dtype=torch.float32,
        fp16_lm_cross_entropy: bool = False,
        share_embeddings_and_output_weights: bool = True,
        seq_len_interpolation_factor: float = None,
        scatter_embedding_sequence_parallel=False,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config.foundation)
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.disable_param_offloading = True

        if config.image_encoder is not None and add_encoder:
            self.encoder_model = OmniEncoderModel(
                config,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                position_embedding_type=language_position_embedding_type,
                scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
                allow_missing_adapter_checkpoint=allow_missing_adapter_checkpoint,
                vp_stage=vp_stage,
            )
            self.vit_contexts = {}
        else:
            self.encoder_model = None

        if config.foundation is not None and add_decoder:
            # TODO: remove this dependency?
            config.foundation.padded_vocab_size = language_vocab_size
            config.foundation.max_position_embeddings = language_max_sequence_length
            config.foundation.position_embedding_type = language_position_embedding_type
            config.foundation.rotary_percent = language_rotary_percent
            config.foundation.rotary_base = language_rotary_base
            config.foundation.use_rope_scaling = language_rope_scaling
            config.foundation.rope_scaling_factor = language_rope_scaling_factor
            config.foundation.rotary_seq_len_interpolation_factor = seq_len_interpolation_factor
            config.foundation.untie_embeddings_and_output_weights = not share_embeddings_and_output_weights
            config.foundation.fp16_lm_cross_entropy = fp16_lm_cross_entropy
            self.foundation_model = AutoModel.from_config(
                config.foundation,  
                pre_process=self.pre_process,
                post_process=self.post_process,
                parallel_output=parallel_output,
                scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
                language_embedding=self.encoder_model.text_encoder if add_encoder else None,
                rotary_dtype=language_rotary_dtype,
                vp_stage=vp_stage,
            )
        else:
            raise ValueError(
                "OmniCombinationModel requires a foundation_config to initialize foundation_model."
            )

        self.share_embeddings_and_output_weights = (
            self.foundation_model.share_embeddings_and_output_weights
        )
        
    def shared_embedding_or_output_weight(self):
        """Get shared embedding or output weight from foundation model.
        This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`.
        """
        if self.add_decoder:
            return self.foundation_model.shared_embedding_or_output_weight()
        return None

    def set_input_embeddings(
        self, inputs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Skip encoder and use input values directly for foundation + decoder models."""
        input_embeds = inputs.get("inputs_embeds")
        decoder_inputs = inputs.get("decoder_inputs", {})

        if input_embeds is None:
            raise ValueError("In offline mode, `inputs_embeds` must be provided.")

        return input_embeds, decoder_inputs

    def set_input_tensor(self, input_tensor) -> None:
        """Set input tensor for the model"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for llava"

        if self.add_encoder and self.add_decoder:
            if self.pre_process:
                self.encoder_model.set_input_tensor(input_tensor[0])
            else:
                self.foundation_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.encoder_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]  # TODO: Handle encoder hidden state
        else:
            self.foundation_model.set_input_tensor(input_tensor[0])

    def set_output_embeddings(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Skip foundation model and use input values directly for decoder model."""
        output_embeddings = inputs.get("output_embeddings")

        if output_embeddings is None:
            raise ValueError("In offline mode, `output_embeddings` must be provided.")

        return output_embeddings

    def get_modality(self):
        """Get input/output modality types of the current model"""
        input_modality = self.encoder.modality
        output_modality = self.decoder.modality
        return {"input": input_modality, "output": output_modality}

    def prepare_inputs_for_generation(
        self, input_ids: Optional[torch.LongTensor] = None
    ):
        """Prepare inputs for generation process"""
        pass

    def generate_multimodal(self, hidden_states):
        """Generate multimodal data"""
        pass

    def preprocess_for_fine_grained_offloading(self):
        """Preprocess for fine-grained activation offloading."""
        fine_grained_offloading_init_chunk_handler(
            self.vp_stage, self.config.min_offloaded_tensor_size
        )
        if self.disable_param_offloading:
            for param in self.foundation_model.decoder.parameters():
                param.offloading_activation = False
            if self.foundation_model.mtp_process:
                for param in self.foundation_model.mtp.parameters():
                    param.offloading_activation = False
            if self.foundation_model.post_process:
                for param in self.foundation_model.output_layer.parameters():
                    param.offloading_activation = False
            self.disable_param_offloading = False

    def hetero_dp_get_tensor_shape(
        self, group, src, local_rank, forward_group_id=None, tensor_name=None,
        idx=None, local_tensor=None
    ):
        """Broadcast the shape of a tensor from src rank to all ranks in the group.

        If local_tensor is provided it is used directly; otherwise the tensor is
        looked up from vit_contexts[forward_group_id][tensor_name] (with optional
        indexed access via idx).  Non-src ranks only need local_tensor for its
        ndim so that they can allocate a zero-filled shape buffer of the right
        length before the broadcast.
        """
        if local_tensor is None:
            local_tensor = self.vit_contexts[forward_group_id][tensor_name]
            if idx is not None:
                local_tensor = local_tensor[idx]
        if local_rank == src:
            shape = torch.tensor(local_tensor.shape, dtype=torch.long, device='cuda')
        else:
            shape = torch.zeros(local_tensor.dim(), dtype=torch.long, device='cuda')

        torch.distributed.broadcast(shape, group=group, src=src)

        return shape

    def hetero_dp_get_tensor(
        self, group, src, local_rank, forward_group_id=None, tensor_name=None,
        shape=None, needs_grad=True, idx=None, local_tensor=None
    ):
        """Broadcast a tensor from src rank to all ranks in the group.

        If local_tensor is provided it is used directly; otherwise the tensor is
        looked up from vit_contexts[forward_group_id][tensor_name] (with optional
        indexed access via idx).  Non-src ranks only need local_tensor for its
        dtype so that they can allocate a correctly-typed zero buffer before the
        broadcast.
        """
        if local_tensor is None:
            local_tensor = self.vit_contexts[forward_group_id][tensor_name]
            if idx is not None:
                local_tensor = local_tensor[idx]
        if local_rank == src:
            tensor = local_tensor.detach()
        else:
            tensor = torch.zeros(tuple(shape.tolist()), dtype=local_tensor.dtype, device='cuda')

        if needs_grad:
            tensor.requires_grad_(needs_grad)

        torch.distributed.broadcast(tensor, group=group, src=src)

        return tensor

    def forward(
        self,
        image_inputs: Optional[Dict[str, torch.Tensor]] = None,
        video_inputs: Optional[Dict[str, torch.Tensor]] = None,
        audio_inputs: Optional[Dict[str, torch.Tensor]] = None,
        *,
        input_ids: Optional[torch.LongTensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_params=None,
        labels: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_params: InferenceParams = None,
        visual_pos_masks: Optional[list[Tensor]] = None,
        deepstack_visual_embeds: Optional[list[Tensor]] = None,
        enable_encoder_hetero_dp: bool = False,
        batch_list: Optional[list] = None,
        forward_group_id: Optional[int] = None,
        inner_group_id: Optional[int] = None,
        enable_full_hetero_dp: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Forward pass supporting multiple execution paths.

        Execution paths:
            1. Full path: encoder -> foundation -> decoder
            2. Offline encoder: use preprocessed inputs_embeds
            3. Offline foundation model: use preprocessed output_embeddings
            4. Decoder only: freeze encoder and foundation
        """
        _ImageEncoderDataParallelSize = get_encoder_dp_size('image_encoder')

        if self.config.fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()

        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if use_inference_kv_cache:
            vision_embeddings = None
        elif self.add_encoder:
            if not enable_encoder_hetero_dp and not enable_full_hetero_dp:
                combined_embeddings, decode_input, visual_pos_masks, deepstack_visual_embeds = self.encoder_model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    image_inputs=image_inputs,
                    video_inputs=video_inputs,
                    inference_params=inference_params,
                    enable_encoder_hetero_dp=enable_encoder_hetero_dp,
                )

                if self.config.context_parallel_size > 1:
                    combined_embeddings = get_inputs_on_this_cp_rank(combined_embeddings, packed_seq_params)

                if self.config.sequence_parallel:
                    combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
            elif enable_encoder_hetero_dp and inner_group_id == 0:
                batch_id = mpu.get_tensor_model_parallel_rank()

                input_embeds_list = []
                for i in range(_ImageEncoderDataParallelSize):
                    input_embeds = self.encoder_model.text_forward(
                        batch_list[i]["tokens"], 
                        batch_list[i]["position_ids"]
                    )
                    input_embeds_list.append(input_embeds)

                (
                    local_images,
                    local_image_grid_thw,
                    local_pixel_values_videos,
                    local_video_grid_thw,
                    local_input_ids,
                    local_attn_mask,
                    local_labels,
                    local_cu_lengths,
                    local_max_lengths,
                    local_position_ids,
                    local_loss_mask,
                    local_packed_seq_params,
                ) = batch_list[batch_id].values()

                combined_embeddings, decode_input, visual_pos_masks, deepstack_visual_embeds = self.encoder_model(
                    input_ids=local_input_ids,
                    position_ids=local_position_ids,
                    image_inputs=dict(
                        images=local_images,
                        image_grid_thw=local_image_grid_thw,
                    ) if local_images is not None else None,
                    video_inputs=dict(
                        pixel_values_videos=local_pixel_values_videos,
                        video_grid_thw=local_video_grid_thw,
                    ) if local_pixel_values_videos is not None else None,
                    inference_params=inference_params,
                    inputs_embeds=input_embeds_list[batch_id],
                    enable_encoder_hetero_dp=enable_encoder_hetero_dp,
                )

                self.vit_contexts.setdefault(forward_group_id, {
                    "local_embedding": combined_embeddings,
                    "grads": None,
                    "local_visual_pos_masks": visual_pos_masks,
                    "local_deepstack_visual_embeds": deepstack_visual_embeds,
                    "local_deepstack_visual_embeds_grads": (
                        [None for _ in deepstack_visual_embeds] 
                        if deepstack_visual_embeds is not None 
                        else None
                    ),
                })
        if not self.pre_process:
            combined_embeddings = None

        if self.add_encoder and mpu.is_pipeline_first_stage() and enable_encoder_hetero_dp:
            group = mpu.get_tensor_model_parallel_group()
            src = torch.distributed.get_global_rank(group, inner_group_id)
            local_rank = torch.distributed.get_rank()

            # combined_embeddings communication
            shape = self.hetero_dp_get_tensor_shape(group, src, local_rank, forward_group_id, "local_embedding")
            combined_embeddings = self.hetero_dp_get_tensor(
                group, src, local_rank, forward_group_id, 
                "local_embedding", shape
            )

            def vit_grad_hook_factory(forward_group_id, inner_group_id, vit_contexts):
                def hook(grad):
                    ctx = vit_contexts[forward_group_id]
                    tp_id = mpu.get_tensor_model_parallel_rank()
                    if tp_id == inner_group_id:
                        ctx["grads"] = grad.clone()

                    if inner_group_id == _ImageEncoderDataParallelSize - 1:
                        bwd_tensors = [ctx["local_embedding"]]
                        bwd_grads = [ctx["grads"]]
                        if ctx["local_deepstack_visual_embeds"] is not None:
                            for t, g in zip(
                                ctx["local_deepstack_visual_embeds"], 
                                ctx["local_deepstack_visual_embeds_grads"]
                            ):
                                if t.requires_grad and t.grad_fn is not None:
                                    bwd_tensors.append(t)
                                    bwd_grads.append(g)
                        torch.autograd.backward(
                            tensors=bwd_tensors,
                            grad_tensors=bwd_grads,
                            retain_graph=False
                        )
                        del vit_contexts[forward_group_id]
                        
                return hook

            combined_embeddings.register_hook(
                vit_grad_hook_factory(forward_group_id, 
                                    inner_group_id, 
                                    self.vit_contexts
                )
            )

            if self.config.context_parallel_size > 1:
                combined_embeddings = get_inputs_on_this_cp_rank(combined_embeddings, packed_seq_params)

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)

            # visual positional encoding communication
            if self.vit_contexts[forward_group_id]["local_visual_pos_masks"] is not None:
                shape = self.hetero_dp_get_tensor_shape(
                    group, src, local_rank, 
                    forward_group_id, "local_visual_pos_masks"
                )
                visual_pos_masks = self.hetero_dp_get_tensor(
                    group, src, local_rank, forward_group_id, 
                    "local_visual_pos_masks", shape, needs_grad=False
                )

            if self.vit_contexts[forward_group_id]["local_deepstack_visual_embeds"] is not None:
                len_deepstack_visual_embeds = len(self.vit_contexts[forward_group_id]["local_deepstack_visual_embeds"])
                shape = self.hetero_dp_get_tensor_shape(
                    group, src, local_rank, forward_group_id, 
                    "local_deepstack_visual_embeds", idx=0
                )
                deepstack_visual_embeds = []
                for i in range(len_deepstack_visual_embeds):
                    tmp_deepstack_visual_embeds = self.hetero_dp_get_tensor(
                        group, src, local_rank, forward_group_id, 
                        "local_deepstack_visual_embeds", shape, idx=i
                    )
                    
                    def deepstack_visual_embeds_grad_hook_factory(forward_group_id, inner_group_id, vit_contexts, idx):
                        def hook(grad):
                            ctx = vit_contexts[forward_group_id]
                            tp_id = mpu.get_tensor_model_parallel_rank()
                            if tp_id == inner_group_id:
                                ctx["local_deepstack_visual_embeds_grads"][idx] = grad.clone()
                                
                        return hook

                    tmp_deepstack_visual_embeds.register_hook(
                        deepstack_visual_embeds_grad_hook_factory(forward_group_id, 
                                                                inner_group_id, 
                                                                self.vit_contexts,
                                                                i
                        )
                    )
                    deepstack_visual_embeds.append(tmp_deepstack_visual_embeds)

        if self.add_encoder and mpu.is_pipeline_first_stage() and enable_full_hetero_dp:
            from loongforge.train.pretrain.pretrain_vlm import (
                get_grad_list, get_embedding_list,
                get_visual_pos_masks_list, get_deepstack_visual_embeds_list,
                get_deepstack_grad_list,
            )
            from loongforge.train.initialize import get_model_size
            group = mpu.get_tensor_model_parallel_group()
            src_rank = torch.distributed.get_global_rank(group, 0)
            local_rank = torch.distributed.get_rank()

            embedding_list = get_embedding_list()
            visual_pos_masks_list = get_visual_pos_masks_list()
            deepstack_visual_embeds_list = get_deepstack_visual_embeds_list()
            model_size = get_model_size()
            round_num = forward_group_id // model_size
            inner_num = forward_group_id % model_size

            # src rank broadcasts its actual embedding; other ranks supply only a
            # dtype/ndim reference so the helpers can allocate the right buffer.
            ref_tensor = self.vit_contexts[round_num]["local_embedding"]
            local_tensor = embedding_list[round_num][inner_num] if local_rank == src_rank else ref_tensor

            shape = self.hetero_dp_get_tensor_shape(
                group, src_rank, local_rank, local_tensor=local_tensor
            )
            combined_embeddings = self.hetero_dp_get_tensor(
                group, src_rank, local_rank, shape=shape, local_tensor=local_tensor
            )

            def full_hetero_dp_grad_hook_factory(group):
                def hook(grad):
                    if torch.distributed.get_rank(group=group) == 0:
                        grad = grad.clone()
                        get_grad_list().append(grad)
                return hook

            combined_embeddings.register_hook(
                full_hetero_dp_grad_hook_factory(group)
            )

            if self.config.context_parallel_size > 1:
                combined_embeddings = get_inputs_on_this_cp_rank(combined_embeddings, packed_seq_params)

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)

            if self.vit_contexts[round_num]["local_visual_pos_masks"] is not None:
                ref_masks = self.vit_contexts[round_num]["local_visual_pos_masks"]
                local_masks = visual_pos_masks_list[round_num][inner_num] if local_rank == src_rank else ref_masks
                shape = self.hetero_dp_get_tensor_shape(
                    group, src_rank, local_rank, local_tensor=local_masks
                )
                visual_pos_masks = self.hetero_dp_get_tensor(
                    group, src_rank, local_rank, shape=shape,
                    local_tensor=local_masks, needs_grad=False,
                )

            if self.vit_contexts[round_num]["local_deepstack_visual_embeds"] is not None:
                ref_embeds = self.vit_contexts[round_num]["local_deepstack_visual_embeds"]

                def full_hetero_dp_deepstack_grad_hook_factory(group, round_num, inner_num, i):
                    def hook(grad):
                        if torch.distributed.get_rank(group=group) == 0:
                            get_deepstack_grad_list()[round_num][i][inner_num] = grad.clone()
                    return hook

                deepstack_visual_embeds = []
                for i in range(len(ref_embeds)):
                    local_embed = (
                        deepstack_visual_embeds_list[round_num][i][inner_num] 
                        if local_rank == src_rank 
                        else ref_embeds[i]
                    )
                    shape = self.hetero_dp_get_tensor_shape(
                        group, src_rank, local_rank, local_tensor=local_embed
                    )
                    embed = self.hetero_dp_get_tensor(
                        group, src_rank, local_rank, shape=shape, local_tensor=local_embed,
                    )
                    embed.register_hook(
                        full_hetero_dp_deepstack_grad_hook_factory(group, round_num, inner_num, i)
                    )
                    deepstack_visual_embeds.append(embed)

        extra_kwargs = {
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": deepstack_visual_embeds,
        }
        kwargs.update(extra_kwargs)
        output = self.foundation_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            # rotary_pos_emb=rotary_pos_emb,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=kwargs,
        )

        return output

    def build_schedule_plan(
        self,
        image_inputs: Optional[Dict[str, torch.Tensor]] = None,
        video_inputs: Optional[Dict[str, torch.Tensor]] = None,
        audio_inputs: Optional[Dict[str, torch.Tensor]] = None,
        *,
        input_ids: Optional[torch.LongTensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_params=None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):

        if self.config.fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()

        """Build the schedule plan for the model."""
        from .model_chunk_schedule_plan import TransformerModelChunkSchedulePlan

        return TransformerModelChunkSchedulePlan(
            self,
            image_inputs=image_inputs,
            video_inputs=video_inputs,
            audio_inputs=audio_inputs,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )
