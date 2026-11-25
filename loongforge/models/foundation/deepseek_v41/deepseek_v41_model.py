# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash model shell."""

from typing import Optional

import torch
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from torch import Tensor

from loongforge.models.foundation.base import BaseGPTModel

from .deepseek_v41_config import DeepseekV41Config
from .deepseek_v41_layers import DeepseekV41DecoderBlock
from .deepseek_v41_vision import (
    DeepseekV41VisionAligner,
    DeepseekV41VisionConfig,
    DeepseekV41VisionTransformer,
)

TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(-1, 4)


def _load_state_dict_hook_ignore_extra_state(module, incompatible_keys):
    """Ignore Transformer Engine FP8 metadata keys during checkpoint loading."""

    del module
    incompatible_keys.missing_keys[:] = [key for key in incompatible_keys.missing_keys if "._extra_state" not in key]
    incompatible_keys.unexpected_keys[:] = [
        key for key in incompatible_keys.unexpected_keys if "._extra_state" not in key
    ]


class DeepseekV41Model(BaseGPTModel):
    """V4.1 model using LoongForge's common embedding/output/loss contract."""

    config_class = DeepseekV41Config

    def __init__(
        self,
        config: DeepseekV41Config,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        scatter_embedding_sequence_parallel: bool = True,
        language_embedding: Optional[torch.nn.Module] = None,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        **kwargs,
    ) -> None:
        if config.mtp_num_layers:
            raise ValueError("DeepSeek-V4.1 DSpark is not enabled; mtp_num_layers must be zero")
        if config.context_parallel_size != 1:
            raise ValueError("DeepSeek-V4.1 CP is not enabled")
        super().__init__(
            config=config,
            transformer_layer_spec=None,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=getattr(config, "fp16_lm_cross_entropy", False),
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=(not config.untie_embeddings_and_output_weights),
            position_embedding_type=config.position_embedding_type,
            language_embedding=language_embedding,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            rope_scaling=getattr(config, "use_rope_scaling", False),
            rope_scaling_factor=getattr(config, "rope_scaling_factor", config.rotary_scaling_factor),
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=getattr(config, "rotary_seq_len_interpolation_factor", None),
            mtp_block_spec=None,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            decoder_cls=DeepseekV41DecoderBlock,
        )
        if post_process:
            # The released head stores BF16 weights but computes logits in FP32.
            self.output_layer.weight.data = self.output_layer.weight.data.float()
            self.output_layer.weight._keep_in_float32 = True
        self.vision = None
        self.aligner = None
        if pre_process:
            if config.vision_config is not None and int(config.vision_config.get("num_hidden_layers", 0)) > 0:
                vision_config = DeepseekV41VisionConfig.from_mapping(config.vision_config)
                self.vision = DeepseekV41VisionTransformer(vision_config)
                self.aligner = DeepseekV41VisionAligner(vision_config, config.hidden_size)
                self.vision.to(dtype=config.params_dtype)
                self.aligner.to(dtype=config.params_dtype)
                for module in (
                    self.vision.norm,
                    *(block.norm1 for block in self.vision.blocks),
                    *(block.norm2 for block in self.vision.blocks),
                ):
                    module.weight._keep_in_float32 = True
                    module.weight.data = module.weight.data.float()
            self.image_start = torch.nn.Parameter(torch.empty(config.hidden_size))
            self.image_end = torch.nn.Parameter(torch.empty(config.hidden_size))
            self.image_newline = torch.nn.Parameter(torch.empty(config.hidden_size))
            for parameter in (self.image_start, self.image_end, self.image_newline):
                config.init_method(parameter)
                parameter.data = parameter.data.to(config.params_dtype)
        self.register_load_state_dict_post_hook(_load_state_dict_hook_ignore_extra_state)

    def _fuse_vision(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        pixel_values: Tensor,
        image_grid_hws: Tensor,
        vision_token_types: Tensor,
    ) -> Tensor:
        if self.vision is None or self.aligner is None:
            raise ValueError("DeepSeek-V4.1 vision inputs require vision_config")
        if vision_token_types.shape != input_ids.shape:
            raise ValueError("vision_token_types must match input_ids")
        if vision_token_types.dtype not in (torch.int32, torch.int64):
            raise ValueError("vision_token_types must be an integer tensor")
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [patches, 3, patch, patch]")
        if torch.any(input_ids[vision_token_types >= 0] != self.config.image_token_id):
            raise ValueError("image span tokens must use the checkpoint image_token_id")
        embedded = self.embedding.word_embeddings(input_ids)
        if self.embedding.reduce_scatter_embeddings:
            embedded = tensor_parallel.gather_from_sequence_parallel_region(
                embedded, tensor_parallel_output_grad=False, group=self.pg_collection.tp
            ).transpose(0, 1)
        patches = self.vision(pixel_values, image_grid_hws)
        features = self.aligner(patches, image_grid_hws).to(embedded.dtype)
        content_positions = (vision_token_types == IMAGE).nonzero(as_tuple=False)
        if content_positions.shape[0] != features.shape[0]:
            raise ValueError(
                "image token span count does not match the native vision aligner output: "
                f"{content_positions.shape[0]} != {features.shape[0]}"
            )
        flat = embedded.contiguous().view(-1, embedded.shape[-1])
        destination = content_positions[:, 0] * input_ids.shape[1] + content_positions[:, 1]
        flat = flat.index_copy(0, destination, features)
        fused = flat.view_as(embedded)
        for marker, value in (
            (IMAGE_START, self.image_start),
            (IMAGE_NEW_LINE, self.image_newline),
            (IMAGE_END, self.image_end),
        ):
            fused = torch.where(
                (vision_token_types == marker).unsqueeze(-1),
                value.to(fused.dtype),
                fused,
            )
        fused = fused.transpose(0, 1).contiguous()
        if self.config.fp32_residual_connection:
            fused = fused.float()
        if self.config.sequence_parallel:
            if self.embedding.scatter_to_sequence_parallel:
                fused = tensor_parallel.scatter_to_sequence_parallel_region(fused, group=self.pg_collection.tp)
                if self.config.clone_scatter_output_in_embedding:
                    fused = fused.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                return self.embedding.embedding_dropout(fused)
        return self.embedding.embedding_dropout(fused)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if packed_seq_params is not None:
            raise NotImplementedError("DeepSeek-V4.1 packed sequences are not supported")
        extra_block_kwargs = dict(extra_block_kwargs or {})
        pixel_values = extra_block_kwargs.pop("pixel_values", None)
        image_grid_hws = extra_block_kwargs.pop("image_grid_hws", None)
        vision_token_types = extra_block_kwargs.pop("vision_token_types", None)
        if pixel_values is not None:
            if image_grid_hws is None or vision_token_types is None:
                raise ValueError("pixel_values requires image_grid_hws and vision_token_types")
            if self.pre_process:
                decoder_input = self._fuse_vision(
                    input_ids,
                    position_ids,
                    pixel_values,
                    image_grid_hws,
                    vision_token_types,
                )
        elif vision_token_types is not None:
            if self.pre_process and torch.any(vision_token_types >= 0):
                raise ValueError("vision_token_types contains image spans but pixel_values is missing")
        if vision_token_types is not None:
            extra_block_kwargs["v41_image_mask"] = vision_token_types >= 0
            extra_block_kwargs["v41_vision_token_types"] = vision_token_types
        extra_block_kwargs["input_ids"] = input_ids
        extra_block_kwargs["v41_position_ids"] = position_ids
        extra_block_kwargs.setdefault("v41_sequence_ids", extra_block_kwargs.get("sequence_ids"))
        extra_block_kwargs.setdefault(
            "v41_image_mask",
            extra_block_kwargs.get("image_mask", extra_block_kwargs.get("visual_pos_masks")),
        )
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            loss_mask=loss_mask,
        )
