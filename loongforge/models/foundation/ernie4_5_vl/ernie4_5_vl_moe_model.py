# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Ernie4.5vl model"""

import torch
from typing import Optional, Dict
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from loongforge.models.foundation.base import BaseGPTModel
from .ernie_decoder_layer_spec import get_ernie4_5_vl_decoder_spec
from .ernie_pos_embedding import ErnieRopeEmbedding
from .ernie_config import ErnieMoeConfig


class ErnieMoeModel(BaseGPTModel):
    """Ernie4.5VLMoeModel"""
    def __init__(self, 
        config: ErnieMoeConfig,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        scatter_embedding_sequence_parallel: bool = True,
        language_embedding: Optional[torch.nn.Module] = None,
        share_embeddings_and_output_weights: bool = False,
        rotary_dtype: torch.dtype = torch.float32,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        **kwargs,
        ):
        transformer_layer_spec = get_ernie4_5_vl_decoder_spec(config)
        rotary_pos_emb = None
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=config.vocab_size_in_config_file,
            max_sequence_length=config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=(
                not config.untie_embeddings_and_output_weights),
            position_embedding_type=config.position_embedding_type,
            language_embedding=language_embedding,
            rotary_dtype=rotary_dtype,
            rotary_emb_func=config.rotary_emb_func,
            rotary_pos_emb=rotary_pos_emb,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            rope_scaling=config.use_rope_scaling,
            rope_scaling_factor=config.rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
  
        self.rotary_pos_emb = ErnieRopeEmbedding(
            self.config.kv_channels,
            compression_ratio=1.0,
            base=500000,
            freq_allocation=20,
        )

        # TODO: implement learned absolute position embedding
        if self.pre_process:
            if language_embedding is None:
                self.embedding = LanguageModelEmbedding(
                    config=self.config,
                    vocab_size=self.config.vocab_size_in_config_file,
                    max_sequence_length=self.max_sequence_length,
                    position_embedding_type=self.position_embedding_type,
                    scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                )
            else:
                self.embedding = language_embedding
        # Transformer
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )


    def set_input_embeddings(self, value):
        """Sets the encoder embeddings."""
        self.language_model.set_input_embeddings(value)

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1'
        self.decoder.set_input_tensor(input_tensor[0])
 
    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[Dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility
        (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(
            prefix, sharded_offsets, metadata
        )
        output_layer_extra_state_key = f"{prefix}output_layer._extra_state"

        # Old GPT checkpoints only stored the output layer weight key. So we remove the
        # _extra_state key but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f"Expected output layer extra state to be empty, got: {output_extra_state}"

        return sharded_state_dict


    def forward(
        self,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: Optional[AttnMaskType] = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        visual_pos_masks: torch.Tensor = None,
        # reserved_params
        input_ids: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        deepstack_visual_embeds: torch.Tensor = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Dict: 
        """
        Forward func of Ernie4.5VLMoeModel.
        """
        # visual_pos_masks do not include <img_start> <img_end> token
        all_image_mask = extra_block_kwargs["token_type_ids"]
        rotary_pos_emb = self.rotary_pos_emb(position_ids)

        # Run decoder.
        extra_block_kwargs = {}
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            # for heterogenous moe layer
            context_mask=~all_image_mask,
            inference_params=None,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            loss_mask=None,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=None,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
        )
