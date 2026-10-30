# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiMo model."""

from typing import Optional

import torch
from torch import Tensor

from megatron.core import InferenceParams
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection

from loongforge.models.foundation.base import BaseGPTModel
from loongforge.models.utils import import_module
from .mimo_config import MimoConfig


def _load_state_dict_hook_ignore_extra_state(module, incompatible_keys):
    """Ignore TE extra-state keys for compatibility across TE versions."""
    keys_to_remove = [
        key
        for key in incompatible_keys.missing_keys
        if "input_layernorm._extra_state" in key
        or "pre_mlp_layernorm._extra_state" in key
        or "enorm._extra_state" in key
        or "hnorm._extra_state" in key
        or "eh_proj._extra_state" in key
        or "output_layernorm._extra_state" in key
        or "self_attention.q_layernorm._extra_state" in key
        or "self_attention.k_layernorm._extra_state" in key
        or "linear_fc1._extra_state" in key
        or "linear_fc2._extra_state" in key
    ]

    for key in keys_to_remove:
        if key in incompatible_keys.missing_keys:
            incompatible_keys.missing_keys.remove(key)


class MimoModelWithMTP(BaseGPTModel):
    """MiMo Transformer language model with optional MTP."""

    config_class = MimoConfig

    def __init__(
        self,
        config: MimoConfig,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        scatter_embedding_sequence_parallel: bool = True,
        language_embedding: Optional[torch.nn.Module] = None,
        rotary_dtype: torch.dtype = torch.float32,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        **kwargs,
    ) -> None:
        if config.model_spec is None:
            model_spec = [
                "loongforge.models.foundation.mimo.mimo_layer_spec",
                "get_mimo_decoder_block_and_mtp_spec",
            ]
        else:
            model_spec = config.model_spec

        transformer_layer_spec, mtp_layer_spec = import_module(
            model_spec, config, vp_stage=vp_stage
        )

        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=(
                not config.untie_embeddings_and_output_weights
            ),
            position_embedding_type=config.position_embedding_type,
            language_embedding=language_embedding,
            rotary_dtype=rotary_dtype,
            rotary_emb_func="RotaryEmbedding",
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            rope_scaling=config.use_rope_scaling,
            rope_scaling_factor=config.rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
            mtp_block_spec=mtp_layer_spec,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        self.register_load_state_dict_post_hook(_load_state_dict_hook_ignore_extra_state)

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
        """Forward method for MiMo model with MTP."""
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
