# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 model"""

from contextlib import nullcontext
import logging
from collections import OrderedDict
from typing import Dict, Literal, Optional, Any

import torch
from torch import Tensor

from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.inference.contexts import BaseInferenceContext
from loongforge.models.utils import import_module
from loongforge.models.foundation.base import BaseGPTModel
from .deepseek_v4_config import DeepseekV4Config


def _load_state_dict_hook_ignore_extra_state(module, incompatible_keys):
    """Hook to ignore Transformer Engine ``_extra_state`` keys only.

    ``_extra_state`` is TE's FP8 amax-history holder; older ckpts don't have it and newer TE
    versions add it, so ignoring missing ``_extra_state`` entries is safe. ``tid2eid`` and
    ``expert_bias`` must be populated from the HF checkpoint; if they land in missing_keys
    that is a real load failure that needs to surface, not be swallowed here.

    Args:
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    keys_to_remove = [
        key
        for key in incompatible_keys.missing_keys
        if "._extra_state" in key
    ]

    for key in keys_to_remove:
        if key in incompatible_keys.missing_keys:
            incompatible_keys.missing_keys.remove(key)


class DeepseekV4Model(BaseGPTModel):
    """DeepSeek-V4 Transformer language model with MTP module supported.

    Extends DeepSeek-V3 with Hyper-Connections (mHC), Compressed Sparse
    Attention (CSA/HCA), grouped output projection, and SwiGLU clamping.

    Args:
        config (DeepseekV4Config): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks.
            Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights
            are shared. Defaults to False.
        share_mtp_embeddings_and_output_weights (bool, optional): When True, MTP layers' embeddings and output logit
            weights are shared with main model. Default to True.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to
            'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'. Defaults to 10000.
    """

    config_class = DeepseekV4Config

    def __init__(
        self,
        config: DeepseekV4Config,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        scatter_embedding_sequence_parallel: bool = True,
        language_embedding: Optional[torch.nn.Module] = None,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        **kwargs,
    ) -> None:

        if config.model_spec is None:
            model_spec = [
                "loongforge.models.foundation.deepseek_v4.deepseek_v4_layer_spec",
                "get_deepseek_v4_decoder_block_and_mtp_spec",
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

        self.register_load_state_dict_post_hook(
            _load_state_dict_hook_ignore_extra_state
        )

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
        """Forward function of the DeepSeek-V4 Model.

        This function passes the input tensors through the embedding layer,
        and then the decoder and finally into the post processing layer (optional).

        It either returns the Loss values if labels are given or the final hidden units.

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
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
