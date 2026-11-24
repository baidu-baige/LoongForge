# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 model backed by the native MCore HybridModel."""

from typing import Optional

import torch
from torch import Tensor

from megatron.core import InferenceParams
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection

from loongforge.models.utils import import_module
from .deepseek_v4_config import DeepseekV4Config


def get_deepseek_v4_hybrid_stack_spec(config=None, vp_stage=None):
    """Return MCore's native DeepSeek-V4 HybridStack specification."""
    return hybrid_dsv4_stack_spec


def _load_state_dict_hook_ignore_extra_state(module, incompatible_keys):
    """Ignore Transformer Engine ``_extra_state`` differences only."""
    keys_to_remove = [
        key
        for key in (*incompatible_keys.missing_keys, *incompatible_keys.unexpected_keys)
        if "._extra_state" in key
    ]

    for key in keys_to_remove:
        if key in incompatible_keys.missing_keys:
            incompatible_keys.missing_keys.remove(key)
        if key in incompatible_keys.unexpected_keys:
            incompatible_keys.unexpected_keys.remove(key)


class DeepseekV4Model(HybridModel):
    """DeepSeek-V4 language model using native HybridStack execution."""

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
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported DeepSeek-V4 HybridModel arguments: {unexpected}")
        if language_embedding is not None:
            raise NotImplementedError(
                "External language embeddings are not supported by the Phase 1 "
                "DeepSeek-V4 HybridModel integration."
            )

        # The LoongForge model provider reapplies --rotary-percent after the
        # config's __post_init__. DS-V4 W layers require full RoPE over
        # qk_pos_emb_head_dim; compressed C/H layers select Yarn independently.
        config.rotary_percent = 1.0

        hybrid_layer_pattern = config.hybrid_layer_pattern
        if hybrid_layer_pattern is None:
            raise ValueError("DeepSeekV4Config did not derive hybrid_layer_pattern.")

        model_spec = config.model_spec or [
            "loongforge.models.foundation.deepseek_v4.deepseek_v4_model",
            "get_deepseek_v4_hybrid_stack_spec",
        ]
        hybrid_stack_spec = import_module(model_spec, config, vp_stage=vp_stage)
        if hybrid_stack_spec is None:
            raise ImportError(f"Failed to load DeepSeek-V4 HybridStack spec: {model_spec}")
        super().__init__(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_position_embeddings,
            hybrid_layer_pattern=hybrid_layer_pattern,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=(
                not config.untie_embeddings_and_output_weights
            ),
            position_embedding_type=config.position_embedding_type,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        # Compatibility attributes used by existing LoongForge inference/export paths.
        self.transformer_layer_spec = hybrid_stack_spec
        self.max_position_embeddings = config.max_position_embeddings
        self.rotary_percent = config.rotary_percent
        self.rotary_base = config.rotary_base

        self.register_load_state_dict_post_hook(_load_state_dict_hook_ignore_extra_state)

    @classmethod
    def _from_config(cls, config: DeepseekV4Config, **kwargs) -> "DeepseekV4Model":
        """Construct through the transformers AutoModel dispatch path."""
        return cls(config=config, **kwargs)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        extra_block_kwargs: Optional[dict] = None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[Tensor] = None,
        mtp_input_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        compute_mtp_loss: bool = True,
        cp_batch=None,
    ) -> Tensor:
        """Run the native DeepSeek-V4 HybridModel forward path."""
        if extra_block_kwargs:
            raise ValueError(
                "DeepSeek-V4 HybridModel Phase 1 does not support chunk-pipe "
                "extra_block_kwargs."
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
            runtime_gather_output=runtime_gather_output,
            loss_mask=loss_mask,
            mtp_input_mask=mtp_input_mask,
            padding_mask=padding_mask,
            compute_mtp_loss=compute_mtp_loss,
            cp_batch=cp_batch,
        )
