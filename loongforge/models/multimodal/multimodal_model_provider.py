# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal Model Provider. Provides model construction interfaces compatible with existing training frameworks.
"""

from __future__ import annotations

from typing import Optional

from .combination_model import MultimodalCombinationModel
from megatron.core import mpu
import torch


def multimodal_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: Optional[int] = None,
    *,
    args,
    model_config,
    runtime,
) -> MultimodalCombinationModel:
    """
    Construct and return a multimodal combination model instance.
    
    This function builds a multimodal combination model from the caller's training args
    and model config, with support for parallel processing and pre/post-processing options.

    Args:
        pre_process: Whether to perform pre-processing (default: True)
        post_process: Whether to perform post-processing (default: True)
        parallel_output: Whether to enable parallel output (default: True)
        args: Training args of the run.
        model_config: Model config of the run.
        runtime: ``MultimodalRuntime`` with the engine bindings, built by the training method.

    Returns:
        MultimodalCombinationModel: Constructed combination model instance

    Notes:
        - Supports encoder pipeline parallel processing
        - Includes language model related parameter configuration
    """
    # build_transformer_config(args)
    # FIXME: Need to handle when model_type is encoder_and_decoder
    if args.enable_full_hetero_dp:
        # encoder only resides in the first VPP chunk (model[0]); when VPP is
        # disabled vp_stage is None, which also satisfies the condition.
        add_encoder = (vp_stage is None or vp_stage == 0)
    else:
        add_encoder = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)

    # temporary fix for multimodal model config
    for name in ["image_encoder", "image_projector"]:
        component_config = getattr(model_config, name)
        component_config.pipeline_model_parallel_size = 1
        component_config.tensor_model_parallel_size = 1
        component_config.sequence_parallel = False
        component_config.tp_comm_overlap = False
        component_config.context_parallel_size = 1
        component_config.context_parallel_ulysses_degree = 1
        component_config.pipeline_model_parallel_layout = None
        component_config.mtp_num_layers = None

    # TODO: fp8 support
    model = MultimodalCombinationModel(
        model_config,
        parallel_output=parallel_output,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        language_rotary_dtype=torch.float32,  # if args.rope_in_fp32 else args.params_dtype,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
        allow_missing_adapter_checkpoint=args.allow_missing_adapter_checkpoint,
        vp_stage=vp_stage,
        use_vit_dp_balance=args.use_vit_dp_balance,
        enable_encoder_hetero_dp=args.enable_encoder_hetero_dp,
        enable_full_hetero_dp=args.enable_full_hetero_dp,
        runtime=runtime,
    )

    return model
