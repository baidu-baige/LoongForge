# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Omni Model Provider.Provides model construction interfaces compatible with existing training frameworks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .omni_combination_model import OmniCombinationModel
from loongforge.utils import get_args
from megatron.core import mpu
import torch
from loongforge.utils import build_transformer_config, get_model_config
from loongforge.models.common import BaseModelConfig


def check_model_config(model_config: BaseModelConfig):
    """Validate model configuration including hidden size and number of attention heads."""
    for config in model_config:  # TODO: Implement actual validation logic
        # Check hidden_size and num_attention_heads
        print(config)
    return model_config


def omni_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: Optional[int] = None,
) -> OmniCombinationModel:
    """
    Construct and return an Omni combination model instance.
    
    This function builds a multimodal combination model based on provided parameters 
    and global args configuration. Model configuration is obtained from args.model_name,
    with support for parallel processing and pre/post-processing options.

    Args:
        pre_process: Whether to perform pre-processing (default: True)
        post_process: Whether to perform post-processing (default: True)
        parallel_output: Whether to enable parallel output (default: True)

    Returns:
        OmniCombinationModel: Constructed combination model instance

    Notes:
        - Model configuration is obtained from args.model_name
        - Supports encoder pipeline parallel processing
        - Includes language model related parameter configuration
    """
    args = get_args()
    model_config = get_model_config()
    # check_model_config(model_config)
    # build_transformer_config(args)
    # FIXME: Need to handle when model_type is encoder_and_decoder
    if args.enable_full_hetero_dp:
        # encoder only resides in the first VPP chunk (model[0]); when VPP is
        # disabled vp_stage is None, which also satisfies the condition.
        add_encoder = (vp_stage is None or vp_stage == 0)
    else:
        add_encoder = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)

    # temporary fix for omni model config
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
    model = OmniCombinationModel(
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
    )

    return model
