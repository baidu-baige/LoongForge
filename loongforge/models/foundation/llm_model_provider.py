# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""llm model provider"""

import inspect
from typing import Optional
from contextlib import nullcontext
from transformers import AutoModel

from megatron.core.transformer.spec_utils import import_module
from loongforge.utils import (
    get_args, get_model_config, print_rank_0
)
from loongforge.models.common import BaseMegatronLanguageModule


def llm_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: Optional[int] = None,
):
    """Generic LLM model provider.
    
    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits. Defaults to True.
    
    Returns:
        The corresponding LLM model.
    """
    args = get_args()

    config = get_model_config()

    print_rank_0(f"Building {config.model_type} model...")

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")
    
    # TODO: abstract this to a common dataclass
    config.padded_vocab_size = args.padded_vocab_size
    config.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
    config.rotary_seq_len_interpolation_factor = args.rotary_seq_len_interpolation_factor
    config.use_rope_scaling = args.use_rope_scaling
    config.rope_scaling_factor = args.rope_scaling_factor
    config.max_position_embeddings = args.max_position_embeddings
    config.rotary_base = args.rotary_base
    config.rotary_percent = args.rotary_percent
    config.use_rope_scaling = args.use_rope_scaling
    config.rope_scaling_factor = args.rope_scaling_factor
    config.rotary_seq_len_interpolation_factor = args.rotary_seq_len_interpolation_factor

    # copied from qwen model provider
    # TODO: remove or not?
    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather and not getattr(config, "selective_fp8", False):
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except:
            raise RuntimeError(
                "--fp8-param-gather requires `fp8_model_init` from TransformerEngine,but not found.")

    with build_model_context(**build_model_context_args):
        model: BaseMegatronLanguageModule = AutoModel.from_config(
            config=config,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=parallel_output,
            vp_stage=vp_stage,
        )

    # Validate selective FP8 coverage: detect TE modules that lack the
    # init guard, which would cause "quantized weights without quantized
    # compute" warnings at runtime.
    # Check both top-level config and sub-configs (VLM: foundation, image_encoder, etc.)
    _has_selective_fp8 = getattr(config, "selective_fp8", False)
    if not _has_selective_fp8:
        for _attr in ("foundation", "image_encoder", "video_encoder", "audio_encoder"):
            _sub_cfg = getattr(config, _attr, None)
            if _sub_cfg and getattr(_sub_cfg, "selective_fp8", False):
                _has_selective_fp8 = True
                break
    if _has_selective_fp8:
        try:
            from megatron.core.fp8_utils import validate_selective_fp8_coverage
            validate_selective_fp8_coverage(model)
        except ImportError:
            pass

    return model
