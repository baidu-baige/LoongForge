# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image model provider."""

import torch
from megatron.core.transformer.spec_utils import import_module

from loongforge.models.factory import register_model_provider
from loongforge.utils import build_transformer_config, get_args, print_rank_0
from loongforge.utils.constants import CustomModelFamilies

from .qwen_image_config import QwenImageConfig
from .qwen_image_layer_spec import get_qwen_image_layer_with_te_spec
from .qwen_image_model import QwenImageModel


@register_model_provider(model_family=[CustomModelFamilies.QWEN_IMAGE])
def qwen_image_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: int = None,
) -> QwenImageModel:
    """Build and return the Qwen-Image DiT model wired for Megatron FSDP + TP."""
    args = get_args()
    if args.context_parallel_size != 1:
        raise AssertionError("Qwen-Image provider uses FSDP + TP; set CP to 1.")
    print_rank_0(f"building {args.model_name} model ...")
    config = build_transformer_config(args, config_class=QwenImageConfig)
    config.pipeline_dtype = torch.float32
    config.normalization = "LayerNorm"
    config.num_query_groups = config.num_attention_heads
    if config.num_attention_heads % config.tensor_model_parallel_size != 0:
        raise AssertionError("Qwen-Image attention heads must divide tensor parallel size.")

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if args.transformer_impl != "transformer_engine":
            raise AssertionError("Qwen-Image requires --transformer-impl transformer_engine")
        transformer_layer_spec = get_qwen_image_layer_with_te_spec()

    model = QwenImageModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size or 0,
        max_sequence_length=args.max_position_embeddings or args.seq_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=parallel_output,
    )
    return model
