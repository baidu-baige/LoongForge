"""cogvlm model provider"""

import torch
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_args

from aiak_training_omni.utils import get_args, build_transformer_config, print_rank_0
from aiak_training_omni.utils.constants import VisionLanguageModelFamilies

from aiak_training_omni.models.factory import register_model_provider

from .cogvlm_model import CogVLMModel
from .cogvlm_config import get_vision_config, get_adapeter_config
from .cogvlm_layer_spec import (
    get_language_layer_with_spec,
    get_vision_layer_with_spec,
    get_adapeter_layer_with_spec,
)
from dataclasses import dataclass, asdict
from copy import deepcopy


@register_model_provider(model_family=[VisionLanguageModelFamilies.COGVLM2])
def cogvlm_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
) -> CogVLMModel:
    """Builds the CogVlm model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits

    Returns:
        CogVLMModel: The returned model
    """
    args = get_args()

    print_rank_0("building CovVLM model ...")

    config = build_transformer_config(args)

    language_config = deepcopy(config)
    vision_config = deepcopy(config)
    adapter_config = deepcopy(config)

    for k, v in asdict(get_vision_config()).items():
        setattr(vision_config, k, v)

    for k, v in asdict(get_adapeter_config()).items():
        setattr(adapter_config, k, v)

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        adapter_layer_spec = get_adapeter_layer_with_spec()
        vision_layer_spec = get_vision_layer_with_spec()
        language_layer_spec = get_language_layer_with_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )

    model = CogVLMModel(
        language_config=language_config,
        vision_config=vision_config,
        adapter_config=adapter_config,
        language_layer_spec=language_layer_spec,
        vision_layer_spec=vision_layer_spec,
        adapter_layer_spec=adapter_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        language_rotary_dtype=torch.float32 if args.rope_in_fp32 else args.params_dtype,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )

    if args.trainable_modules != ["all"]:
        train_language_model = "language_model" in args.trainable_modules
        train_vision_model = "vision_model" in args.trainable_modules
        train_adapter = "adapter" in args.trainable_modules
        model.freeze(
            freeze_language_model=not train_language_model,
            freeze_vision_model=not train_vision_model,
            freeze_adapter=not train_adapter,
        )

        train_language_expert_linear = (
            "language_expert_linear" in args.trainable_modules
        )
        train_vision_expert_linear = "vision_expert_linear" in args.trainable_modules
        model.unfreeze_expert_linear(
            train_language_expert_linear, train_vision_expert_linear
        )

    return model
