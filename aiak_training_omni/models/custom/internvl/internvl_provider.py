"""intern_vl clip model provider"""

import inspect
from contextlib import nullcontext
import torch
from megatron.core import mpu
from megatron.core.transformer.spec_utils import import_module

from aiak_training_omni.utils import get_args, get_tokenizer, build_transformer_config, print_rank_0
from aiak_training_omni.utils.constants import VisionLanguageModelFamilies

from aiak_training_omni.models.factory import register_model_provider, get_model_family

from .internvl_model import InternVLModel
from .internvl_config import get_vision_config, get_adapeter_config
from .internvl_layer_spec import (
    get_language_layer_with_te_spec,
    get_vision_layer_with_te_spec,
    get_adapeter_layer_with_te_spec,
)
from dataclasses import dataclass, asdict
from copy import deepcopy

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


@register_model_provider(model_family=[VisionLanguageModelFamilies.INTERN_VL])
def internvl_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
) -> InternVLModel:
    """Builds the intern_vl clip model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits

    Returns:
        InternClipModel: The returned model
    """
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0(f'building InternVL model: {args.model_family}')

    config = build_transformer_config(args)

    language_config = deepcopy(config)
    vision_config = deepcopy(config)
    adapter_config = deepcopy(config)

    for k, v in asdict(get_vision_config[args.model_family]()).items():
        setattr(vision_config, k, v)

    for k, v in asdict(get_adapeter_config[args.model_family]()).items():
        setattr(adapter_config, k, v)
        
    # FIXME: fix this if model_type is encoder_and_decoder
    if args.communicate_dataset:
        setattr(language_config, 'communicate_dataset', True)
    vision_config.pipeline_model_parallel_size = 1
    adapter_config.tensor_model_parallel_size = 1
    adapter_config.pipeline_model_parallel_size = 1
    add_encoder = mpu.is_pipeline_first_stage()  # vision model  # FIXME optimizer empty params list error
    add_decoder = True

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        adapter_layer_spec = get_adapeter_layer_with_te_spec()
        vision_layer_spec = get_vision_layer_with_te_spec(vision_config)
        # print(f"model_family: {args.model_family}, language_config: {language_config}")
        language_layer_spec = get_language_layer_with_te_spec(args.model_family, language_config)

    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except:
            raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine,but not found.")

    with build_model_context(**build_model_context_args):
        model = InternVLModel(language_config=language_config,
            vision_config=vision_config,
            adapter_config=adapter_config,
            language_layer_spec=language_layer_spec,
            vision_layer_spec=vision_layer_spec,
            adapter_layer_spec=adapter_layer_spec,
            language_vocab_size=args.padded_vocab_size,
            language_max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=add_decoder,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            language_position_embedding_type=args.position_embedding_type,
            language_rotary_percent=args.rotary_percent,
            language_rotary_base=args.rotary_base,
            language_rotary_dtype=torch.float32 if args.rope_in_fp32 else args.params_dtype,
            language_seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
            img_context_token_id=tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        )

    if args.trainable_modules != ['all']:
        train_language_model = "language_model" in args.trainable_modules
        train_vision_model = "vision_model" in args.trainable_modules
        train_adapter = "adapter" in args.trainable_modules
        model.freeze(freeze_language_model=not train_language_model,
                     freeze_vision_model=not train_vision_model,
                     freeze_adapter=not train_adapter)

    return model
