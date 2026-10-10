# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model utilities."""

from typing import Tuple
import torch
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.activations import squared_relu
import torch.nn.functional as F
from loongforge import constants
from copy import deepcopy
from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class
from loongforge.models.multimodal.vlm_model_config import VLMModelConfig
from collections.abc import Iterable
from loongforge.engines.mcore.global_vars import get_args_dict, get_tokenizer


def import_module(module_path: Tuple[str], config: TransformerConfig, **kwargs):
    """Import a named object from a module in the context of this function.

    TODO: make this importer module more robust, at least make sure there
    are no side effects of using this as is
    """
    base_path, name = module_path
    try:
        module = __import__(base_path, globals(), locals(), [name])
    except ImportError as e:
        print(f"couldn't import module due to {e}")
        return None
    return vars(module)[name](config, **kwargs)


def convert_megatron_transformer_config_args(megatron_args, config_class=None):
    """convert megatron args to transformer config"""
    # Config class.
    if config_class is None:
        config_class = TransformerConfig if not megatron_args["multi_latent_attention"] else MLATransformerConfig

    transformer_config_args = {}
    for k, v in megatron_args.items():
        if k in config_class.__dataclass_fields__:
            transformer_config_args[k] = v
    transformer_config_args["persist_layer_norm"] = not megatron_args[
        "no_persist_layer_norm"
    ]
    transformer_config_args["layernorm_zero_centered_gamma"] = megatron_args[
        "apply_layernorm_1p"
    ]
    transformer_config_args["layernorm_epsilon"] = megatron_args["norm_epsilon"]
    transformer_config_args["deallocate_pipeline_outputs"] = True
    transformer_config_args["pipeline_dtype"] = megatron_args["params_dtype"]
    transformer_config_args["batch_p2p_comm"] = not megatron_args["overlap_p2p_comm"]
    transformer_config_args["num_moe_experts"] = megatron_args["num_experts"]
    transformer_config_args["rotary_interleaved"] = megatron_args["rotary_interleaved"]
    transformer_config_args["num_layers_in_first_pipeline_stage"] = megatron_args[
        "decoder_first_pipeline_num_layers"
    ]
    transformer_config_args["num_layers_in_last_pipeline_stage"] = megatron_args[
        "decoder_last_pipeline_num_layers"
    ]
    transformer_config_args["fp8_param"] = megatron_args["fp8_param_gather"]

    if "activation_func_fp8_input_store" in megatron_args:
        transformer_config_args["activation_func_fp8_input_store"] = megatron_args[
            "activation_func_fp8_input_store"
        ]

    if megatron_args["swiglu"]:
        transformer_config_args["activation_func"] = F.silu
        transformer_config_args["gated_linear_unit"] = True
        transformer_config_args["bias_activation_fusion"] = megatron_args[
            "bias_swiglu_fusion"
        ]
    else:
        transformer_config_args["bias_activation_fusion"] = megatron_args[
            "bias_gelu_fusion"
        ]

    if megatron_args["squared_relu"]:
        assert not megatron_args["swiglu"]
        transformer_config_args["activation_func"] = squared_relu

    if megatron_args["init_method_xavier_uniform"]:
        transformer_config_args["init_method"] = torch.nn.init.xavier_uniform_
        transformer_config_args["scaled_init_method"] = torch.nn.init.xavier_uniform_

    if megatron_args["group_query_attention"]:
        transformer_config_args["num_query_groups"] = megatron_args["num_query_groups"]
    else:
        transformer_config_args["num_query_groups"] = None
    if len(megatron_args["cp_comm_type"]) == 1:
        transformer_config_args["cp_comm_type"] = megatron_args["cp_comm_type"][0]
    transformer_config_args["config_logger_dir"] = megatron_args["config_logger_dir"]
    
    if megatron_args["rope_type"] is None:
        # Pop 'rope_type' to let the config class use the default value.
        transformer_config_args.pop('rope_type', None)
    else:
        assert (megatron_args["multi_latent_attention"] or megatron_args["rope_type"] == 'rope'), (
            f'Common attention only support rope_type="rope", but got {megatron_args["rope_type"]}.'
        )

    return transformer_config_args


def build_model_config(args, config):
    """Build model config from args and config"""

    model_cfgs = {}

    if (hasattr(config, "model_type") and config.model_type in
            (set(constants.LanguageModelFamilies.names()) |
            set(constants.CustomModelFamilies.names()) |
            set(constants.VisionLanguageActionModelFamilies.names()))):
        model_type = config.model_type
        model_config = config
    else:
        if not hasattr(config, "model"):
            raise ValueError("Invalid model configuration structure")
        model_type = config.model.model_type
        model_config = config.model

    # assert hasattr(config.model, "model_type"), "model_type is required in model config"
    vision_custom_names = {*constants.VisionLanguageModelFamilies.names(), *constants.CustomModelFamilies.names()}
    if model_type in vision_custom_names:
        # get the global args dict which contains all model component args
        global_args_dict = get_args_dict()
        for name, config_values in model_config.items():
            # must have _target_ field
            if isinstance(config_values, Iterable) and "_target_" in config_values:
                # get corresponding args dict
                args_dict = deepcopy(vars(global_args_dict[name])) \
                    if hasattr(global_args_dict[name], "__dict__") else deepcopy(global_args_dict[name])
                if name == "peft_config":
                    model_cfgs[name] = instantiate(config_values)
                else:
                    # Filter by the class the YAML names, so fields declared on a
                    # LoongForge config subclass survive and unknown keys fail.
                    config_cls = get_class(config_values["_target_"])
                    # Derive CLI-based fields first, then let YAML override them.
                    merged = convert_megatron_transformer_config_args(deepcopy(args_dict), config_cls)
                    merged.update(OmegaConf.to_container(config_values, resolve=True))
                    model_cfgs[name] = instantiate(config_values, **merged)
                    # Configs with an `image_token` field have no fixed id: it is whatever the tokenizer assigns.
                    if hasattr(model_cfgs[name], "image_token") and model_cfgs[name].image_token_id is None:
                        model_cfgs[name].image_token_id = get_tokenizer().tokenizer.convert_tokens_to_ids(
                            model_cfgs[name].image_token
                        )
            else:
                model_cfgs[name] = config_values
        model_cfgs = VLMModelConfig(**model_cfgs)
    elif model_type in (set(constants.LanguageModelFamilies.names())
                        | set(constants.CustomModelFamilies.names())):
        if "_target_" not in model_config:
            raise ValueError(
                "Model config missing '_target_' field.\n"
                "This field is required for llm or custom model types.\n")
        args_dict = deepcopy(vars(args))
        config_cls = get_class(model_config["_target_"])
        # Derive CLI-based fields first, then let YAML override them.
        merged = convert_megatron_transformer_config_args(deepcopy(args_dict), config_cls)
        merged.update(OmegaConf.to_container(model_config, resolve=True))
        model_cfgs = instantiate(model_config, **merged)
    elif model_type in constants.VisionLanguageActionModelFamilies.names():
        # Vision-language-action models are self-contained (no Megatron config merge yet)
        if "_target_" not in model_config:
            raise ValueError("Model config missing '_target_' field for vla type.\n")

        # Remove dispatcher-only metadata like model_type before instantiation.
        vla_config = deepcopy(model_config)
        if isinstance(vla_config, dict) and "model_type" in vla_config:
            vla_config = deepcopy(vla_config)
            vla_config.pop("model_type", None)
        elif "model_type" in getattr(vla_config, "keys", lambda: [])():
            # DictConfig / OmegaConf path
            vla_config = OmegaConf.create(OmegaConf.to_container(vla_config, resolve=True))
            vla_config.pop("model_type", None)
        vla_config.random_fallback_cpu = args.random_fallback_cpu
        model_cfgs = instantiate(vla_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model_cfgs
