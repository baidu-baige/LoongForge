# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Parser arguments."""

import argparse
import functools
import os
from collections.abc import Iterable
from copy import deepcopy

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import get_class, instantiate
from megatron.core.activations import squared_relu
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.training.arguments import add_megatron_arguments, moe_freq_type
from megatron.training.checkpointing import load_args_from_checkpoint
from omegaconf import DictConfig, OmegaConf

from loongforge.engines.megatron.get_loss_func import default_loss_func
from loongforge.engines.megatron.utils.global_vars import (
    get_args_dict,
    get_hydra_config,
    get_tokenizer,
    set_args_dict,
    set_hydra_config,
    set_model_config,
)
from loongforge.engines.megatron.validators import (
    validate_custom_model_args,
    validate_loongforge_extra_args,
    validate_megatron_args,
)
from loongforge.models.catalog import (
    TOP_LEVEL_MODEL_FAMILIES,
    DiffusionModelFamilies,
    VisionLanguageModelFamilies,
    get_config_from_model_name,
    split_model_config,
)
from loongforge.models.common.vlm_model_config import VLMModelConfig
from loongforge.models.omni_models.get_position_idx_func import POSITION_IDX_FUNCS


def register_custom_resolvers():
    """register custom omegaconf resolvers"""
    # Activation functions
    ACTIVATION_MAP = {
        "relu": F.relu,
        "gelu": F.gelu,
        "silu": F.silu,
        "gelu_tanh": functools.partial(F.gelu, approximate="tanh"),
    }
    LOSS_FUNC_MAP = {
        "default": default_loss_func,
    }
    OmegaConf.register_new_resolver(
        "act", lambda name: ACTIVATION_MAP[name.lower()], replace=True
    )
    OmegaConf.register_new_resolver(
        "position_func", lambda name: POSITION_IDX_FUNCS[name.lower()], replace=True
    )
    OmegaConf.register_new_resolver(
        "loss_func", lambda name: LOSS_FUNC_MAP[name.lower()], replace=True
    )

    # moe layer freq resolver
    OmegaConf.register_new_resolver(
        "moe_freq",
        lambda expr: moe_freq_type(expr),
        replace=True
    )


def parse_megatron_arguments(
    extra_args_provider=None, parse_unknown_args=False, *, extra_args_providers=()
):
    """Parse megatron arguments."""
    parser = argparse.ArgumentParser(
        description="Megatron-LM Arguments", allow_abbrev=False
    )

    parser = add_megatron_arguments(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # LoongForge providers register on the same parser as the upstream args.
    for provider in extra_args_providers:
        parser = provider(parser)

    # Unknown flags are Hydra overrides.
    hydra_overrides = []
    if parse_unknown_args:
        args, hydra_overrides = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Args from environment
    # support MPI
    args.rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "-1"))
    if args.rank == -1:
        args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "-1"))
    if args.world_size == -1:
        args.world_size = int(os.getenv("WORLD_SIZE", "1"))

    return args, hydra_overrides


def load_and_merge_config(config_path, config_name, hydra_overrides):
    """
    Load configuration using the Hydra API and handle defaults inheritance.

    This function will:
    1. Load the configuration using Hydra's compose API.
    2. Automatically handle combined configurations in the defaults list.
    3. Handle package redirection with the @ symbol.
    4. Apply command-line overrides.
    """
    # Convert to absolute path
    config_path = os.path.abspath(config_path)

    # Clear previous Hydra instance (if exists)
    GlobalHydra.instance().clear()

    try:
        # Initialize using Hydra's initialize_config_dir
        with initialize_config_dir(config_dir=config_path, version_base=None):
            # Load configuration using compose, which automatically processes defaults
            cfg = compose(config_name=config_name, overrides=hydra_overrides)

        return cfg

    except Exception as e:
        print(
            f"Cannot load hydra config: {e}. Config path: {config_path}, \
                config name: {config_name}"
        )
        raise


def get_config_from_file(config_file: str):
    """
    Split a full config file path into:
    - config_path (directory)
    - config_name (file name without extension)
    """
    config_file = os.path.abspath(config_file)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config_dir = os.path.dirname(config_file)
    file_name = os.path.basename(config_file)

    if not file_name.endswith((".yaml", ".yml")):
        raise ValueError("Config file must end with .yaml or .yml")

    # remove extension
    config_name = os.path.splitext(file_name)[0]

    return config_dir, config_name


def parse_arguments(
    extra_args_provider=None,
    validate_extra_args_provider=None,
    args_defaults={},
    parse_unknown_args=False,
    *,
    args_providers=(),
    extra_validator=None,
):
    """Parse arguments."""
    args, hydra_overrides = parse_megatron_arguments(
        extra_args_provider,
        parse_unknown_args,
        extra_args_providers=args_providers,
    )

    # Prep for checkpoint conversion.
    if args.ckpt_convert_format is not None:
        assert args.ckpt_convert_save is not None
        assert args.load is not None
        args.exit_on_missing_checkpoint = True

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoints-args requires --load argument"
        assert args.non_persistent_ckpt_type != "local", (
            "--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local. "
            "Two-stage checkpoint loading is not implemented, and all arguments must be defined "
            "before initializing LocalCheckpointManager."
        )
        load_args_from_checkpoint(args)

    hydra_cfg = None

    # mapping those parameters that can't be parsed
    register_custom_resolvers()

    # mapping model name to config path and name
    if hasattr(args, "model_name") and args.model_name is not None:
        args.config_path, args.config_name = get_config_from_model_name(args.model_name)
    elif args.config_file:
        args.config_path, args.config_name = get_config_from_file(args.config_file)
    else:
        raise ValueError("Either --model-name or --config-file must be specified.")

    if args.config_path and args.config_name:
        hydra_cfg = load_and_merge_config(
            args.config_path, args.config_name, hydra_overrides
        )

    model_type, model_config = split_model_config(hydra_cfg)

    # TODO: remove this in the future
    args.model_family = model_type

    if model_type in VisionLanguageModelFamilies.names():
        args_dict = {}
        for name, config_values in model_config.items():
            # exclude those non-iterable config values
            if not (isinstance(config_values, (dict, DictConfig)) and "_target_" in config_values):
                continue
            # Validate arguments.
            args_deepcopy = deepcopy(args)
            if validate_extra_args_provider is not None:
                if extra_validator is None:
                    validate_extra_args_provider(args_deepcopy, config_values)
                else:
                    validate_extra_args_provider(args_deepcopy, config_values, extra_validator)

            for key in args_defaults:
                # just overwrite the args with defaults
                setattr(args_deepcopy, key, args_defaults[key])

            assert args_deepcopy.yaml_cfg is None, "yaml_cfg is not supported in LoongForge yet"

            # TODO: any better way to do this?
            # set default model values for projector so that it can be validated by megatron
            if "foundation" in name:
                validate_megatron_args(args_deepcopy)
            else:
                validate_custom_model_args(name, args_deepcopy)

            args_dict[name] = args_deepcopy

        if "foundation" not in args_dict:
            raise ValueError("args_dict does not contain 'foundation'")
        args = args_dict["foundation"]

        # set global args dict
        set_args_dict(args_dict)

    elif model_type in TOP_LEVEL_MODEL_FAMILIES:
        # Validate arguments.
        if validate_extra_args_provider is not None:
            if extra_validator is None:
                validate_extra_args_provider(args, hydra_cfg)
            else:
                validate_extra_args_provider(args, hydra_cfg, extra_validator)

        for key in args_defaults:
            # just overwrite the args with defaults
            setattr(args, key, args_defaults[key])

        assert args.yaml_cfg is None, "yaml_cfg is not supported in LoongForge yet"
        validate_megatron_args(args)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return args, hydra_cfg


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


def _instantiate_from_args(args_dict, yaml_node):
    """Instantiate ``yaml_node``: derive fields from Megatron args first, then let YAML override them."""
    # Filter by the class the YAML names, so fields declared on a
    # LoongForge config subclass survive and unknown keys fail.
    merged = convert_megatron_transformer_config_args(deepcopy(args_dict), get_class(yaml_node["_target_"]))
    merged.update(OmegaConf.to_container(yaml_node, resolve=True))
    return instantiate(yaml_node, **merged)


def build_model_config(args, config):
    """Build model config from args and config"""
    model_type, model_config = split_model_config(config)

    if model_type in {*VisionLanguageModelFamilies.names(), *DiffusionModelFamilies.names()}:
        # get the global args dict which contains all model component args
        global_args_dict = get_args_dict()
        model_cfgs = {}
        for name, config_values in model_config.items():
            # must have _target_ field
            if not (isinstance(config_values, Iterable) and "_target_" in config_values):
                model_cfgs[name] = config_values
            elif name == "peft_config":
                model_cfgs[name] = instantiate(config_values)
            else:
                component_args = global_args_dict[name]
                model_cfgs[name] = _instantiate_from_args(
                    vars(component_args) if hasattr(component_args, "__dict__") else component_args,
                    config_values,
                )
                # Configs with an `image_token` field have no fixed id: it is whatever the tokenizer assigns.
                if hasattr(model_cfgs[name], "image_token") and model_cfgs[name].image_token_id is None:
                    model_cfgs[name].image_token_id = get_tokenizer().tokenizer.convert_tokens_to_ids(
                        model_cfgs[name].image_token
                    )
        return VLMModelConfig(**model_cfgs)
    if model_type in TOP_LEVEL_MODEL_FAMILIES:
        if "_target_" not in model_config:
            raise ValueError(
                "Model config missing '_target_' field.\n"
                "This field is required for llm or custom model types.\n")
        return _instantiate_from_args(vars(args), model_config)
    raise ValueError(f"Unsupported model type: {model_type}")


def parse_args_from_config(args):
    """parse args from config"""
    config = get_hydra_config()
    model_cfgs = build_model_config(args, config)
    set_model_config(model_cfgs)
    _register_selective_fp8_decision()


def _register_selective_fp8_decision():
    """Register selective-FP8 init decision callback.

    Always registering keeps behavior stable in long-lived processes and avoids
    assumptions about ``model_cfgs`` shape (dict vs dataclass/object).
    The callback itself falls back to static whitelist when no dynamic policy is
    configured in a given model config.
    """
    try:
        from loongforge.engines.megatron.fp8_dynamic_policy import selective_fp8_init_decision
        from megatron.core.fp8_utils import register_selective_fp8_init_decision
    except ImportError:
        return

    register_selective_fp8_init_decision(selective_fp8_init_decision)


def parse_train_args(args_defaults={}, *, args_providers=(), extra_validator=None):
    """Parse arguments for training.

    ``args_providers`` registers training-side arguments on the same parser as
    the engine arguments. ``extra_validator`` runs after the tokenizer checks
    and before the engine's training and parallel checks.
    """
    args, hydra_cfg = parse_arguments(
        args_providers=args_providers,
        validate_extra_args_provider=validate_loongforge_extra_args,
        args_defaults=args_defaults,
        parse_unknown_args=True,
        extra_validator=extra_validator,
    )
    set_hydra_config(hydra_cfg)
    # Apply data/model config to args early so downstream init sees hydrated values

    return args
