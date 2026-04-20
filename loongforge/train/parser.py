# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Parser arguments."""

import argparse
import os
import sys
from copy import deepcopy

import torch.nn.functional as F
import functools

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from megatron.training.arguments import add_megatron_arguments, moe_freq_type
from megatron.training.checkpointing import load_args_from_checkpoint
from omegaconf import DictConfig, OmegaConf
from dataclasses import fields

from loongforge.models.utils import build_model_config
from loongforge.train.arguments import loongforge_extra_train_args_provider
from loongforge.train.get_loss_func import (default_loss_func,
                                                    loss_func_internvl)
from loongforge.train.get_position_idx_func import (get_mrope_index, 
                                                            get_position_ids, 
                                                            get_rope_index_internvl, 
                                                            get_rope_index_qwen3vl)
from loongforge.train.validators import (validate_loongforge_extra_args,
                                                validate_custom_model_args,
                                                validate_megatron_args)
from loongforge.utils import constants
from loongforge.utils.config_map import get_config_from_model_name
from loongforge.utils.global_vars import (get_hydra_config,
                                                  set_args_dict,
                                                  set_data_config,
                                                  set_hydra_config,
                                                  set_model_config)
from loongforge.utils.utils import get_config_from_file


def register_custom_resolvers():
    """register custom omegaconf resolvers"""
    # Activation functions
    ACTIVATION_MAP = {
        "relu": F.relu,
        "gelu": F.gelu,
        "silu": F.silu,
        "gelu_tanh": functools.partial(F.gelu, approximate="tanh"),
    }
    POSITION_IDX_FUNC_MAP = {
        "position_ids": get_position_ids,
        "mrope_ids": get_mrope_index,
        "rope_ids_internvl": get_rope_index_internvl, 
        "rope_ids_qwen3vl": get_rope_index_qwen3vl
    }
    LOSS_FUNC_MAP = {
        "default": default_loss_func,
        "loss_func_internvl": loss_func_internvl
    }
    OmegaConf.register_new_resolver(
        "act", lambda name: ACTIVATION_MAP[name.lower()], replace=True
    )
    OmegaConf.register_new_resolver(
        "position_func", lambda name: POSITION_IDX_FUNC_MAP[name.lower()], replace=True
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


def parse_megatron_arguments(extra_args_provider=None, parse_unknown_args=False):
    """Parse megatron arguments."""
    parser = argparse.ArgumentParser(
        description="Megatron-LM Arguments", allow_abbrev=False
    )

    parser = add_megatron_arguments(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
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


def parse_arguments(
    extra_args_provider=None,
    validate_extra_args_provider=None,
    args_defaults={},
    parse_unknown_args=False,
):
    """Parse arguments."""
    args, hydra_overrides = parse_megatron_arguments(
        extra_args_provider, parse_unknown_args
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

    if hasattr(hydra_cfg, "model_type") and hydra_cfg.model_type in \
            (set(constants.LanguageModelFamilies.names()) |
            set(constants.CustomModelFamilies.names()) |
            set(constants.VisionLanguageActionModelFamilies.names())):
        model_config = hydra_cfg
        model_type = hydra_cfg.model_type
    else:
        if not hasattr(hydra_cfg, "model"):
            raise ValueError("Invalid model configuration structure")
        model_config = hydra_cfg.model
        model_type = hydra_cfg.model.model_type

    # TODO: remove this in the future
    args.model_family = model_type
    
    if model_type in constants.VisionLanguageModelFamilies.names():
        args_dict = {}
        for name, config_values in model_config.items():
            # exclude those non-iterable config values
            if not (isinstance(config_values, (dict, DictConfig)) and "_target_" in config_values):
                continue
            # Validate arguments.
            args_deepcopy = deepcopy(args)
            if validate_extra_args_provider is not None:
                validate_extra_args_provider(args_deepcopy, config_values)

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

    elif model_type in (set(constants.LanguageModelFamilies.names()) |
            set(constants.CustomModelFamilies.names()) |
            set(constants.VisionLanguageActionModelFamilies.names())):
        # Validate arguments.
        if validate_extra_args_provider is not None:
            validate_extra_args_provider(args, hydra_cfg)

        for key in args_defaults:
            # just overwrite the args with defaults
            setattr(args, key, args_defaults[key])

        assert args.yaml_cfg is None, "yaml_cfg is not supported in LoongForge yet"
        validate_megatron_args(args)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return args, hydra_cfg


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
        from loongforge.train.fp8_dynamic_policy import selective_fp8_init_decision
        from megatron.core.fp8_utils import register_selective_fp8_init_decision
    except ImportError:
        return

    register_selective_fp8_init_decision(selective_fp8_init_decision)


def parse_train_args(args_defaults={}):
    """parse arguments for training"""
    args, hydra_cfg = parse_arguments(
        extra_args_provider=loongforge_extra_train_args_provider,
        validate_extra_args_provider=validate_loongforge_extra_args,
        args_defaults=args_defaults,
        parse_unknown_args=True,
    )
    set_hydra_config(hydra_cfg)
    # Apply data/model config to args early so downstream init sees hydrated values

    return args