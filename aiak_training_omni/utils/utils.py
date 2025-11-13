"""utils"""

import os
import torch
from copy import deepcopy
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from pathlib import Path
from typing import Optional, Tuple
from importlib.metadata import version
from packaging.version import Version as PkgVersion

from .constants import DEFAULT_DATASET_CONFIG
from megatron.core.transformer import TransformerConfig
from megatron.training.activations import squared_relu
from megatron.training.arguments import moe_freq_type
import torch.nn.functional as F
from aiak_training_omni.utils import constants

_te_version = None


def convert_megatron_transformer_config_args(megatron_args):
    """convert megatron args to transformer config"""
    transformer_config_args = {}
    for k, v in megatron_args.items():
        if k in TransformerConfig.__dataclass_fields__:
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
    return transformer_config_args


def flatten_foundation_model(config):
    """
    Detect the config.model.foundation field. If there is a model field under it, 
    move the contents of model up one level:
    e.g. from config.model.foundation.model.xxx to config.model.foundation.xxx.
    """
    # Check if config.model.foundation exists
    if not hasattr(config.model, 'foundation'):
        return config
    elif hasattr(config.model, 'foundation') and config.model.foundation is not None:
        foundation = config.model.foundation

        if hasattr(foundation, 'model') and foundation.model is not None:
            with open_dict(foundation):
                model_dict = OmegaConf.to_container(
                    foundation.model, resolve=False)

                del foundation['model']

                for key, value in model_dict.items():
                    foundation[key] = value

    return config


def transform_overrides_for_nested_model(hydra_overrides):
    """
    Convert flattened override paths to nested paths
    model.foundation.xxx -> model.foundation.model.xxx
    """
    if hydra_overrides is None:
        return hydra_overrides
    
    transformed = []
    for override in hydra_overrides:
        # Check if the override starts with model.foundation
        if override.startswith('model.foundation.') and not override.startswith('model.foundation.model.'):
            # Extract the prefix and the remaining part
            # For example: model.foundation.num_layers=32
            parts = override.split('=', 1)  # Split into key and value
            if len(parts) == 2:
                key, value = parts
                # Convert model.foundation.xxx to model.foundation.model.xxx
                key = key.replace('model.foundation.',
                                  'model.foundation.model.', 1)
                transformed.append(f"{key}={value}")
            else:
                transformed.append(override)
        else:
            transformed.append(override)

    return transformed


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
        # Filter out empty strings
        hydra_overrides = [o for o in hydra_overrides if o.strip()]
        # Change hydra_overrides to nested format
        transformed_overrides = transform_overrides_for_nested_model(
            hydra_overrides)

        # Initialize using Hydra's initialize_config_dir
        with initialize_config_dir(config_dir=config_path, version_base=None):
            # Load configuration using compose, which automatically processes defaults
            config = compose(config_name=config_name,
                             overrides=transformed_overrides)

        config = flatten_foundation_model(config)

        return config

    except Exception as e:
        print(f"Cannot load hydra config: {e}. Config path: {config_path}, \
                config name: {config_name}")
        raise


def build_model_config(args, config):
    """Build model config from args and config"""
    args_dict = deepcopy(vars(args))

    model_cfgs = {}

    if not hasattr(config, "model"):
        raise ValueError("Invalid model configuration structure")

    assert hasattr(config.model, "model_type"), "model_type is required in model config"
    if config.model.model_type in constants.VisionLanguageModelFamilies.names():
        from aiak_training_omni.models.common.vlm_model_config import VLMModelConfig

        model_config = config.model
        for name, config_values in model_config.items():
            # must have _target_ field
            if "_target_" in config_values:
                merged = deepcopy(args_dict)
                merged.update(OmegaConf.to_container(
                    config_values, resolve=True))
                merged = convert_megatron_transformer_config_args(merged)
                model_cfgs[name] = instantiate(config_values, **merged)
            else:
                model_cfgs[name] = config_values
        model_cfgs = VLMModelConfig(**model_cfgs)
    elif model_config.model_type in constants.LanguageModelFamilies.names():
        # Language model
        if "_target_" not in model_config:
            raise ValueError(
                "Model config 'model' missing '_target_' field for llm type.\n")

        merged = deepcopy(args_dict)
        merged.update(OmegaConf.to_container(model_config, resolve=True))
        merged = convert_megatron_transformer_config_args(merged)

        model_cfgs = instantiate(model_config, **merged)
    elif model_config.model_type in constants.VideoLanguageModelFamilies.names():
        merged = deepcopy(args_dict)
        merged.update(OmegaConf.to_container(model_config, resolve=True))
        model_cfgs = instantiate(model_config, **merged)
    else:
        raise ValueError(f"Unsupported model type: {model_config.model_type}")
    return model_cfgs


def register_custom_resolvers():
    """
    To resolve parameters that cannot be directly mapped to the specified type by YAML.
    """
    # Activation functions
    ACTIVATION_MAP = {
        "relu": F.relu,
        "gelu": F.gelu,
        "silu": F.silu,
    }

    OmegaConf.register_new_resolver(
        "act", lambda name: ACTIVATION_MAP[name.lower()], replace=True
    )


def import_module(module_path: Tuple[str], config: TransformerConfig):
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
    return vars(module)[name](config)


def print_rank_0(message, rank=None):
    """print rank 0"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        if rank is None or rank == 0:
            print(message, flush=True)


def build_transformer_config(args, config_class=None):
    """create transformer config from args"""
    from megatron.training.arguments import core_transformer_config_from_args

    config = core_transformer_config_from_args(args, config_class=config_class)
    return config


def get_default_sft_dataset_config() -> Optional[str]:
    """get default sft dataset config"""
    default_config = str(
        Path(__file__).parent.parent.parent
        / "configs"
        / "data"
        / DEFAULT_DATASET_CONFIG
    )
    if os.path.exists(default_config):
        return default_config

    return None


def get_te_version():
    """Get TE version from __version__; if not available use pip's. Use caching."""

    def get_te_version_str():
        import transformer_engine as te

        if hasattr(te, "__version__"):
            return str(te.__version__)
        else:
            return version("transformer-engine")

    global _te_version
    if _te_version is None:
        _te_version = PkgVersion(get_te_version_str())
    return _te_version


def is_te_min_version(version, check_equality=True):
    """Check if minimum version of `transformer-engine` is installed."""
    if check_equality:
        return get_te_version() >= PkgVersion(version)
    return get_te_version() > PkgVersion(version)
