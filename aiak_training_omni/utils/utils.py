"""utils"""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple
from importlib.metadata import version
from packaging.version import Version as PkgVersion

from .constants import DEFAULT_DATASET_CONFIG
from megatron.core.transformer import TransformerConfig
from megatron.training.activations import squared_relu
import torch.nn.functional as F

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
        Path(__file__).parent.parent.parent / "configs" / DEFAULT_DATASET_CONFIG
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
