"""utils"""

import os
import torch
from pathlib import Path
from typing import Optional
from importlib.metadata import version
from packaging.version import Version as PkgVersion

from .constants import DEFAULT_DATASET_CONFIG

try:
    _torch_version = PkgVersion(torch.__version__)
except Exception:
    # This is a WAR for building docs, where torch is not actually imported
    _torch_version = PkgVersion("0.0.0")

_te_version = None


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


def get_torch_version():
    """Get pytorch version from __version__; if not available use pip's. Use caching."""

    def get_torch_version_str():
        import torch

        if hasattr(torch, '__version__'):
            return str(torch.__version__)
        else:
            return version("torch")

    global _torch_version
    if _torch_version is None:
        _torch_version = PkgVersion(get_torch_version_str())
    return _torch_version


def is_te_min_version(version, check_equality=True):
    """Check if minimum version of `transformer-engine` is installed."""
    if check_equality:
        return get_te_version() >= PkgVersion(version)
    return get_te_version() > PkgVersion(version)


def is_torch_min_version(version, check_equality=True):
    """Check if minimum version of `torch` is installed."""
    if check_equality:
        return get_torch_version() >= PkgVersion(version)
    return get_torch_version() > PkgVersion(version)


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


def get_device_arch_version():
    """Returns GPU arch version (8: Ampere, 9: Hopper, 10: Blackwell, ...)"""
    return torch.cuda.get_device_properties(torch.device("cuda:0")).major