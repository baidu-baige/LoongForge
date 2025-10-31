"""utils"""
import os
import torch
from pathlib import Path
from typing import Optional
from importlib.metadata import version
from packaging.version import Version as PkgVersion

from .constants import DEFAULT_DATASET_CONFIG

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
    default_config = str(Path(__file__).parent.parent.parent / 'configs' / DEFAULT_DATASET_CONFIG)
    if os.path.exists(default_config):
        return default_config

    return None


def get_te_version():
    """Get TE version from __version__; if not available use pip's. Use caching."""

    def get_te_version_str():
        import transformer_engine as te

        if hasattr(te, '__version__'):
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
