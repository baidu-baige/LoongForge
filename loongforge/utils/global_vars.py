# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Global Variables"""

from typing import TYPE_CHECKING, Optional

from megatron.training import get_args as _get_args
from megatron.training.global_vars import (
    _ensure_var_is_initialized,
    _ensure_var_is_not_initialized,
)

from loongforge.tokenizer import build_tokenizer
from loongforge.data import ChatTemplate

from .constants import TrainingPhase


if TYPE_CHECKING:
    from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer


_GLOBAL_CHAT_TEMPLATE: Optional["ChatTemplate"] = None
_GLOBAL_LOONGFORGE_TOKENIZER: Optional["MegatronLegacyTokenizer"] = None
_GLOBAL_MODEL_CONFIG = None
_GLOBAL_HYDRA_CONFIG = None
_GLOBAL_DATA_CONFIG = None
_GLOBAL_ARGS_DICT = None


def get_model_config():
    """Get the model configuration"""
    return _GLOBAL_MODEL_CONFIG


def get_hydra_config():
    """Get the hydra configuration"""
    return _GLOBAL_HYDRA_CONFIG


def get_data_config():
    """Get the data configuration"""
    return _GLOBAL_DATA_CONFIG


def get_args_dict():
    """Get the args dictionary"""
    return _GLOBAL_ARGS_DICT


def set_model_config(model_config):
    """Set the model configuration"""
    global _GLOBAL_MODEL_CONFIG
    _ensure_var_is_not_initialized(_GLOBAL_MODEL_CONFIG, "model config")
    _GLOBAL_MODEL_CONFIG = model_config


def set_hydra_config(hydra_config):
    """Set the hydra configuration"""
    global _GLOBAL_HYDRA_CONFIG
    _ensure_var_is_not_initialized(_GLOBAL_HYDRA_CONFIG, "hydra config")
    _GLOBAL_HYDRA_CONFIG = hydra_config


def set_data_config(data_config):
    """Set the data configuration"""
    global _GLOBAL_DATA_CONFIG
    _ensure_var_is_not_initialized(_GLOBAL_DATA_CONFIG, "data config")
    _GLOBAL_DATA_CONFIG = data_config


def set_args_dict(args_dict):
    """Set the args dictionary"""
    global _GLOBAL_ARGS_DICT
    _ensure_var_is_not_initialized(_GLOBAL_ARGS_DICT, "args dict")
    _GLOBAL_ARGS_DICT = args_dict


def set_loongforge_extra_global_vars(args, build_tokenizer=True) -> None:
    """Set LOONGFORGE extra global variables"""
    assert args is not None
    if build_tokenizer:
        _ = _build_chat_template(args)
        _ = _build_tokenizer(args)


def _build_chat_template(args) -> Optional["ChatTemplate"]:
    """Build the chat template."""
    if args.training_phase == TrainingPhase.SFT and args.chat_template is not None:
        global _GLOBAL_CHAT_TEMPLATE
        _ensure_var_is_not_initialized(_GLOBAL_CHAT_TEMPLATE, "loongforge-chat-template")
        _GLOBAL_CHAT_TEMPLATE = ChatTemplate.from_name(args.chat_template)
        assert (
            _GLOBAL_CHAT_TEMPLATE is not None
        ), f"chat_template {args.chat_template} not supported."
        return _GLOBAL_CHAT_TEMPLATE

    return None


def _build_tokenizer(args) -> Optional["MegatronLegacyTokenizer"]:
    """Initialize tokenizer."""
    global _GLOBAL_LOONGFORGE_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_LOONGFORGE_TOKENIZER, "loongforge-tokenizer")
    _GLOBAL_LOONGFORGE_TOKENIZER = build_tokenizer(args, chat_template=_GLOBAL_CHAT_TEMPLATE)
    return _GLOBAL_LOONGFORGE_TOKENIZER


def get_tokenizer() -> Optional["MegatronLegacyTokenizer"]:
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_LOONGFORGE_TOKENIZER, "loongforge-tokenizer")
    return _GLOBAL_LOONGFORGE_TOKENIZER


def get_chat_template() -> Optional["ChatTemplate"]:
    """Return chat template."""
    _ensure_var_is_initialized(_GLOBAL_CHAT_TEMPLATE, "loongforge-chat-template")
    return _GLOBAL_CHAT_TEMPLATE


def get_args():
    """Return args for Megatron now."""
    return _get_args()
