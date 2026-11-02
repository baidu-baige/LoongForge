# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Megatron runtime accessors and shared utilities."""

from .utils import xpu_init
from .utils import global_vars
from .utils.runtime import (
    build_transformer_config,
    convert_custom_pipeline_to_layout,
    get_device_arch_version,
    is_te_min_version,
    is_torch_min_version,
    print_rank_0,
)
from .utils.global_vars import (
    get_args,
    get_chat_template,
    get_data_config,
    get_model_config,
    get_tokenizer,
)

__all__ = [
    "global_vars",
    "get_args",
    "get_chat_template",
    "get_data_config",
    "get_model_config",
    "get_tokenizer",
    "build_transformer_config",
    "convert_custom_pipeline_to_layout",
    "get_device_arch_version",
    "is_te_min_version",
    "is_torch_min_version",
    "print_rank_0",
]
