# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Megatron engine -- public API surface.

Only the global-state accessors, which many callers in training/ use, are re-exported here.
Everything else should be imported from its canonical module (e.g. engines.megatron.model_setup).
"""

from . import global_vars
from .global_vars import (
    get_args,
    get_chat_template,
    get_model_config,
    get_tokenizer,
)

__all__ = [
    "global_vars",
    "get_args",
    "get_chat_template",
    "get_model_config",
    "get_tokenizer",
]
