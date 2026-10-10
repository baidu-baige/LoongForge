# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge MCore engine -- public API surface.

Only high-frequency symbols that are imported by many callers across models/,
data/, and training/ are re-exported here.  Everything else should be imported
from its canonical module (e.g. engines.mcore.model_setup, training.registry).
"""

from . import global_vars
from .global_vars import (
    get_args,
    get_chat_template,
    get_data_config,
    get_model_config,
    get_tokenizer,
)
from .utils import (
    build_transformer_config,
    print_rank_0,
)

__all__ = [
    "global_vars",
    "get_args",
    "get_chat_template",
    "get_data_config",
    "get_model_config",
    "get_tokenizer",
    "build_transformer_config",
    "print_rank_0",
]
