# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge train module"""

# MCore owns these registration side effects; Native imports only its selected model.
from loongforge.models import llm as _llm  # noqa: F401
from loongforge.models import vision as _vision  # noqa: F401
from loongforge.models import diffusion as _diffusion  # noqa: F401

from .parser import parse_train_args, parse_args_from_config
from .trainer_builder import build_model_trainer

from .pretrain import pretrain_llm, pretrain_vlm

from .sft import sft_llm, sft_vlm, sft_internvl, sft_ernie
from .diffusion import pretrain_wan
from .diffusion import pretrain_qwen_image


__all__ = ["parse_train_args", "build_model_trainer", "parse_args_from_config"]
