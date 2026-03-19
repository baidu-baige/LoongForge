# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0

"""baige omni train module"""

from .parser import parse_train_args, parse_args_from_config, parse_args_from_config
from .trainer_builder import build_model_trainer

from .pretrain import pretrain_llm, pretrain_vlm

try:
    from .embodied import sft_pi05
except ImportError:
    sft_pi05 = None

from .sft import sft_llm, sft_vlm, sft_internvl, sft_ernie
from .diffusion import pretrain_wan


__all__ = ["parse_train_args", "build_model_trainer" "parse_args_from_config", "parse_args_from_config"]
