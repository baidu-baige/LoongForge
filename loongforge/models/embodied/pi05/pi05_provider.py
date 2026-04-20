# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""pi05 model provider."""

from __future__ import annotations

import torch

from megatron.training import get_args
from loongforge.models.factory import register_model_provider
from loongforge.utils.constants import VisionLanguageActionModelFamilies
from loongforge.utils.global_vars import get_model_config

from .modeling_pi05 import PI05Policy


@register_model_provider(model_family=[VisionLanguageActionModelFamilies.PI05])
def pi05_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: int | None = None,
    config=None,
):
    """Build the pi05 policy.

    The pi05 policy is self contained (vision + language + action heads) so we
    simply materialize the config and hand it to the policy implementation
    copied from LeRobot.
    """
    model_config = config if config is not None else get_model_config()
    if model_config is None:
        raise ValueError("pi05 model config was not initialized; pass a config or use --config-file")

    # Default device placement mirrors the hf helpers: prefer CUDA when available.
    if getattr(model_config, "device", None) is None:
        model_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    args = get_args()
    if args.ckpt_format == "torch":
        model = PI05Policy.from_pretrained(args.load, config=model_config)
    else:
        model = PI05Policy(model_config)
    return model
