"""Gr00tN1d6 model provider."""

from __future__ import annotations

import torch

from loongforge.models.factory import register_model_provider
from loongforge.utils.constants import VisionLanguageActionModelFamilies
from loongforge.utils.global_vars import get_model_config

from .configuration_groot import Gr00tN1d6OmniConfig
from .modeling_groot import Gr00tN1d6


@register_model_provider(model_family=[VisionLanguageActionModelFamilies.GROOT_N1_6])
def groot_n1_6_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: int | None = None,
    config=None,
):
    """Build the Gr00tN1d6 model for LoongForge."""

    model_config = config if config is not None else get_model_config()
    if model_config is None:
        raise ValueError("groot_n1_6 model config was not initialized; pass a config or use --config-file")

    if not isinstance(model_config, Gr00tN1d6OmniConfig):
        # Allow loading plain Gr00tN1d6Config and wrap it for LoongForge defaults.
        model_config = Gr00tN1d6OmniConfig(**model_config.__dict__)

    if getattr(model_config, "device", None) is None:
        model_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Gr00tN1d6(model_config)

    # Move to device only. No dtype manipulation needed here because:
    # - EagleBackbone.__init__ loads frozen params in bf16 (load_bf16=True)
    #   and casts trainable params to fp32 (backbone_trainable_params_fp32=True)
    # - Action head initializes in fp32 by default
    # - Float16Module (applied later in training_utils) will cast all to bf16,
    #   then use_fp32_dtype_for_param_pattern restores trainable params to fp32.
    model = model.to(device=model_config.device)

    print("====== Gr00tN1d6 Model Structure ======")
    print(model)
    print("=======================================")

    return model
