# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Global model registry — maps model_type strings to model classes.

Usage:
    # Register
    @register_model("pi05")
    class Pi05Model(nn.Module): ...

    # Build
    model = build_model(model_cfg)   # model_cfg.model_type = "pi05"
"""

import importlib
from typing import Dict, Type

import torch.nn as nn

# model_type -> modeling module, relative to ``loongforge.models.embodied.``.
# Only the module of the requested model_type is imported.
MODEL_MODULES: Dict[str, str] = {
    "pi05": "pi05.modeling_pi05",
    "Gr00tN1d6": "groot_n1_6.modeling_groot_n1_6",
    "Gr00tN1d7": "groot_n1_7.modeling_groot_n1_7",
    "xvla": "xvla.modeling_xvla",
    "fastwam": "fastwam.modeling_fastwam",
    "cosmos3": "cosmos3.modeling_cosmos3",
    "dreamzero": "dreamzero.modeling_dreamzero",
    "lingbot_va": "lingbot_va.modeling_lingbot_va",
    "wall_oss_0_5": "wall_oss_0_5.modeling_wall_oss_0_5",
}

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(model_type: str):
    """Decorator that registers a model class into MODEL_REGISTRY."""
    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator


def build_model(model_cfg) -> nn.Module:
    """Build a model instance by model_cfg.model_type.

    Args:
        model_cfg: typed ModelConfig instance (must have a ``model_type`` attribute).

    Returns:
        Initialized nn.Module.
    """
    model_type = model_cfg.model_type
    if model_type not in MODEL_MODULES:
        raise KeyError(
            f"Unknown model_type: '{model_type}'. "
            f"Registered: {sorted(MODEL_MODULES)}"
        )
    importlib.import_module(f"loongforge.models.embodied.{MODEL_MODULES[model_type]}")

    cls = MODEL_REGISTRY[model_type]
    return cls.from_pretrained(model_cfg)
