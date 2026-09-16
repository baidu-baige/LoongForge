# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Native model registry — maps model_type strings to model classes.

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

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(model_type: str):
    """Decorator that registers a model class into MODEL_REGISTRY."""
    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator


_MODEL_MODULES = {
    "pi05": "loongforge.models.vla.pi05.modeling_pi05",
    "gr00tn1d6": "loongforge.models.vla.groot_n1_6.modeling_groot_n1_6",
    "gr00tn1d7": "loongforge.models.vla.groot_n1_7.modeling_groot_n1_7",
    "xvla": "loongforge.models.vla.xvla.modeling_xvla",
    "walloss05": "loongforge.models.vla.wall_oss_0_5.modeling_wall_oss_0_5",
    "cosmos3": "loongforge.models.world.cosmos3.modeling_cosmos3",
    "dreamzero": "loongforge.models.world.dreamzero.modeling_dreamzero",
    "fastwam": "loongforge.models.world.fastwam.modeling_fastwam",
    "lingbotva": "loongforge.models.world.lingbot_va.modeling_lingbot_va",
}


def _auto_import_model_modules(model_type: str):
    """Import only the selected model so dependency errors stay visible."""
    key = model_type.lower().replace("-", "").replace("_", "")
    try:
        importlib.import_module(_MODEL_MODULES[key])
    except KeyError as exc:
        raise KeyError(f"Unknown native model_type: {model_type}") from exc


def build_model(model_cfg) -> nn.Module:
    """Build a model instance by model_cfg.model_type.

    Args:
        model_cfg: typed ModelConfig instance (must have a ``model_type`` attribute).

    Returns:
        Initialized nn.Module.
    """
    model_type = model_cfg.model_type
    _auto_import_model_modules(model_type)
    if model_type not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type: '{model_type}'. "
            f"Registered: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[model_type]
    return cls.from_pretrained(model_cfg)
