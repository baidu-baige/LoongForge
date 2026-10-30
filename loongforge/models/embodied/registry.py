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

from __future__ import annotations

import importlib
from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

# model_type -> package name. The same name is used under ``loongforge.models.embodied``
# (``<package>/modeling_<package>.py``) and ``loongforge.data.embodied`` (``<package>/__init__.py``).
MODEL_PACKAGES: Dict[str, str] = {
    "vlda": "vlda",
    "pi05": "pi05",
    "Gr00tN1d6": "gr00t_n1_6",
    "Gr00tN1d7": "gr00t_n1_7",
    "xvla": "xvla",
    "fastwam": "fastwam",
    "cosmos3": "cosmos3",
    "dreamzero": "dreamzero",
    "lingbot_va": "lingbot_va",
    "wall_oss_0_5": "wall_oss_0_5",
}

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(model_type: str):
    """Decorator that registers a model class into MODEL_REGISTRY."""
    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator


def build_model(model_cfg, training_args=None) -> nn.Module:
    """Build a model instance by model_cfg.model_type.

    Args:
        model_cfg: typed ModelConfig instance (must have a ``model_type`` attribute).

    Returns:
        Initialized nn.Module.
    """
    model_type = model_cfg.model_type
    if model_type not in MODEL_PACKAGES:
        raise KeyError(
            f"Unknown model_type: '{model_type}'. "
            f"Registered: {sorted(MODEL_PACKAGES)}"
        )
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    package = MODEL_PACKAGES[model_type]
    importlib.import_module(f"loongforge.models.embodied.{package}.modeling_{package}")

    cls = MODEL_REGISTRY[model_type]
    if hasattr(cls, "from_training_args"):
        if training_args is None:
            raise ValueError(f"model_type '{model_type}' needs training args to build.")
        return cls.from_training_args(model_cfg, training_args)
    return cls.from_pretrained(model_cfg)
