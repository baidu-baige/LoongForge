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
import logging
import pkgutil
from typing import Dict, Type

import torch.nn as nn

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(model_type: str):
    """Decorator that registers a model class into MODEL_REGISTRY."""
    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator


def _auto_import_model_modules():
    """Auto-import all sub-packages under model/ to trigger @register_model decorators."""
    import loongforge.embodied.model as _model_pkg
    import os

    model_dir = os.path.dirname(_model_pkg.__file__)
    for _, pkg_name, is_pkg in pkgutil.iter_modules([model_dir]):
        if is_pkg and pkg_name not in ("compose", "modules", "__pycache__"):
            try:
                # import modeling_<pkg_name>.py to trigger registration
                modeling_mod = f"loongforge.embodied.model.{pkg_name}.modeling_{pkg_name}"
                importlib.import_module(modeling_mod)
            except ImportError as exc:
                # Lazy import: keeps the module-level ``register_model``
                # decorator path free of the training-side distributed stack.
                from loongforge.embodied.distributed.utils import is_rank_zero
                if is_rank_zero():   
                    logger.warning(
                        "Skipping optional model package during auto-registration:\n"
                        "  - package: %s\n"
                        "  - module: %s\n"
                        "  - error: %s\n"
                        "If this run is not training the '%s' model, this warning "
                        "can be ignored. If you intend to train it, fix the missing "
                        "or incompatible dependency.",
                        pkg_name,
                        modeling_mod,
                        exc,
                        pkg_name,
                    )


def build_model(model_cfg) -> nn.Module:
    """Build a model instance by model_cfg.model_type.

    Args:
        model_cfg: typed ModelConfig instance (must have a ``model_type`` attribute).

    Returns:
        Initialized nn.Module.
    """
    _auto_import_model_modules()

    model_type = model_cfg.model_type
    if model_type not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type: '{model_type}'. "
            f"Registered: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[model_type]
    return cls.from_pretrained(model_cfg)
