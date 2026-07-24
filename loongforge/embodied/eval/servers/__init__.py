# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from .mock_policy import MockPolicy
from .loongforge_policy import (
    GenericPredictActionPolicy,
    PredictActionModelSpec,
)
from .predict_action_interface import PredictActionModel, call_predict_action, validate_predict_action_model


def __getattr__(name):
    # Lazy re-exports to avoid circular imports between servers/ and factories/.
    if name in ("PI05ModelFactory", "LoongForgePI05Policy"):
        from loongforge.embodied.eval.factories.pi05_factory import PI05ModelFactory, LoongForgePI05Policy
        return {"PI05ModelFactory": PI05ModelFactory, "LoongForgePI05Policy": LoongForgePI05Policy}[name]
    if name in ("MODEL_FACTORY_REGISTRY", "register_factory", "build_model_spec"):
        import loongforge.embodied.eval.factories.registry as _reg
        return getattr(_reg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MockPolicy",
    "GenericPredictActionPolicy",
    "LoongForgePI05Policy",
    "PI05ModelFactory",
    "PredictActionModelSpec",
    "MODEL_FACTORY_REGISTRY",
    "register_factory",
    "build_model_spec",
    "PredictActionModel",
    "call_predict_action",
    "validate_predict_action_model",
]
