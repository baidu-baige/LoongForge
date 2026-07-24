# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model factory registry for the LoongForge eval server.

Usage::

    # Register a factory (in e.g. pi05_factory.py)
    from loongforge.embodied.eval.factories.registry import register_factory
    from loongforge.embodied.eval.servers.eval_server_config import EvalServerArgs
    from loongforge.embodied.eval.servers.loongforge_policy import PredictActionModelSpec

    @register_factory("pi05")
    class PI05ModelFactory:
        model_config_cls = Pi05ModelConfig

        @classmethod
        def build(cls, model_cfg, server_args: EvalServerArgs) -> PredictActionModelSpec: ...

    # Build from server main()
    from loongforge.embodied.eval.factories.registry import build_model_spec
    model_spec = build_model_spec(server_args, raw_model_dict)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Type

from loongforge.embodied.eval.servers.eval_server_config import EvalServerArgs
from loongforge.embodied.eval.servers.loongforge_policy import PredictActionModelSpec

MODEL_FACTORY_REGISTRY: Dict[str, Type] = {}
MODEL_CONFIG_REGISTRY: Dict[str, Type] = {}


def register_factory(model_type: str):
    """Decorator that registers a model factory class into MODEL_FACTORY_REGISTRY.

    If the factory class has a ``model_config_cls`` attribute, it is also
    registered into MODEL_CONFIG_REGISTRY for OmegaConf-based config construction.

    Args:
        model_type: String key to register the factory under (e.g. ``"pi05"``).

    Returns:
        Class decorator that registers the decorated class and returns it unchanged.
    """
    def decorator(cls):
        """Register cls into MODEL_FACTORY_REGISTRY and MODEL_CONFIG_REGISTRY."""
        MODEL_FACTORY_REGISTRY[model_type] = cls
        if hasattr(cls, "model_config_cls"):
            MODEL_CONFIG_REGISTRY[model_type] = cls.model_config_cls
        return cls
    return decorator


def _auto_import_factory_modules() -> None:
    """Import known factory modules to trigger @register_factory decorators."""
    import importlib

    _FACTORY_MODULES = [
        "loongforge.embodied.eval.factories.pi05_factory",
        "loongforge.embodied.eval.factories.xvla_factory",
    ]
    for mod in _FACTORY_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            logging.warning("Failed to import factory module %s: %s", mod, e)


def build_model_config(model_type: str, raw_model_dict: Dict[str, Any]) -> Any:
    """Build a typed ModelConfig by merging raw YAML model dict over dataclass defaults.

    Args:
        model_type: Registered model type string (e.g. ``"pi05"``).
        raw_model_dict: The ``model:`` section dict from the eval YAML config.

    Returns:
        A frozen typed ModelConfig instance (e.g. ``Pi05ModelConfig``).

    Raises:
        KeyError: If model_type has no registered config class.
    """
    from omegaconf import OmegaConf

    _auto_import_factory_modules()
    if model_type not in MODEL_CONFIG_REGISTRY:
        raise KeyError(
            f"No model_config_cls registered for model_type={model_type!r}. "
            f"Registered: {sorted(MODEL_CONFIG_REGISTRY.keys())}"
        )
    config_cls = MODEL_CONFIG_REGISTRY[model_type]
    import dataclasses
    known_fields = {f.name for f in dataclasses.fields(config_cls)}
    filtered = {k: v for k, v in raw_model_dict.items() if k in known_fields}
    merged = OmegaConf.merge(OmegaConf.structured(config_cls), filtered)
    return OmegaConf.to_object(merged)


def build_model_spec(
    server_args: EvalServerArgs,
    raw_model_dict: Dict[str, Any],
) -> PredictActionModelSpec:
    """Build a PredictActionModelSpec using the registered factory.

    Args:
        server_args: Typed EvalServerArgs with runtime/infra options.
        raw_model_dict: The raw ``model:`` section dict from the eval YAML.

    Returns:
        PredictActionModelSpec with a loaded model ready for eval.

    Raises:
        SystemExit: If model_type is not registered.
    """
    _auto_import_factory_modules()

    model_type = server_args.model_type
    if model_type not in MODEL_FACTORY_REGISTRY:
        raise SystemExit(
            f"Unsupported model_type: {model_type!r}. "
            f"Registered: {sorted(MODEL_FACTORY_REGISTRY.keys())}"
        )

    model_cfg = build_model_config(model_type, raw_model_dict)
    factory_cls = MODEL_FACTORY_REGISTRY[model_type]
    return factory_cls.build(model_cfg, server_args)
