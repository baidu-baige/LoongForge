# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model-type registry for embodied data: per-sample transforms, collators, samplers,
dataset formats and build strategies.

Each model package ``loongforge.data.embodied.<package>`` registers its builders
with the decorators below when imported. Lookups import only the package listed
in ``MODEL_PACKAGES`` for the requested ``model_type``, so one broken or heavy
model package never affects the others.

Adding a model: create ``data/embodied/<package>/`` (same name as the model
package) whose ``__init__`` imports its transform / collator / (optional)
sampler modules, add one entry to ``MODEL_PACKAGES`` in
``loongforge/data/embodied/registry.py``, and (if the model needs a custom
dataset build) one entry to ``DATASET_STRATEGIES``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from loongforge.data.embodied.transforms.base import BaseTransform, ComposedTransform

# model_type -> package name under ``loongforge.data.embodied``.
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
    "lingbot_vla_v2": "lingbot_vla_v2",
    "wall_oss_0_5": "wall_oss_0_5",
}

# Data-only model types that map to a plain module under ``loongforge.data.embodied``.
# Importing that module must register the model's builders.
PLAIN_MODULES: Dict[str, str] = {
    "dummy": "transforms.collator",
}

_KNOWN_MODEL_TYPES = frozenset({*MODEL_PACKAGES, *PLAIN_MODULES})

# dataset_strategy name -> "module:callable" string.  Resolved lazily so that
# lerobot / model-specific deps only load when actually selected.  The
# ``"default"`` strategy is the stock lerobot build; model-specific strategies
# plug custom multi-frame geometry into the same generic dataset classes.
DATASET_STRATEGIES: Dict[str, str] = {
    "default": "datasets.lerobot_dataset:build_default_lerobot_dataset",
    "vlda": "vlda:build_vlda_dataset",
    "fastwam": "fastwam.fastwam_dataset:build_fastwam_lerobot_dataset",
    "lingbot_va": "lingbot_va.latent_lerobot_dataset:build_lingbot_dataset",
    "lingbot_vla_v2": "lingbot_vla_v2.lingbot_vla_v2_dataset:build_lingbot_vla_v2_dataset",
    "gr00t_n1_7": "gr00t_n1_7.gr00t_n1_7_dataset:build_groot_n1_7_lerobot_dataset",
    "cosmos3_droid": "cosmos3.droid_lerobot_dataset:build_droid_dataset",
    "dreamzero": "dreamzero.builder:build_dreamzero_dataset",
    "wall_oss_0_5": "wall_oss_0_5.lerobot_dataset:build_wall_oss_0_5_lerobot_dataset",
}

# --dataset-format -> "module:callable" (relative to ``loongforge.data.embodied``).
# "lerobot_datasets" further dispatches on --dataset-strategy through DATASET_STRATEGIES.
DATASET_FORMATS: Dict[str, str] = {
    "lerobot_datasets": "registry:build_dataset_by_strategy",
    "hdf5_datasets": "datasets.hdf5_dataset:build_hdf5_dataset",
    "dummy_datasets": "datasets.dummy_dataset:build_dummy_dataset",
}


@dataclass(frozen=True)
class TransformBuilderContext:
    """Shared inputs available to model-specific transform builders."""

    model_cfg: Any
    data_cfg: Any
    training_args: Any
    dataset: Any
    dataset_stats: dict[str, Any] | None


TransformBuilder = Callable[[TransformBuilderContext], Iterable["BaseTransform"]]

_TRANSFORM_BUILDERS: Dict[str, TransformBuilder] = {}
_PREPROCESSOR_REGISTRY: Dict[str, Type] = {}
_SAMPLER_BUILDERS: Dict[str, Callable] = {}
_IMPORTED: set[str] = set()


def _register(table: Dict[str, Any], name: str):
    def decorator(obj):
        table[name] = obj
        return obj

    return decorator


def register_transform_builder(model_type: str):
    """Decorator to register a model-specific per-sample transform builder."""
    return _register(_TRANSFORM_BUILDERS, model_type)


def register_preprocessor(name: str):
    """Decorator to register a preprocessor class for a model."""
    return _register(_PREPROCESSOR_REGISTRY, name)


def register_sampler_builder(model_type: str):
    """Decorator to register a model-specific sampler builder."""
    return _register(_SAMPLER_BUILDERS, model_type)


def _data_module(model_type: str) -> Optional[str]:
    """Return the ``loongforge.data.embodied`` module that registers ``model_type``."""
    return PLAIN_MODULES.get(model_type) or MODEL_PACKAGES.get(model_type)


def _ensure_imported(model_type: str) -> None:
    """Import the package that registers builders for ``model_type``."""
    module = _data_module(model_type)
    if model_type in _IMPORTED or module is None:
        return
    importlib.import_module(f"loongforge.data.embodied.{module}")
    _IMPORTED.add(model_type)


def get_transform_builder(model_type: str) -> TransformBuilder:
    """Look up the per-sample transform builder for ``model_type``."""
    _ensure_imported(model_type)
    if model_type not in _TRANSFORM_BUILDERS:
        raise ValueError(
            f"Unknown transform builder for model_type '{model_type}'. "
            f"Known model types: {sorted(_KNOWN_MODEL_TYPES)}"
        )
    return _TRANSFORM_BUILDERS[model_type]


def get_preprocessor(name: str) -> Type:
    """Look up a registered preprocessor class by name."""
    _ensure_imported(name)
    if name not in _PREPROCESSOR_REGISTRY:
        raise ValueError(
            f"Unknown preprocessor '{name}'. "
            f"Available: {list(_PREPROCESSOR_REGISTRY.keys())}"
        )
    return _PREPROCESSOR_REGISTRY[name]


def get_sampler_builder(model_type: str) -> Callable:
    """Look up a sampler builder by model type, falling back to the default."""
    _ensure_imported(model_type)
    from loongforge.data.embodied.sampler_builder import default_sampler_builder

    return _SAMPLER_BUILDERS.get(model_type, default_sampler_builder)


def build_preprocessor(
    name: str,
    model_cfg,
    data_cfg,
    training_args=None,
    dataset_stats=None,
    dataset=None,
):
    """Instantiate a registered preprocessor via its from_config classmethod."""
    return get_preprocessor(name).from_config(
        model_cfg,
        data_cfg,
        training_args=training_args,
        dataset_stats=dataset_stats,
        dataset=dataset,
    )


def build_transforms_from_args(
    model_cfg,
    data_cfg,
    training_args,
    dataset,
    dataset_stats,
) -> Optional["ComposedTransform"]:
    """Build per-sample transforms for ``model_cfg.model_type``; None if there are none."""
    from loongforge.data.embodied.transforms.base import ComposedTransform

    model_type = model_cfg.model_type
    if not model_type:
        return None

    ctx = TransformBuilderContext(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        training_args=training_args,
        dataset=dataset,
        dataset_stats=dataset_stats,
    )
    transforms = list(get_transform_builder(model_type)(ctx))
    if not transforms:
        return None
    return ComposedTransform(transforms)


def build_dataset_by_strategy(model_cfg, data_cfg, training_args):
    """Build a lerobot dataset via the strategy named by ``training_args.dataset_strategy``.

    Falls back to ``"default"`` when the field is absent, empty, or names a
    dataset that has no dedicated strategy.
    """
    name = training_args.dataset_strategy or "default"
    entry = DATASET_STRATEGIES.get(name, DATASET_STRATEGIES["default"])
    module_rel, func_name = entry.split(":")
    mod = importlib.import_module(f"loongforge.data.embodied.{module_rel}")
    return getattr(mod, func_name)(model_cfg, data_cfg, training_args)


def build_dataset(model_cfg, data_cfg, training_args):
    """Build the dataset named by ``training_args.dataset_format``.

    Lerobot datasets dispatch further on ``training_args.dataset_strategy``
    through ``DATASET_STRATEGIES``.
    """
    name = training_args.dataset_format
    if name not in DATASET_FORMATS:
        raise ValueError(
            f"Unknown dataset_format: '{name}'. Supported: {sorted(DATASET_FORMATS)}"
        )
    module_rel, func_name = DATASET_FORMATS[name].split(":")
    mod = importlib.import_module(f"loongforge.data.embodied.{module_rel}")
    return getattr(mod, func_name)(model_cfg, data_cfg, training_args)
