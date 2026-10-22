# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model-type registry for embodied data: per-sample transforms, collators, samplers,
dataset formats and build strategies, and the ``DataRuntime`` contract they receive.

Each model package ``loongforge.data.embodied.<package>`` registers its builders
with the decorators below when imported. Lookups import only the package listed
in ``MODEL_MODULES`` for the requested ``model_type``, so one broken or heavy
model package never affects the others.

Adding a model: create ``data/embodied/<package>/`` whose ``__init__`` imports
its transform / collator / (optional) sampler modules, add one entry to
``MODEL_MODULES``, and (if the model needs a custom dataset build) one entry
to ``DATASET_STRATEGIES``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from loongforge.data.embodied.transforms.base import ComposedTransform

# model_type (from the model YAML) -> directory name under ``loongforge.data.embodied``.
# Importing that package must register the model's builders. "dummy" points at
# the plain module ``collator`` that defines ``DummyCollator``.
MODEL_MODULES: Dict[str, str] = {
    "vlda": "vlda",
    "pi05": "pi05",
    "Gr00tN1d6": "groot_n1_6",
    "Gr00tN1d7": "groot_n1_7",
    "xvla": "xvla",
    "fastwam": "fastwam",
    "cosmos3": "cosmos3",
    "dreamzero": "dreamzero",
    "lingbot_va": "lingbot_va",
    "wall_oss_0_5": "wall_oss_0_5",
    "dummy": "collator",
}

# dataset_strategy name -> "module:callable" string.  Resolved lazily so that
# lerobot / model-specific deps only load when actually selected.  The
# ``"default"`` strategy is the stock lerobot build; model-specific strategies
# plug custom multi-frame geometry into the same generic dataset classes.
DATASET_STRATEGIES: Dict[str, str] = {
    "default": "datasets.lerobot_dataset:build_default_lerobot_dataset",
    "vlda": "vlda.vlda_dataset:build_vlda_dataset",
    "fastwam": "fastwam.fastwam_dataset:build_fastwam_lerobot_dataset",
    "lingbot_va": "lingbot_va.lingbot_va_dataset:build_lingbot_dataset",
    "groot_n1_7": "groot_n1_7.groot_n1_7_dataset:build_groot_n1_7_lerobot_dataset",
    "cosmos3_droid": "cosmos3.cosmos3_dataset:build_droid_dataset",
    "dreamzero": "dreamzero.dreamzero_dataset:build_dreamzero_dataset",
    "wall_oss_0_5": "wall_oss_0_5.wall_oss_0_5_dataset:build_wall_oss_0_5_lerobot_dataset",
}

# --dataset-format -> "module:callable" (relative to ``loongforge.data.embodied``).
# "lerobot_datasets" further dispatches on --dataset-strategy through DATASET_STRATEGIES.
DATASET_FORMATS: Dict[str, str] = {
    "lerobot_datasets": "registry:build_dataset_by_strategy",
    "hdf5_datasets": "xvla.xvla_dataset:build_hdf5_dataset",
    "dummy_datasets": "datasets.dummy_dataset:build_dummy_dataset",
}


@dataclass(frozen=True)
class DataRuntime:
    """Execution facts the training method resolves for embodied data components.

    Data code reads these instead of backend arguments or the backend's
    distributed context, so it stays independent of the engine schema.
    """

    rank: int = 0
    world_size: int = 1
    is_distributed: bool = False
    grad_accum_steps: int = 1
    deterministic: bool = False
    graph_enabled: bool = False
    graph_pad_length: int | None = None


@dataclass(frozen=True)
class TransformBuilderContext:
    """Shared inputs available to model-specific transform builders."""

    model_cfg: Any
    data_cfg: Any
    training_args: Any
    dataset: Any
    dataset_stats: dict[str, Any] | None
    runtime: DataRuntime


TransformBuilder = Callable[["TransformBuilderContext"], Iterable[Any]]

_TRANSFORM_BUILDERS: Dict[str, TransformBuilder] = {}
_COLLATORS: Dict[str, Type] = {}
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


def register_collator(model_type: str):
    """Decorator to register a model-specific collator (``BaseCollator`` subclass)."""
    return _register(_COLLATORS, model_type)


def register_sampler_builder(model_type: str):
    """Decorator to register a model-specific sampler builder."""
    return _register(_SAMPLER_BUILDERS, model_type)


def _ensure_imported(model_type: str) -> None:
    """Import the package that registers builders for ``model_type``."""
    if model_type in _IMPORTED or model_type not in MODEL_MODULES:
        return
    importlib.import_module(f"loongforge.data.embodied.{MODEL_MODULES[model_type]}")
    _IMPORTED.add(model_type)


def get_transform_builder(model_type: str) -> TransformBuilder:
    """Look up the per-sample transform builder for ``model_type``."""
    _ensure_imported(model_type)
    if model_type not in _TRANSFORM_BUILDERS:
        raise ValueError(
            f"Unknown transform builder for model_type '{model_type}'. "
            f"Known model types: {sorted(MODEL_MODULES)}"
        )
    return _TRANSFORM_BUILDERS[model_type]


def get_collator(model_type: str) -> Type:
    """Look up the collator class for ``model_type``."""
    _ensure_imported(model_type)
    if model_type not in _COLLATORS:
        raise ValueError(
            f"Unknown collator for model_type '{model_type}'. "
            f"Known model types: {sorted(MODEL_MODULES)}"
        )
    return _COLLATORS[model_type]


def get_sampler_builder(model_type: str) -> Optional[Callable]:
    """Return the sampler builder for ``model_type``, or None to use the default."""
    _ensure_imported(model_type)
    return _SAMPLER_BUILDERS.get(model_type)


def build_collator(
    model_type: str,
    model_cfg,
    data_cfg,
    training_args=None,
    dataset_stats=None,
    dataset=None,
    runtime: Optional[DataRuntime] = None,
):
    """Instantiate the registered collator via its ``from_config`` classmethod."""
    return get_collator(model_type).from_config(
        model_cfg,
        data_cfg,
        training_args=training_args,
        dataset_stats=dataset_stats,
        dataset=dataset,
        runtime=runtime,
    )


def build_transforms_from_args(
    model_cfg,
    data_cfg,
    training_args,
    dataset,
    dataset_stats,
    runtime: DataRuntime,
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
        runtime=runtime,
    )
    transforms = list(get_transform_builder(model_type)(ctx))
    if not transforms:
        return None
    return ComposedTransform(transforms)


def build_dataset_by_strategy(model_cfg, data_cfg, training_args, runtime):
    """Build a lerobot dataset via the strategy named by ``training_args.dataset_strategy``.

    Uses ``"default"`` when the field is absent or empty; raises ``ValueError``
    for a name that is not in ``DATASET_STRATEGIES``.
    """
    name = training_args.dataset_strategy or "default"
    if name not in DATASET_STRATEGIES:
        raise ValueError(
            f"Unknown dataset_strategy '{name}'. Supported: {sorted(DATASET_STRATEGIES)}"
        )
    entry = DATASET_STRATEGIES[name]
    module_rel, func_name = entry.split(":")
    mod = importlib.import_module(f"loongforge.data.embodied.{module_rel}")
    return getattr(mod, func_name)(model_cfg, data_cfg, training_args, runtime)


def build_dataset(model_cfg, data_cfg, training_args, runtime):
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
    return getattr(mod, func_name)(model_cfg, data_cfg, training_args, runtime)
