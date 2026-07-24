# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Public API for data transform pipelines and registries.

Per-sample transforms:
  - BaseTransform / ComposedTransform: Transform base classes
  - Normalizer: Multi-mode normalization (q99, min_max, mean_std, scale, binary)
  - ImageTransform: Image preprocessing (configurable resize strategy + normalize mode)
  - ActionTransform: Action chunking + normalization (configurable padding strategy)
  - StateDiscretizationTransform: State discretization for text-conditioned VLA models

Batch-level collators (DataLoader collate_fn):
  - BasePreprocessor / PreparedBatch: Base classes
  - Pi05Preprocessor / Pi05PreparedBatch: Pi0.5 collator
  - register_preprocessor: Registry decorator

Utilities:
  - tokenize_prompts
  - convert_stats: Convert dataset stats to numpy format
"""

from loongforge.embodied.data.datasets.transforms.base import BaseTransform, ComposedTransform
from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    PreparedBatch,
    build_preprocessor,
    get_preprocessor,
    register_preprocessor,
)
from loongforge.embodied.data.datasets.transforms.pipeline import build_transforms_from_args
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    get_transform_builder,
    register_transform_builder,
)

__all__ = [
    "BasePreprocessor",
    "BaseTransform",
    "ComposedTransform",
    "PreparedBatch",
    "TransformBuilderContext",
    "build_preprocessor",
    "build_transforms_from_args",
    "get_preprocessor",
    "get_transform_builder",
    "register_preprocessor",
    "register_transform_builder",
]
