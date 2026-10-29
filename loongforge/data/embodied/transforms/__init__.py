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
  - BaseCollator / PreparedBatch: Base classes
  - Pi05Collator / Pi05PreparedBatch: Pi0.5 collator
  - register_collator: Registry decorator

Utilities:
  - tokenize_prompts
  - convert_stats: Convert dataset stats to numpy format
"""

from loongforge.data.embodied.transforms.base import BaseTransform, ComposedTransform
from loongforge.data.embodied.transforms.collator import (
    BaseCollator,
    PreparedBatch,
)
from loongforge.data.embodied.registry import (
    TransformBuilderContext,
    build_collator,
    build_transforms_from_args,
    get_collator,
    get_transform_builder,
    register_collator,
    register_transform_builder,
)

__all__ = [
    "BaseCollator",
    "BaseTransform",
    "ComposedTransform",
    "PreparedBatch",
    "TransformBuilderContext",
    "build_collator",
    "build_transforms_from_args",
    "get_collator",
    "get_transform_builder",
    "register_collator",
    "register_transform_builder",
]
