# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLA transforms pipeline builder.

Pipeline builder:
    - build_transforms_from_args: Build per-sample transforms from config + CLI args
"""

from typing import Optional

from loongforge.embodied.data.datasets.transforms.base import ComposedTransform
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    get_transform_builder,
)


# ═══════════════════════════════════════════════════════════════
# Pipeline Builder
# ═══════════════════════════════════════════════════════════════

def build_transforms_from_args(
    model_cfg,
    data_cfg,
    training_args,
    dataset,
    dataset_stats,
) -> Optional[ComposedTransform]:
    """Build per-sample transforms from typed ModelConfig + DataConfig (+ TrainingArgs).

    Args:
        model_cfg: typed ModelConfig (model structure + shared fields).
        data_cfg: typed DataConfig (data-processing fields).
        training_args: TrainingArgs (generic CLI params).
        dataset: The dataset instance (used to discover image keys).
        dataset_stats: Dict of normalization statistics from dataset.meta.stats.

    Returns:
        ComposedTransform or None if not applicable.
    """
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
