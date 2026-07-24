# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.

from loongforge.embodied.data.datasets.transforms.registry import (
    register_transform_builder,
)

from .dreamzero_collator import (
    DefaultDataCollator,
    DreamTransform,
    DreamZeroPreparedBatch,
)


@register_transform_builder("dreamzero")
def build_dreamzero_transforms(_context):
    """Keep DreamZero's deterministic modality transforms dataset-owned."""
    return []


__all__ = [
    "DefaultDataCollator",
    "DreamTransform",
    "DreamZeroPreparedBatch",
    "build_dreamzero_transforms",
]
