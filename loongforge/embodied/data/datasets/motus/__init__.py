# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Motus multi-frame dataset strategy: behaviour hooks for ``LeRobotV3Dataset``."""

from loongforge.embodied.data.datasets.motus.index_samplers import (
    IndexSampleContext,
    get_index_sampler,
    register_index_sampler,
)
from loongforge.embodied.data.datasets.motus.motus_dataset import (
    build_motus_lerobot_dataset,
    motus_delta_timestamps,
    motus_index_map,
    motus_length,
)

# Import for side effect: registers ``@register_sampler_builder("motus")`` so the
# Motus DP sampler (base-matching DistributedSampler geometry, drop_last=False) is
# available as soon as this package is imported, mirroring dreamzero's in-__init__
# registration (datasets/dreamzero/__init__.py).
from loongforge.embodied.data.datasets.motus import samplers  # noqa: F401

__all__ = [
    "build_motus_lerobot_dataset",
    "motus_delta_timestamps",
    "motus_index_map",
    "motus_length",
    "IndexSampleContext",
    "get_index_sampler",
    "register_index_sampler",
]
