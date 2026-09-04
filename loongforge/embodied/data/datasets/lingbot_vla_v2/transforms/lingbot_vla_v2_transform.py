# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Sample-level transforms for lingbot_vla_v2.

The vendored MultiVLADataset already performs all sample-level processing
(image resize/normalise, action/state normalisation, prompt tokenisation)
inside ``__getitem__`` — identical to the upstream benchmark. An empty
transform list keeps the LoongForge pipeline contract satisfied without
duplicating any processing.
"""

from loongforge.embodied.data.datasets.transforms.registry import (
    register_transform_builder,
)


@register_transform_builder("lingbot_vla_v2")
def build_lingbot_vla_v2_transforms(ctx):
    """No-op: all sample processing lives inside the vendored dataset."""
    return []
