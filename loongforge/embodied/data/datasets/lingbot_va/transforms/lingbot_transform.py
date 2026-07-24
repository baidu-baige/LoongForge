# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA per-sample transform builder."""

from __future__ import annotations

from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)


@register_transform_builder("lingbot_va")
def build_lingbot_va_transforms(ctx: TransformBuilderContext):
    """Return no per-sample transforms for LingBot latent datasets.

    The LingBot latent dataset already emits model-ready sample keys; batching is
    owned by ``LingBotVAPreprocessor``.
    """
    return []
