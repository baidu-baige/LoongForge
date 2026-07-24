# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA dataset package."""

__all__ = [
    "LingBotVALatentDataset",
    "LingBotVALatentDatasetConfig",
    "LingBotVAMultiLatentDataset",
    "_LingBotBalancedDistributedSampler",
    "build_lingbot_dataset",
    "build_lingbot_va_distributed_sampler",
    "LingBotVADataConfig",
    "LingBotVAPreparedBatch",
    "LingBotVAPreprocessor",
    "build_lingbot_va_transforms",
]


def __getattr__(name):
    if name in {
        "LingBotVALatentDataset",
        "LingBotVALatentDatasetConfig",
        "LingBotVAMultiLatentDataset",
        "build_lingbot_dataset",
    }:
        from loongforge.embodied.data.datasets.lingbot_va import (
            latent_lerobot_dataset as dataset_mod,
        )

        return getattr(dataset_mod, name)
    if name in {
        "_LingBotBalancedDistributedSampler",
        "build_lingbot_va_distributed_sampler",
    }:
        from loongforge.embodied.data.datasets.lingbot_va import samplers as sampler_mod

        return getattr(sampler_mod, name)
    if name in {
        "LingBotVADataConfig",
        "LingBotVAPreparedBatch",
        "LingBotVAPreprocessor",
        "build_lingbot_va_transforms",
    }:
        from loongforge.embodied.data.datasets.lingbot_va import (
            transforms as transforms_mod,
        )

        return getattr(transforms_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
