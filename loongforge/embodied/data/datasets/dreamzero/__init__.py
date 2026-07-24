# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.
# Copyright (c) Baidu Inc. All rights reserved.
"""DreamZero data pipeline.

Runtime imports stay inside this package so training does not depend on an
external DreamZero source checkout.
"""

from loongforge.embodied.data.datasets.sampler_builder import (
    SamplerBuilderContext,
    register_sampler_builder,
)


@register_sampler_builder("dreamzero")
def build_dreamzero_registered_sampler(context: SamplerBuilderContext):
    """Adapt DreamZero sampler modes to the framework sampler registry."""
    data_cfg = context.dataset._dreamzero_sampler_cfg

    from .builder import build_dreamzero_sampler

    base_dataset = context.dataset._dreamzero_base_dataset
    sampler_seed = (
        context.seed
        if data_cfg.sampler_seed is None
        else int(data_cfg.sampler_seed)
    )
    sampler, _ = build_dreamzero_sampler(
        train_dataset=context.dataset,
        base_dataset=base_dataset,
        sampler_type=data_cfg.sampler_type,
        ctx=context.ctx,
        batch_size=context.batch_size,
        num_workers=context.training_args.num_workers,
        sampler_seed=sampler_seed,
        data_cfg=data_cfg,
    )
    return sampler
