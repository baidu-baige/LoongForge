# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Wall-OSS-0.5 trainer hooks: model-specific FSDP2 + DMuon wrapping.

DMuon claims its Muon-route parameters before any FSDP2 group exists, so the
generic wrap planner cannot express this model's layout. The engine keeps the
model-agnostic preparation and calls the adapter supplied here.
"""

from __future__ import annotations


def build_hooks(training_args):
    """Return the TrainerHooks that supply the DMuon FSDP2 wrap."""
    from loongforge.engines.torch.distributed import wrap_model
    from loongforge.engines.torch.runner import TrainerHooks
    from loongforge.training.embodied.custom.wall_oss_0_5.wall_oss_fsdp2_adapter import shard_wall_oss_dmuon

    def wrap(model, args, ctx):
        return wrap_model(model, args, ctx, dmuon_shard=shard_wall_oss_dmuon)

    return TrainerHooks(wrap_model=wrap)
