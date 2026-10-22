# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Wall-OSS-0.5 optimization factory: model-specific FSDP2 + DMuon wrapping.

DMuon claims its Muon-route parameters before any FSDP2 group exists, so the
generic wrap planner cannot express this model's layout. The engine keeps the
model-agnostic preparation and calls the adapter supplied here.
"""

from __future__ import annotations


def build_optimization(training_args):
    """Return a TorchOptimization that supplies the DMuon FSDP2 wrap."""
    from loongforge.engines.torch.distributed import wrap_model
    from loongforge.engines.torch.train_step import TorchOptimization
    from loongforge.extensions.torch.wall_oss_0_5.fsdp2_adapter import shard_wall_oss_dmuon

    def wrap(model, args, ctx):
        return wrap_model(model, args, ctx, dmuon_shard=shard_wall_oss_dmuon)

    return TorchOptimization(wrap_model=wrap)
