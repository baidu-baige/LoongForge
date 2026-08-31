# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DDP/FSDP wrapping with mixed precision managed by the parallel strategy."""

from __future__ import annotations

import logging

import torch
import torch._dynamo
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .activation_checkpointing import apply_activation_checkpointing
from .context import DistributedContext

from .ddp_utils import resolve_comm_hook
from .fsdp_utils import FSDPWrapContext, configure_prefetch, fully_shard_unit, resolve_wrap_runs
from .utils import filter_kwargs, is_mixed_param_dtype


logger = logging.getLogger(__name__)


def wrap_model(model: nn.Module, training_args, ctx: DistributedContext) -> nn.Module:
    """Wrap model with DDP or FSDP based on CLI training_args; mixed precision included."""
    # Imported here, not at module scope: train/__init__ pulls in trainers, which
    # import this module, so a top-level import would be circular.
    from loongforge.embodied.train.training_args import parse_dtype_from_str

    dtype = parse_dtype_from_str(training_args.dtype)
    apply_activation_checkpointing(
        model,
        training_args.activation_checkpoint_module_patterns,
        training_args.activation_checkpoint_skip_modules,
    )

    if not ctx.is_distributed:
        if is_mixed_param_dtype(model, trainable_only=False):
            return model.to(device=ctx.device)
        return model.to(dtype=dtype, device=ctx.device)

    strategy = training_args.distributed_strategy
    if strategy == "fsdp":
        return _wrap_fsdp(model, training_args, ctx, dtype)
    else:
        return _wrap_ddp(model, training_args, ctx, dtype)


def _wrap_fsdp(
    model: nn.Module,
    training_args,
    ctx: DistributedContext,
    dtype: torch.dtype,
) -> nn.Module:
    """Apply FSDP2 wrapping to the model.

    FSDP units are created bottom-up, then the root is wrapped as a catch-all
    group for every parameter not owned by an inner unit (FSDP2 automatically
    skips parameters already assigned to a nested group).

    Unit selection priority:
    1. ``--fsdp-wrap-modules`` explicitly names the inner FSDP units, one unit
       per matched module.
    2. Otherwise consecutive same-class children of ordered containers are
       stacked into one unit until that unit reaches ``--fsdp-min-param-num``,
       plus any ``model._no_split_modules`` class (HuggingFace convention) that
       lives outside such a container.

    ``--fsdp-no-wrap-modules`` classes are never used as units; their
    parameters are still sharded by the closest enclosing group.

    Each step lives in ``fsdp_utils``: unit selection in ``units``, group
    creation in ``sharding``, prefetch edges in ``prefetch``.
    """
    # Storage dtype is decoupled from ``--dtype``: ``--dtype`` also drives the
    # autocast dtype in the trainer, so keeping fp32 master weights under bf16
    # compute requires a separate knob. ``build_mp_policy`` mirrors this by
    # defaulting the all-gather dtype to ``--dtype`` for uniform-dtype models.
    storage_dtype = dtype
    if training_args.fsdp_original_param_dtype is not None:
        from loongforge.embodied.train.training_args import parse_dtype_from_str

        storage_dtype = parse_dtype_from_str(training_args.fsdp_original_param_dtype)

    if not is_mixed_param_dtype(model, trainable_only=False):
        model.to(dtype=storage_dtype)

    wrap_ctx = FSDPWrapContext.create(model, training_args, ctx)

    runs = resolve_wrap_runs(model, training_args, wrap_ctx.ignored_params)
    # Deepest first so inner units claim their parameters before outer ones.
    for run in sorted(runs, key=lambda run: run.depth, reverse=True):
        for unit in run.units:
            fully_shard_unit(unit, wrap_ctx)

    # Root goes last, as the catch-all group described above. FSDP2 always keeps
    # the root's parameters unsharded after forward, whatever reshard policy the
    # group resolves to.
    fully_shard_unit([model], wrap_ctx)
    configure_prefetch(runs, training_args)
    return model


def _wrap_ddp(model: nn.Module, training_args, ctx: DistributedContext, dtype: torch.dtype) -> nn.Module:
    """Wrap model with DistributedDataParallel."""
    if is_mixed_param_dtype(model, trainable_only=False):
        model = model.to(device=ctx.device)
    else:
        model = model.to(dtype=dtype, device=ctx.device)

    torch._dynamo.config.optimize_ddp = training_args.dynamo_optimize_ddp

    ddp_kwargs = {
        "broadcast_buffers": training_args.ddp_broadcast_buffers,
        "init_sync": training_args.ddp_init_sync,
        "bucket_cap_mb": training_args.ddp_bucket_cap_mb,
        "find_unused_parameters": training_args.ddp_find_unused_parameters,
        "gradient_as_bucket_view": training_args.ddp_gradient_as_bucket_view,
        "static_graph": training_args.ddp_static_graph,
        "skip_all_reduce_unused_params": training_args.ddp_skip_all_reduce_unused_params,
        "bucket_cap_mb_list": training_args.ddp_bucket_cap_mb_list,
        "batched_grad_copy": training_args.ddp_batched_grad_copy,
    }

    ddp_model = DDP(model, **filter_kwargs(DDP, ddp_kwargs))
    if training_args.ddp_comm_hook:
        comm_hook = resolve_comm_hook(
            training_args.ddp_comm_hook,
            use_logging=training_args.ddp_comm_hook_logging,
        )
        ddp_model.register_comm_hook(state=None, hook=comm_hook)
    return ddp_model
