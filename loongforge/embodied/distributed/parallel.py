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
from .fsdp_utils import (
    FSDPWrapContext,
    build_fsdp_device_mesh,
    build_mp_policy,
    configure_prefetch,
    fully_shard_unit,
    resolve_wrap_runs,
)
from .fp8_utils import apply_fp8_linear_conversion
from .utils import filter_kwargs, is_mixed_param_dtype


logger = logging.getLogger(__name__)


def wrap_model(model: nn.Module, training_args, ctx: DistributedContext) -> nn.Module:
    """Wrap model with DDP or FSDP based on CLI training_args; mixed precision included."""
    # Imported here, not at module scope: train/__init__ pulls in trainers, which
    # import this module, so a top-level import would be circular.
    from loongforge.embodied.train.training_args import parse_dtype_from_str

    dtype = parse_dtype_from_str(training_args.dtype)
    # FP8 conversion must precede activation checkpointing: the checkpoint
    # wrapper has to wrap already-converted modules, otherwise the set of
    # tensors saved by the original forward and by recompute diverge.
    apply_fp8_linear_conversion(model, training_args, ctx.device)
    apply_activation_checkpointing(
        model,
        training_args.activation_checkpoint_module_patterns,
        training_args.activation_checkpoint_skip_modules,
        fp8_backend=training_args.fp8_backend if training_args.fp8 else None,
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

    ``--optimizer=dmuon`` bypasses all of the above: DMuon has to claim its
    Muon-route parameters before any group exists, so the model owns the wrap
    (see ``_wrap_dmuon_fsdp``).
    """
    if training_args.optimizer.lower() == "dmuon":
        return _wrap_dmuon_fsdp(model, training_args, ctx, dtype)

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


def _wrap_dmuon_fsdp(
    model: nn.Module,
    training_args,
    ctx: DistributedContext,
    dtype: torch.dtype,
) -> nn.Module:
    """Apply the model-owned FSDP2 + DMuon wrap path.

    DMuon requires the model to own its FSDP wrapping (``convert_to_fsdp``) so
    ``dmuon.dedicate_params()`` can claim the Muon-route parameters before any
    ``fully_shard`` group is created. The generic planner in ``_wrap_fsdp``
    cannot express that ordering, so this path hands mesh and mixed precision
    policy to the model and lets it build every group itself — wall-oss-0.5
    (``modeling_qwen2_5_vl_act.convert_to_fsdp``) is the reference implementation.

    Unlike ``_wrap_fsdp``, storage defaults to fp32 rather than ``--dtype``:
    DMuon keeps fp32 master weights for its dedicated params, so the whole model
    is cast to fp32 and bf16 compute comes from the mixed precision policy.
    """
    if not hasattr(model, "convert_to_fsdp"):
        raise NotImplementedError(
            f"Model {model.__class__.__name__} has no convert_to_fsdp method "
            f"required by --optimizer=dmuon."
        )

    # Local import to break the train/__init__ -> trainers -> this module cycle.
    from loongforge.embodied.train.training_args import parse_dtype_from_str

    storage_dtype = (
        parse_dtype_from_str(training_args.fsdp_original_param_dtype)
        if training_args.fsdp_original_param_dtype is not None
        else torch.float32
    )
    model.to(dtype=storage_dtype)

    # model=None: the cast above made the model uniform-dtype, so an unset
    # --fsdp-unshard-param-dtype must follow --dtype rather than "no cast".
    mp_policy = build_mp_policy(training_args, model=None)
    # FSDP2's own default; ``fsdp_reshard_default`` unset means "leave it alone".
    reshard_after_forward = (
        True if training_args.fsdp_reshard_default is None
        else training_args.fsdp_reshard_default
    )
    logger.info(
        "DMuon FSDP2: storage_dtype=%s training_dtype=%s reshard_after_forward=%s",
        storage_dtype, dtype, reshard_after_forward,
    )

    wrapped = model.convert_to_fsdp(
        mesh=build_fsdp_device_mesh(training_args, ctx),
        mp_policy=mp_policy,
        offload_policy=None,
        reshard_after_forward=reshard_after_forward,
        use_dmuon=True,
    )
    # The fp32 cast plus per-group sharding leaves the pre-shard full parameters
    # behind in the caching allocator; reclaim before the first forward.
    torch.cuda.empty_cache()
    return wrapped


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
