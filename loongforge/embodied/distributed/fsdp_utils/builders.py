# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Builders for the FSDP2 arguments that are fixed for a whole wrap pass."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from ..context import DistributedContext
from ..utils import is_mixed_param_dtype, is_rank_zero

logger = logging.getLogger(__name__)


def build_mp_policy(training_args, model: nn.Module | None = None) -> MixedPrecisionPolicy:
    """Build the FSDP2 MixedPrecisionPolicy from training args.

    Built once per wrap pass and reused for every group, so all groups agree on
    precision — a per-group policy would make gradient reductions run at
    different dtypes in the same step.

    Args:
        training_args: Supplies ``fsdp_unshard_param_dtype``,
            ``fsdp_reduce_dtype``, ``fsdp_output_dtype`` as dtype strings and
            ``fsdp_cast_forward_inputs`` as a bool. ``fsdp_original_param_dtype``
            and ``dtype`` are read only to resolve the all-gather default below.
        model: Optional model used to detect authored mixed parameter dtypes.
            When omitted, mixed-dtype detection is skipped and an unset
            ``--fsdp-unshard-param-dtype`` follows the non-mixed default.

    Returns:
        Policy consumed by every ``fully_shard`` call in this pass.

    Note:
        An unset ``--fsdp-unshard-param-dtype`` means "do not extra-cast" only
        for models authored with mixed parameter dtypes. Uniform-dtype models
        fall back to ``--dtype``, matching the pre-refactor default so
        all-gather / compute stay on the training dtype (typically bf16) and
        ``cast_forward_inputs`` still has a target.

        The other fallback is when ``--fsdp-original-param-dtype`` decouples
        storage from ``--dtype`` (fp32 master weights under bf16 compute):
        "unset" then also resolves to ``--dtype``. Otherwise the all-gather
        would hand fp32 parameters to the compute path and the run would
        silently lose the bf16 compute it asked for, at double the activation
        bandwidth.

        Sharded parameter storage is not set here — it is the model's own
        dtype at wrap time. Reductions fall back to FSDP2's own default when
        ``fsdp_reduce_dtype`` is unset; forward outputs keep the compute dtype
        unless ``fsdp_output_dtype`` is set.

        ``cast_forward_inputs`` only casts the inputs FSDP sees at a group
        boundary, and only when ``param_dtype`` is not ``None``. Models whose
        ``forward`` builds tensors internally (masks, position ids, rotary
        caches) still need to get those dtypes right themselves; leaving this
        ``False`` while feeding fp32 batches into a bf16 policy is the usual
        source of dtype-mismatch errors deep inside a layer.
    """
    # Local import to break the train/__init__ -> trainers -> this module cycle.
    from loongforge.embodied.train.training_args import parse_dtype_from_str

    authored_mixed_dtype = (
        is_mixed_param_dtype(model, trainable_only=False) if model is not None else False
    )
    if training_args.fsdp_unshard_param_dtype is not None:
        unshard_param_dtype = parse_dtype_from_str(training_args.fsdp_unshard_param_dtype)
    elif authored_mixed_dtype:
        # Preserve mixed authored dtypes: an extra all-gather cast would force
        # every group onto one compute dtype.
        unshard_param_dtype = None
    else:
        # Uniform-dtype model, or storage explicitly decoupled from --dtype:
        # "unset" follows --dtype rather than "no cast".
        unshard_param_dtype = parse_dtype_from_str(training_args.dtype)
    reduce_dtype = (
        parse_dtype_from_str(training_args.fsdp_reduce_dtype)
        if training_args.fsdp_reduce_dtype is not None else None
    )
    output_dtype = (
        parse_dtype_from_str(training_args.fsdp_output_dtype)
        if training_args.fsdp_output_dtype is not None else None
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=unshard_param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        cast_forward_inputs=training_args.fsdp_cast_forward_inputs,
    )
    logger.info(
        "FSDP2 runtime: unshard_param_dtype=%s reduce_dtype=%s output_dtype=%s cast_forward_inputs=%s",
        unshard_param_dtype, reduce_dtype, output_dtype, training_args.fsdp_cast_forward_inputs,
    )
    return mp_policy


def build_ignored_params(model: nn.Module, ctx: DistributedContext) -> set:
    """Collect parameters that FSDP must not manage.

    FSDP2 shards parameters on dim-0, so 0-dim (scalar) parameters have nothing
    to shard: ``fully_shard`` rejects them outright and they are handed over as
    ``ignored_params`` instead.

    Being ignored means FSDP never touches them, gradients included: nothing
    reduces their gradients across ranks, so a *trainable* scalar parameter would
    silently take a different optimizer step on every rank. Training one is
    therefore rejected rather than quietly diverging.

    Args:
        model: Model to scan, before wrapping. Scanned with
            ``named_parameters()``, so tied parameters are collected once by
            identity.
        ctx: Provides the compute device the ignored parameters are moved to.

    Returns:
        Set of parameter objects to pass as ``fully_shard(ignored_params=...)``.
        Empty when the model has no 0-dim parameters. Identity-based, so callers
        can test membership while walking the module tree
        (see ``group_numel_by_dtype``).

    Raises:
        ValueError: If any 0-dim parameter has ``requires_grad=True``.

    Note:
        Side effect: moves the ignored parameters onto ``ctx.device`` in place
        (``p.data = p.to(...)``), because ``fully_shard`` only relocates the
        parameters it manages and leaving these on CPU fails at the first
        forward. External references to those parameter objects stay valid — only
        their storage is replaced — but anything caching ``p.data`` or a
        ``data_ptr`` from before this call is stale afterwards.

        Every rank must call this before wrapping; the returned set is used as an
        identity filter throughout unit selection, so a rank that skipped it
        would size its units differently.
    """
    scalar_params = [(n, p) for n, p in model.named_parameters() if p.ndim == 0]
    trainable = [n for n, p in scalar_params if p.requires_grad]
    if trainable:
        raise ValueError(
            f"FSDP cannot train 0-dim (scalar) parameters: "
            f"{', '.join(trainable[:8])}. They cannot be sharded on dim-0, so "
            f"FSDP ignores them and their gradients are never reduced across "
            f"ranks. Give them shape (1,) so FSDP can manage them, or freeze "
            f"them with requires_grad=False."
        )
    ignored = {p for _, p in scalar_params}

    # fully_shard does not move ignored params to the device, so do it here.
    with torch.no_grad():
        for p in ignored:
            if p.device != ctx.device:
                p.data = p.to(device=ctx.device)

    if scalar_params:
        logger.info(
            "FSDP2 ignores %d frozen scalar params: %s",
            len(scalar_params), ", ".join(n for n, _ in scalar_params[:8]),
        )
    return ignored


def build_fsdp_device_mesh(training_args, ctx: DistributedContext):
    """Build FSDP (1D) or HSDP (2D) device mesh.

    Args:
        training_args: ``hsdp_shard_size`` selects the topology. ``None`` means
            plain FSDP: one 1D mesh over all ranks, i.e. every parameter is
            sharded across the whole world. An integer switches to HSDP, sharding
            within groups of that size and replicating across groups — the point
            being to keep all-gathers inside a node while the cheaper gradient
            all-reduce crosses nodes, so the value is normally the number of GPUs
            per node.
        ctx: Provides ``world_size``; the mesh is always built on the ``"cuda"``
            device type, so this path assumes a CUDA build and an initialized
            process group.

    Returns:
        ``DeviceMesh`` with dim names ``("fsdp",)`` for 1D or
        ``("replica", "shard")`` for 2D. The names are part of the contract —
        anything constructing a submesh (e.g. tensor-parallel or checkpoint code)
        looks them up by name, so they differ between the two cases.

    Raises:
        ValueError: If ``world_size`` is not divisible by ``hsdp_shard_size``;
            an uneven mesh would give ranks different shard counts.

    Note:
        ``init_device_mesh`` is collective — all ranks must reach it with the same
        shape, and the rank-to-coordinate mapping is row-major, so shard groups
        are contiguous rank ranges. A shard size equal to ``world_size`` produces
        a single replica group and only warns, since it is degenerate but not
        wrong: it behaves like plain FSDP with a 2D mesh.
    """
    shard_size = training_args.hsdp_shard_size
    if shard_size is None:
        return init_device_mesh("cuda", (ctx.world_size,), mesh_dim_names=("fsdp",))

    if ctx.world_size % shard_size != 0:
        raise ValueError(
            f"HSDP requires world_size divisible by hsdp_shard_size, "
            f"got {ctx.world_size} % {shard_size} != 0."
        )

    replica_size = ctx.world_size // shard_size
    if replica_size == 1:
        logger.warning(
            "--hsdp-shard-size with one replica group is equivalent to plain FSDP."
        )
    if is_rank_zero():
        logger.info("Using HSDP 2D mesh: replica=%d, shard=%d.", replica_size, shard_size)

    return init_device_mesh(
        "cuda", (replica_size, shard_size), mesh_dim_names=("replica", "shard"),
    )
