# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Gradient handling for optimizer-owned FSDP2 parameters.

FSDP2 leaves reduced sharded gradients on the DTensor parameters held by the
optimizer, while the currently materialized module parameters may carry no
``.grad``.  These helpers therefore start from ``optimizer.param_groups`` and
operate on each parameter's local shard.
"""

from __future__ import annotations

import torch
from collections import defaultdict
from torch.distributed.tensor import DTensor


_DTENSOR_CLIP_LOGGED = False


def ensure_fsdp_param_compat() -> None:
    """Patch FSDP2 grad accumulation for PyTorch builds with stale private access."""
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except Exception:
        return
    if getattr(FSDPParam, "_loongforge_accum_grad_compat", False):
        return

    def _to_accumulated_grad_if_needed(self):
        unsharded_param = getattr(self, "_unsharded_param", None)
        if (
            self.reduce_dtype is None
            or unsharded_param is None
            or unsharded_param.grad is None
            or unsharded_param.grad.dtype == self.reduce_dtype
        ):
            return
        unsharded_grad = unsharded_param.grad
        unsharded_param.grad = None
        self.unsharded_accumulated_grad = unsharded_grad.to(self.reduce_dtype)

    FSDPParam.to_accumulated_grad_if_needed = _to_accumulated_grad_if_needed
    FSDPParam._loongforge_accum_grad_compat = True


def optimizer_parameters(optimizer) -> list:
    """Return each trainable optimizer-owned parameter exactly once."""
    parameters = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in seen or not parameter.requires_grad:
                continue
            seen.add(id(parameter))
            parameters.append(parameter)
    return parameters


def local_gradient_groups(optimizer):
    """Collect mutable local DTensor gradients by device and dtype."""
    parameters = optimizer_parameters(optimizer)
    non_dtensor = [
        parameter for parameter in parameters if not isinstance(parameter, DTensor)
    ]
    if non_dtensor:
        raise RuntimeError(
            "Optimizer-owned gradient handling requires pure DTensor parameters; "
            f"found {len(non_dtensor)} non-DTensor parameters."
        )

    groups = defaultdict(list)
    gradient_count = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not isinstance(gradient, DTensor):
            raise RuntimeError(
                "Optimizer-owned gradient handling requires DTensor gradients; "
                f"got {type(gradient).__name__}."
            )
        local_gradient = gradient._local_tensor
        if local_gradient.is_sparse:
            raise RuntimeError("FSDP2 gradient handling does not support sparse gradients.")
        groups[(local_gradient.device, local_gradient.dtype)].append(local_gradient)
        gradient_count += 1
    return parameters, list(groups.values()), gradient_count


def local_norm_sq(gradient_groups, device):
    total_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
    for gradients in gradient_groups:
        norms = torch._foreach_norm(gradients, 2.0, dtype=torch.float32)
        total_norm_sq += torch.stack(norms).square().sum()
    return total_norm_sq


def clip_optimizer_gradients(optimizer, max_norm):
    """Clip dense DTensor gradients sharing one mesh and placement layout.

    FSDP and HSDP sum squared norms only over shard dimensions. Replicas
    already hold the same reduced gradient and must not be counted again.
    """
    global _DTENSOR_CLIP_LOGGED

    if max_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}.")
    parameters, gradient_groups, gradient_count = local_gradient_groups(optimizer)
    if not parameters:
        return 0.0
    mesh = parameters[0].device_mesh
    placements = parameters[0].placements
    if any(not (p.is_shard() or p.is_replicate()) for p in placements):
        raise RuntimeError("Gradient clipping requires reduced Shard/Replicate DTensors.")
    for parameter in parameters:
        for tensor in (parameter, parameter.grad):
            if tensor is not None and (tensor.device_mesh != mesh or tensor.placements != placements):
                raise RuntimeError("Gradient clipping requires one shared DTensor mesh and placement layout.")
    device = parameters[0]._local_tensor.device
    total_norm_sq = local_norm_sq(gradient_groups, device)
    for axis, placement in enumerate(placements):
        if placement.is_shard():
            torch.distributed.all_reduce(
                total_norm_sq, op=torch.distributed.ReduceOp.SUM, group=mesh.get_group(axis),
            )
    total_norm = total_norm_sq.sqrt()
    clip_coefficient = torch.clamp(
        torch.as_tensor(max_norm, device=device, dtype=torch.float32)
        / (total_norm + 1e-6),
        max=1.0,
    )
    for gradients in gradient_groups:
        torch._foreach_mul_(gradients, clip_coefficient)

    if not _DTENSOR_CLIP_LOGGED and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        print(
            "[fsdp2-dtensor-clip] "
            f"optimizer_params={len(parameters)} dtensor_params={len(parameters)} "
            f"gradients={gradient_count} global_grad_norm={total_norm.item():.10g} "
            f"clip_coefficient={clip_coefficient.item():.10g}",
            flush=True,
        )
        _DTENSOR_CLIP_LOGGED = True
    return total_norm.item()


def clean_optimizer_gradients(optimizer):
    """Replace NaN/Inf in optimizer-owned local DTensor gradients with zero."""
    _, gradient_groups, _ = local_gradient_groups(optimizer)
    for gradients in gradient_groups:
        for gradient in gradients:
            torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0, out=gradient)


def register_post_step_reshard(model, optimizer):
    """Reshard every FSDP group after the optimizer updates its local shard."""
    fsdp_modules = []
    seen = set()
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for module in chunk.modules():
            if id(module) in seen or not (
                hasattr(module, "unshard") and hasattr(module, "reshard")
            ):
                continue
            seen.add(id(module))
            fsdp_modules.append(module)

    if not hasattr(optimizer, "register_step_post_hook"):
        raise TypeError(
            "Optimizer must expose register_step_post_hook for post-step reshard"
        )

    logged = False

    def post_step_reshard(_optimizer, _args, _kwargs):
        nonlocal logged
        for module in fsdp_modules:
            module.reshard()
        if not logged and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            print(
                "[fsdp2-post-step-reshard] "
                f"active modules={len(fsdp_modules)}",
                flush=True,
            )
            logged = True

    optimizer_hook = optimizer.register_step_post_hook(post_step_reshard)
    return optimizer_hook, len(fsdp_modules)
