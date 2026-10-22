# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Gradient clipping and NaN cleaning."""

import torch
import torch.distributed.fsdp as FSDP1
import torch.nn as nn
from torch.distributed.tensor import DTensor


def get_grad_norm(model: nn.Module) -> float:
    """Return the total gradient L2 norm.

    FSDP1 delegates to ``FSDP.clip_grad_norm_`` with an infinite threshold so
    that FSDP aggregates parameter shards over its process group. Although this
    does not clip finite gradients, the PyTorch API may still perform an
    in-place multiplication by the clamped coefficient.

    The non-FSDP1 branch handles FSDP2 and DDP models with ``get_total_norm``.
    FSDP2 gradients remain DTensors, allowing the operation to dispatch the
    required collectives from their device mesh and placements; DDP gradients
    are already synchronized and can be treated as regular tensors. Unwrapped
    models use the same path. A DTensor result is materialized as a replicated
    scalar before conversion to ``float``.
    """
    if isinstance(model, FSDP1.FullyShardedDataParallel):
        total_norm = model.clip_grad_norm_(max_norm=float("inf"), norm_type=2.0)
        return float(total_norm)

    else:
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return 0.0

        total_norm = torch.nn.utils.get_total_norm(gradients, norm_type=2.0)
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()
        return float(total_norm)


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip gradients in place and return their pre-clipping total L2 norm.

    FSDP1 uses ``FSDP.clip_grad_norm_`` so local parameter shards are aggregated
    over the FSDP process group before one global clipping coefficient is
    applied. The non-FSDP1 branch handles FSDP2 and DDP models with
    ``torch.nn.utils.clip_grad_norm_``. FSDP2 derives the required collectives
    from each gradient's DTensor mesh and placements, while DDP gradients are
    already synchronized before clipping. Unwrapped models use the same path.
    """

    if isinstance(model, FSDP1.FullyShardedDataParallel):
        total_norm = model.clip_grad_norm_(max_norm)
        return float(total_norm)
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return float(total_norm)


def clean_nan_gradients(model: nn.Module):
    """Replace NaN/Inf gradients with 0.

    Args:
        model: Model after backward. Gradients are rewritten in place (``out=``),
            so DTensor gradients are cleaned through their local shard and no
            tensor is reallocated.

    Note:
        Returns ``None`` and reports nothing: there is no signal about how much was
        replaced, which makes this a silent mask over divergence. A run that needs
        this every step is broken somewhere upstream (learning rate, loss scaling,
        bad batch) and zeroing gradients only postpones the diagnosis.

        Both ``+inf`` and ``-inf`` become 0.0 rather than a large finite value —
        the intent is to drop the offending contribution, not to clip it.

        Purely local, no collectives, so ranks may clean different amounts. That is
        consistent for sharded gradients (each rank owns a distinct shard), but it
        means a replicated gradient could in principle be cleaned on one rank only;
        DDP has already all-reduced by this point, so in practice the ranks see the
        same values.

        Order matters: cleaning before ``clip_gradients`` keeps a single NaN from
        poisoning the whole model through the shared clip coefficient, while
        cleaning after leaves the coefficient already NaN and the gradients all
        zero.
    """
    for param in model.parameters():
        if param.grad is not None:
            grad = (
                param.grad.to_local()
                if isinstance(param.grad, DTensor)
                else param.grad
            )
            torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0, out=grad)