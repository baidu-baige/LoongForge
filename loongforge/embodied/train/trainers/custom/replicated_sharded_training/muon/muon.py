# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Muon optimizer for the replicated-sharded strategy and MoE expert weights.

``DistributedMuon`` keeps upstream ``torch.optim.Muon`` numerics for 2D weights
and adds batched Newton-Schulz for 3D MoE expert stacks (fused experts are stored
as a single ``[num_experts, in, out]`` tensor).

It operates on the replicated-sharded fp32 master parameters, which are plain
local tensors: compute replicas are never sharded, so Muon never sees a
``DTensor`` here. FSDP2-sharded Muon is a different code path entirely
(``--optimizer=dmuon``, see ``loongforge/embodied/optimizer/dmuon.py``), so this
optimizer rejects ``DTensor`` inputs instead of carrying a second, divergent
sharded implementation.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer


try:
    from torch.optim._muon import (
        DEFAULT_A,
        DEFAULT_B,
        DEFAULT_C,
        DEFAULT_NS_STEPS,
        EPS,
        _adjust_lr,
    )
except ImportError:  # pragma: no cover - torch < 2.9 fallback
    DEFAULT_A = 3.4445
    DEFAULT_B = -4.7750
    DEFAULT_C = 2.0315
    DEFAULT_NS_STEPS = 5
    EPS = 1e-7

    def _adjust_lr(  # type: ignore[no-redef]
        lr: float,
        adjust_lr_fn: Optional[str],
        param_shape: Sequence[int],
    ) -> float:
        """Torch 2.8 fallback for ``torch.optim._muon._adjust_lr``."""
        if adjust_lr_fn is None:
            return lr
        fan_out, fan_in = param_shape[:2]
        if adjust_lr_fn == "original":
            return lr * math.sqrt(max(1.0, fan_out / fan_in))
        if adjust_lr_fn == "match_rms_adamw":
            return lr * 0.2 * math.sqrt(max(fan_out, fan_in))
        raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")


__all__ = [
    "DistributedMuon",
    "split_muon_adamw_params",
]


DEFAULT_NS_COEFFICIENTS: Tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C)

# Newton-Schulz is numerically forgiving, so it runs in bf16 regardless of the
# master-parameter dtype; the result is cast back by the caller.
_NS_COMPUTE_DTYPE = torch.bfloat16


@torch.no_grad()
def batched_newton_schulz(
    grad: Tensor,
    ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
) -> Tensor:
    """Run quintic Newton-Schulz on each trailing ``[M, K]`` matrix.

    ``grad`` is either a single matrix or a ``[B, M, K]`` fused MoE expert stack
    (experts live on dim 0 and are never split). Nothing deeper than 3D can
    reach here, so there is no batch-dim flattening. ``ns_steps`` and
    ``ns_coefficients`` are validated once by ``DistributedMuon.__init__``.
    """
    if grad.ndim not in (2, 3):
        raise ValueError(f"Input must be 2D or 3D, got shape {tuple(grad.shape)}")

    a, b, c = ns_coefficients
    original_dtype = grad.dtype
    ortho = grad.to(_NS_COMPUTE_DTYPE)

    transposed = ortho.size(-2) > ortho.size(-1)
    if transposed:
        ortho = ortho.mT

    norm = ortho.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    ortho = ortho / norm

    if ortho.ndim == 3:
        # Allocate the three NS intermediates once and swap buffers, instead of
        # allocating a [B, M, M] gram pair on every iteration.
        batch, rows, _ = ortho.shape
        gram = torch.empty((batch, rows, rows), device=ortho.device, dtype=ortho.dtype)
        gram_update = torch.empty_like(gram)
        next_ortho = torch.empty_like(ortho)
        for _ in range(ns_steps):
            torch.bmm(ortho, ortho.mT, out=gram)
            torch.baddbmm(gram, gram, gram, beta=b, alpha=c, out=gram_update)
            torch.baddbmm(ortho, gram_update, ortho, beta=a, out=next_ortho)
            ortho, next_ortho = next_ortho, ortho
    else:
        for _ in range(ns_steps):
            gram = ortho @ ortho.mT
            gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            ortho = torch.addmm(ortho, gram_update, ortho, beta=a)

    if transposed:
        ortho = ortho.mT

    return ortho.to(original_dtype)


def split_muon_adamw_params(
    model,
    parameter_policy,
) -> Tuple[List[Tensor], List[Tensor], List[str], List[str]]:
    """Split trainable parameters into Muon-eligible weights and AdamW fallbacks.

    ``model`` only has to answer ``named_parameters()``; the routing decision is
    the policy's alone (``parameter_policy.optimizer_kind``), which already folds
    in the ndim test, the embedding/norm markers and
    ``muon_exclude_name_patterns``.
    """
    muon_params: List[Tensor] = []
    adamw_params: List[Tensor] = []
    muon_names: List[str] = []
    adamw_names: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if parameter_policy.optimizer_kind(name, param) == "muon":
            muon_params.append(param)
            muon_names.append(name)
        else:
            adamw_params.append(param)
            adamw_names.append(name)

    return muon_params, adamw_params, muon_names, adamw_names


class DistributedMuon(Optimizer):
    """Muon optimizer over the replicated-sharded fp32 master parameters.

    One Newton-Schulz pass per parameter; 3D MoE expert stacks are
    orthogonalized per trailing matrix in a single batched pass.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: Optional[str] = None,
    ) -> None:
        """Validate the hyperparameters and reject params Muon cannot handle."""
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= float(lr):
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= float(momentum):
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= float(weight_decay):
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")
        if not 1 <= ns_steps < 100:
            raise ValueError(f"ns_steps must be in [1, 100) but is: {ns_steps}")
        if len(ns_coefficients) != 3:
            raise ValueError("ns_coefficients must be a tuple of exactly 3 values")

        defaults: Dict[str, Any] = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        # Optional hook invoked with each parameter right after its update is
        # applied. ZeRO-1 uses it to start that parameter's replica sync while
        # the remaining Newton-Schulz work is still running.
        self.param_update_callback = None

        for group in self.param_groups:
            for p in group["params"]:
                if isinstance(p, DTensor):
                    raise ValueError(
                        "DistributedMuon operates on local tensors only; got a "
                        "DTensor. The replicated-sharded strategy keeps compute "
                        "replicas unsharded, so this indicates an FSDP2-sharded "
                        "model. Use --optimizer=dmuon for FSDP2."
                    )
                if p.ndim not in (2, 3):
                    raise ValueError(
                        "DistributedMuon supports only 2D and 3D parameters; "
                        f"got param with shape {tuple(p.size())}. Route 1D/4D+ "
                        "params (biases, norms, conv weights) to AdamW via "
                        "split_muon_adamw_params."
                    )
                if torch.is_complex(p):
                    raise ValueError("DistributedMuon does not support complex parameters")

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Run one Muon step: momentum, Newton-Schulz, then the scaled update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_coefficients = tuple(group["ns_coefficients"])
            ns_steps = int(group["ns_steps"])
            eps = float(group["eps"])
            adjust_lr_fn = group["adjust_lr_fn"]
            callback = self.param_update_callback

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("DistributedMuon does not support sparse gradients")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                buf = state["momentum_buffer"]

                # The upstream Muon momentum step:
                # ``buf <- lerp(buf, grad, 1 - momentum)``, then the update is
                # ``lerp(grad, buf, momentum)`` under Nesterov, else ``buf``.
                buf.lerp_(p.grad, 1 - momentum)
                update = p.grad.lerp(buf, momentum) if nesterov else buf.clone()

                ortho = batched_newton_schulz(update, ns_coefficients, ns_steps, eps)
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, p.shape[-2:])

                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(ortho.to(dtype=p.dtype), alpha=-adjusted_lr)

                if callback is not None:
                    callback(p)

        return loss
