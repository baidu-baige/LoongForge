"""Vendor module implementation."""
"""DTensor-aware Muon optimizer for FSDP2 and MoE expert weights.

``DistributedMuon`` keeps upstream ``torch.optim.Muon`` numerics for 2D
weights and adds batched Newton-Schulz for 3D MoE expert stacks.

For FSDP2-sharded 2D params, same-shape parameters are mega-batched:
stacked into a single tensor, gathered with one NCCL call, orthogonalized
with one batched NS pass, and scattered back locally.
"""

from collections import defaultdict
import os
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
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

    _MUON_AVAILABLE = True
except ImportError:  # pragma: no cover - torch < 2.9 fallback
    _MUON_AVAILABLE = False
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
    "DEFAULT_NS_COEFFICIENTS",
    "DEFAULT_NS_STEPS",
    "DistributedMuon",
    "batched_newton_schulz",
    "split_muon_adamw_params",
]


DEFAULT_NS_COEFFICIENTS: Tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C)

_DEFAULT_ADAMW_NAME_PATTERNS: Tuple[str, ...] = (
    "embed_tokens",
    "embedding",
    "lm_head",
    "output_layer",
)

_MEGABATCH_MAX_GROUP_SIZE = 32
_LOCAL_MEGABATCH_DEFAULT_GROUP_SIZE = 8


@torch.no_grad()
def batched_newton_schulz(
    grad: Tensor,
    ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Run quintic Newton-Schulz on each trailing ``[M, K]`` matrix."""
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if grad.ndim < 2:
        raise ValueError(f"Input must have ndim >= 2, got shape {tuple(grad.shape)}")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")

    a, b, c = ns_coefficients
    original_dtype = grad.dtype
    ortho = grad.to(compute_dtype)

    transposed = ortho.size(-2) > ortho.size(-1)
    if transposed:
        ortho = ortho.mT

    norm = ortho.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    ortho = ortho / norm

    # Reuse NS intermediates by default; set LOONGFORGE_MUON_REUSE_NS=0 to fall back.
    if os.environ.get("LOONGFORGE_MUON_REUSE_NS", "1") == "1" and ortho.ndim >= 3:
        *batch, M_, K_ = ortho.shape
        B_ = 1
        for d in batch:
            B_ *= d
        ortho_3d = ortho.reshape(B_, M_, K_)
        A_3d = torch.empty((B_, M_, M_), device=ortho.device, dtype=ortho.dtype)
        gram_update = torch.empty_like(A_3d)
        next_ortho = torch.empty_like(ortho_3d)
        for _ in range(ns_steps):
            torch.bmm(ortho_3d, ortho_3d.mT, out=A_3d)
            torch.baddbmm(A_3d, A_3d, A_3d, beta=b, alpha=c, out=gram_update)
            torch.baddbmm(ortho_3d, gram_update, ortho_3d, beta=a, out=next_ortho)
            ortho_3d, next_ortho = next_ortho, ortho_3d
        ortho = ortho_3d.reshape(*batch, M_, K_)
    else:
        for _ in range(ns_steps):
            A = ortho @ ortho.mT
            if A.ndim == 2:
                gram_update = torch.addmm(A, A, A, beta=b, alpha=c)
                ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
            else:
                *batch, M_, K_ = ortho.shape
                B_ = 1
                for d in batch:
                    B_ *= d
                A_3d = A.reshape(B_, M_, M_)
                ortho_3d = ortho.reshape(B_, M_, K_)
                gram_update = torch.baddbmm(A_3d, A_3d, A_3d, beta=b, alpha=c)
                ortho = torch.baddbmm(ortho_3d, gram_update, ortho_3d, beta=a).reshape(*batch, M_, K_)
            del A

    if transposed:
        ortho = ortho.mT

    return ortho.to(original_dtype)


def _is_adamw_by_name(name: str, extra_patterns: Sequence[str]) -> bool:
    """Vendor interface."""
    lname = name.lower()
    for pat in _DEFAULT_ADAMW_NAME_PATTERNS:
        if pat in lname:
            return True
    for pat in extra_patterns:
        if pat and pat.lower() in lname:
            return True
    return False


def _is_muon_eligible_ndim(param: Tensor) -> bool:
    """Return True for dense linears and 3D MoE expert stacks."""
    return param.ndim in (2, 3)


def split_muon_adamw_params(
    model: "nn.Module",
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
    extra_adamw_name_patterns: Optional[Sequence[str]] = None,
    parameter_policy=None,
) -> Tuple[List[Tensor], List[Tensor], List[str], List[str]]:
    """Split model parameters into Muon-eligible weights and AdamW fallback weights."""
    no_decay_modules = no_decay_modules or []
    no_decay_params = no_decay_params or []
    extra_patterns = list(extra_adamw_name_patterns or ())

    forced_adamw_fqns: set = set()
    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        is_embedding = isinstance(module, nn.Embedding)
        is_no_decay = cls_name in no_decay_modules
        if is_embedding or is_no_decay:
            for pname, _p in module.named_parameters(recurse=False):
                fqn = f"{module_name}.{pname}" if module_name else pname
                forced_adamw_fqns.add(fqn)

    muon_params: List[Tensor] = []
    adamw_params: List[Tensor] = []
    muon_names: List[str] = []
    adamw_names: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if parameter_policy is not None:
            forced_adamw = parameter_policy.optimizer_kind(name, param) != "muon"
        else:
            muon_ok = _is_muon_eligible_ndim(param)
            forced_adamw = (
                (not muon_ok)
                or name in forced_adamw_fqns
                or _is_adamw_by_name(name, extra_patterns)
                or any(p and p.lower() in name.lower() for p in no_decay_params)
            )
        if forced_adamw:
            adamw_params.append(param)
            adamw_names.append(name)
        else:
            muon_params.append(param)
            muon_names.append(name)

    return muon_params, adamw_params, muon_names, adamw_names


_KIND_LOCAL = "local"
_KIND_FSDP_GATHER_2D = "fsdp_gather_2d"
_KIND_MOE_LOCAL_3D = "moe_local_3d"
_KIND_MOE_GATHER_3D = "moe_gather_3d"


def _shard_dims(p: DTensor) -> List[int]:
    """Return the list of tensor dims along which ``p`` is sharded."""
    return [pl.dim for pl in p.placements if isinstance(pl, Shard)]


def _classify_param(p: Tensor) -> str:
    """Return one of ``_KIND_*`` describing how Muon should treat ``p``."""
    if not isinstance(p, DTensor):
        return _KIND_LOCAL

    shard_dims = _shard_dims(p)
    if not shard_dims:
        return _KIND_LOCAL

    if p.ndim == 2:
        return _KIND_FSDP_GATHER_2D

    if p.ndim == 3:
        if all(d == 0 for d in shard_dims):
            return _KIND_MOE_LOCAL_3D
        return _KIND_MOE_GATHER_3D

    raise ValueError(
        f"DistributedMuon got an unexpected param rank {p.ndim} "
        f"(shape={tuple(p.shape)}). Only 2D and 3D params are supported."
    )


def _full_grad(grad: Tensor) -> Tensor:
    """Return a replicated tensor, all-gathering DTensor gradients if needed."""
    if isinstance(grad, DTensor):
        return grad.full_tensor()
    return grad


def _wrap_full_as_dtensor_like(full: Tensor, ref: Tensor) -> Tensor:
    """Wrap ``full`` as a DTensor with ``ref``'s placements."""
    if not isinstance(ref, DTensor):
        return full

    mesh = ref.device_mesh
    replicated = DTensor.from_local(
        full,
        device_mesh=mesh,
        placements=[Replicate()] * mesh.ndim,
        run_check=False,
    )
    return replicated.redistribute(device_mesh=mesh, placements=ref.placements)


def _get_dtensor_shard_info(p: DTensor) -> Tuple[Any, int, int, int]:
    """Extract (process_group, world_size, rank, shard_dim) from a sharded DTensor."""
    mesh = p.device_mesh
    for mesh_dim_idx, placement in enumerate(p.placements):
        if isinstance(placement, Shard):
            pg = mesh.get_group(mesh_dim_idx)
            ws = mesh.size(mesh_dim_idx)
            rk = mesh.get_local_rank(mesh_dim_idx)
            return pg, ws, rk, placement.dim
    raise ValueError(f"No Shard placement found in DTensor with placements={p.placements}")


class DistributedMuon(Optimizer):
    """Muon optimizer with mega-batched Newton-Schulz for FSDP2-sharded params.

    Same-shape FSDP2-sharded 2D parameters are processed together: one batched
    all-gather, one batched NS, then local scatter. This reduces NCCL calls from
    O(num_params) to O(num_unique_shapes).
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
        """Vendor interface."""
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
                if not _is_muon_eligible_ndim(p):
                    raise ValueError(
                        "DistributedMuon supports only 2D and 3D parameters; "
                        f"got param with shape {tuple(p.size())}. Route 1D/4D+ "
                        "params (biases, norms, conv weights) to AdamW via "
                        "split_muon_adamw_params."
                    )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Vendor interface."""
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

            group_config = {
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_coefficients": ns_coefficients,
                "ns_steps": ns_steps,
                "eps": eps,
                "adjust_lr_fn": adjust_lr_fn,
            }

            # Classify ALL params upfront. Group FSDP_GATHER_2D by global shape.
            # CRITICAL: every rank must issue the same collective calls in the same
            # order, so we include ALL params (even grad=None) in the grouping and
            # skip the actual update for grad=None params inside the batch.
            fsdp_2d_groups: Dict[tuple, List[Tensor]] = defaultdict(list)
            other_params: List[Tuple[Tensor, str]] = []

            for p in group["params"]:
                kind = _classify_param(p)
                if kind == _KIND_FSDP_GATHER_2D:
                    global_shape = tuple(p.shape)  # DTensor .shape = global, same on all ranks
                    key = (global_shape, str(p.dtype))
                    fsdp_2d_groups[key].append(p)
                else:
                    if p.grad is None:
                        continue
                    if torch.is_complex(p):
                        raise RuntimeError("DistributedMuon does not support complex parameters")
                    if p.grad.is_sparse:
                        raise RuntimeError("DistributedMuon does not support sparse gradients")
                    other_params.append((p, kind))

            # --- Mega-batch path for FSDP_GATHER_2D ---
            # Build a flat, deterministically-ordered list of same-shape chunks
            # (sorted keys + fixed sub-batch slicing) so every rank issues the
            # exact same sequence of collectives.
            _chunks: List[List[Tensor]] = []
            for _key in sorted(fsdp_2d_groups.keys()):
                _plist = fsdp_2d_groups[_key]
                for _bs in range(0, len(_plist), _MEGABATCH_MAX_GROUP_SIZE):
                    _chunks.append(_plist[_bs:_bs + _MEGABATCH_MAX_GROUP_SIZE])

            _pipe = os.environ.get("AIAK_MUON_PIPELINE", "1") == "1"
            _depth = int(os.environ.get("AIAK_MUON_PIPELINE_DEPTH", "3")) if _pipe else 0
            if _depth >= 1:
                # Pipeline: keep up to _depth async all-gathers (comm) in flight
                # while running Newton-Schulz + apply (compute) of earlier chunks.
                # Numerically identical to the serial path (same gathered data,
                # same NS); only the collective/compute schedule is overlapped.
                _inflight = []
                for _chunk in _chunks:
                    _inflight.append(self._megabatch_launch(_chunk, group_config))
                    if len(_inflight) > _depth:
                        self._megabatch_finish(_inflight.pop(0))
                for _st in _inflight:
                    self._megabatch_finish(_st)
            else:
                for _chunk in _chunks:
                    _st = self._megabatch_launch(_chunk, group_config)
                    self._megabatch_finish(_st)

            # The opt-in local path precomputes only same-shape plain 2D
            # tensors. Updates are still applied by the loop below, preserving
            # the original parameter order and all non-local code paths.
            local_megabatch_orthos: Dict[int, Tensor] = {}
            if os.environ.get("AIAK_MUON_LOCAL_MEGABATCH", "0") == "1":
                local_megabatch_orthos = self._prepare_local_megabatches(
                    other_params, group_config
                )

            # --- Per-param fallback for remaining kinds ---
            # Operate on local tensors to avoid DTensor dispatch issues
            # (torch.compile may produce non-DTensor grads that conflict with
            # DTensor in-place ops).
            for p, kind in other_params:
                ortho_local = local_megabatch_orthos.get(id(p))
                if ortho_local is None:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    buf = state["momentum_buffer"]

                    buf_local = buf.to_local() if isinstance(buf, DTensor) else buf
                    grad_local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad

                    buf_local.lerp_(grad_local, 1 - momentum)
                    if nesterov:
                        update_local = grad_local.lerp(buf_local, momentum)
                    else:
                        update_local = buf_local.clone()

                    # Compute ortho on local tensors
                    if kind == _KIND_LOCAL or kind == _KIND_MOE_LOCAL_3D:
                        ortho_local = batched_newton_schulz(update_local, ns_coefficients, ns_steps, eps)
                    elif kind == _KIND_MOE_GATHER_3D:
                        assert isinstance(p, DTensor)
                        # Need full tensor for NS — gather, compute, slice back
                        full_update = _full_grad(
                            DTensor.from_local(update_local, device_mesh=p.device_mesh,
                                               placements=p.placements, run_check=False)
                        )
                        full_ortho = batched_newton_schulz(full_update, ns_coefficients, ns_steps, eps)
                        # Take this rank's local shard back
                        pg, ws, rk, sdim = _get_dtensor_shard_info(p)
                        global_size = p.shape[sdim]
                        chunk_floor = global_size // ws
                        rem = global_size % ws
                        start = rk * chunk_floor + min(rk, rem)
                        local_size = update_local.shape[sdim]
                        ortho_local = full_ortho.narrow(sdim, start, local_size).contiguous()
                    else:
                        ortho_local = batched_newton_schulz(update_local, ns_coefficients, ns_steps, eps)

                lr_shape = p.shape[-2:] if p.ndim >= 2 else p.shape
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, lr_shape)

                p_local = p.to_local() if isinstance(p, DTensor) else p
                if weight_decay != 0.0:
                    p_local.mul_(1 - lr * weight_decay)
                p_local.add_(ortho_local.to(dtype=p_local.dtype), alpha=-adjusted_lr)

                callback = self.param_update_callback
                if callback is not None:
                    callback(p)

        return loss

    def _prepare_local_megabatches(
        self,
        params_with_kinds: List[Tuple[Tensor, str]],
        config: dict,
    ) -> Dict[int, Tensor]:
        """Return batched orthogonal updates for eligible plain local 2D params.

        Momentum is advanced in the original parameter order. Only chunks with
        at least two tensors are handled here; singleton tails retain the exact
        per-parameter path in ``step``.
        """
        try:
            max_group_size = int(
                os.environ.get(
                    "AIAK_MUON_LOCAL_MEGABATCH_SIZE",
                    str(_LOCAL_MEGABATCH_DEFAULT_GROUP_SIZE),
                )
            )
        except ValueError as exc:
            raise ValueError("AIAK_MUON_LOCAL_MEGABATCH_SIZE must be an integer") from exc
        if max_group_size < 2:
            raise ValueError("AIAK_MUON_LOCAL_MEGABATCH_SIZE must be at least 2")

        grouped: Dict[tuple, List[Tensor]] = defaultdict(list)
        for p, kind in params_with_kinds:
            if kind == _KIND_LOCAL and not isinstance(p, DTensor) and p.ndim == 2:
                key = (kind, tuple(p.shape), p.dtype, p.device)
                grouped[key].append(p)

        chunks: List[List[Tensor]] = []
        eligible_ids = set()
        for plist in grouped.values():
            for start in range(0, len(plist), max_group_size):
                chunk = plist[start:start + max_group_size]
                if len(chunk) < 2:
                    continue
                chunks.append(chunk)
                eligible_ids.update(id(p) for p in chunk)

        if not chunks:
            return {}

        momentum = config["momentum"]
        nesterov = config["nesterov"]
        updates: Dict[int, Tensor] = {}
        # Advance momentum in param-group order, matching the scalar path.
        for p, _kind in params_with_kinds:
            if id(p) not in eligible_ids:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - momentum)
            updates[id(p)] = p.grad.lerp(buf, momentum) if nesterov else buf.clone()

        orthos: Dict[int, Tensor] = {}
        for chunk in chunks:
            stacked_update = torch.stack([updates[id(p)] for p in chunk], dim=0)
            stacked_ortho = batched_newton_schulz(
                stacked_update,
                tuple(config["ns_coefficients"]),
                int(config["ns_steps"]),
                float(config["eps"]),
            )
            for index, p in enumerate(chunk):
                orthos[id(p)] = stacked_ortho[index]
        return orthos

    def _megabatch_launch(self, params: List[Tensor], config: dict) -> Optional[dict]:
        """Momentum update + stack + async all-gather for one same-shape chunk.

        Returns a state dict consumed by ``_megabatch_finish``. The heavy NCCL
        all-gather is issued with ``async_op=True`` so it can overlap with the
        Newton-Schulz compute of a previously-launched chunk.
        """
        if len(params) == 0:
            return None
        momentum = config["momentum"]
        nesterov = config["nesterov"]
        pg, world_size, rank, shard_dim = _get_dtensor_shard_info(params[0])

        local_updates: List[Tensor] = []
        has_grad: List[bool] = []
        for p in params:
            if p.grad is None:
                p_local = p.to_local() if isinstance(p, DTensor) else p
                local_updates.append(torch.zeros_like(p_local))
                has_grad.append(False)
                continue
            has_grad.append(True)
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            buf = state["momentum_buffer"]
            buf_local = buf.to_local() if isinstance(buf, DTensor) else buf
            grad_local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
            buf_local.lerp_(grad_local, 1 - momentum)
            if nesterov:
                update_local = grad_local.lerp(buf_local, momentum)
            else:
                update_local = buf_local.clone()
            local_updates.append(update_local)

        stacked_local = torch.stack(local_updates, dim=0)  # [N, local_M, K]
        del local_updates
        gather_dim = shard_dim + 1
        original_local_size = stacked_local.size(gather_dim)
        global_dim_size = params[0].shape[shard_dim]
        max_local_size = (global_dim_size + world_size - 1) // world_size
        needs_padding = (max_local_size != original_local_size)
        if needs_padding:
            pad_amount = max_local_size - original_local_size
            ndim = stacked_local.ndim
            pad_spec = [0] * (2 * ndim)
            pad_idx = 2 * (ndim - 1 - gather_dim)
            pad_spec[pad_idx + 1] = pad_amount
            stacked_local = torch.nn.functional.pad(stacked_local, pad_spec)

        gather_list = [torch.empty_like(stacked_local) for _ in range(world_size)]
        handle = dist.all_gather(gather_list, stacked_local, group=pg, async_op=True)
        return {
            "handle": handle,
            "gather_list": gather_list,
            "stacked_local": stacked_local,  # keep alive until all-gather completes
            "params": params,
            "has_grad": has_grad,
            "config": config,
            "world_size": world_size,
            "rank": rank,
            "gather_dim": gather_dim,
            "original_local_size": original_local_size,
            "global_dim_size": global_dim_size,
            "max_local_size": max_local_size,
        }

    def _megabatch_finish(self, st: Optional[dict]) -> None:
        """Wait on the async all-gather, then batched Newton-Schulz + apply."""
        if st is None:
            return
        st["handle"].wait()
        config = st["config"]
        ns_coefficients = tuple(config["ns_coefficients"])
        ns_steps = int(config["ns_steps"])
        eps = float(config["eps"])
        lr = config["lr"]
        weight_decay = config["weight_decay"]
        adjust_lr_fn = config["adjust_lr_fn"]
        params = st["params"]
        has_grad = st["has_grad"]
        world_size = st["world_size"]
        rank = st["rank"]
        gather_dim = st["gather_dim"]
        global_dim_size = st["global_dim_size"]
        max_local_size = st["max_local_size"]
        original_local_size = st["original_local_size"]
        gather_list = st["gather_list"]

        lr_shape = params[0].shape[-2:]
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, lr_shape)

        remainder = global_dim_size % world_size
        if remainder == 0:
            stacked_full = torch.cat(gather_list, dim=gather_dim)
        else:
            real_chunks = []
            for r in range(world_size):
                real_size = max_local_size if r < remainder else (global_dim_size // world_size)
                real_chunks.append(gather_list[r].narrow(gather_dim, 0, real_size))
            stacked_full = torch.cat(real_chunks, dim=gather_dim)
        del gather_list, st["gather_list"], st["stacked_local"]

        stacked_ortho = batched_newton_schulz(stacked_full, ns_coefficients, ns_steps, eps)
        del stacked_full

        chunk_floor = global_dim_size // world_size
        shard_start = rank * chunk_floor + min(rank, remainder)
        local_ortho_batch = stacked_ortho.narrow(gather_dim, shard_start, original_local_size)
        del stacked_ortho

        for i, p in enumerate(params):
            if not has_grad[i]:
                continue
            ortho_local = local_ortho_batch[i]
            p_local = p.to_local() if isinstance(p, DTensor) else p
            if weight_decay != 0.0:
                p_local.mul_(1 - lr * weight_decay)
            p_local.add_(ortho_local.to(dtype=p_local.dtype), alpha=-adjusted_lr)

    @staticmethod
    def _compute_ortho(
        update: Tensor,
        kind: str,
        ns_coefficients: Tuple[float, float, float],
        ns_steps: int,
        eps: float,
    ) -> Tensor:
        """Run Newton-Schulz on update according to its layout kind."""
        if kind == _KIND_LOCAL:
            return batched_newton_schulz(update, ns_coefficients, ns_steps, eps)
        if kind == _KIND_FSDP_GATHER_2D:
            full = _full_grad(update)
            return batched_newton_schulz(full, ns_coefficients, ns_steps, eps)
        if kind == _KIND_MOE_LOCAL_3D:
            assert isinstance(update, DTensor)
            local = update._local_tensor
            local_ortho = batched_newton_schulz(local, ns_coefficients, ns_steps, eps)
            return DTensor.from_local(
                local_ortho,
                device_mesh=update.device_mesh,
                placements=update.placements,
                run_check=False,
            )
        if kind == _KIND_MOE_GATHER_3D:
            full = _full_grad(update)
            return batched_newton_schulz(full, ns_coefficients, ns_steps, eps)
        raise ValueError(f"Unknown DistributedMuon kind: {kind!r}")
