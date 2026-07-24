# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""LingBot-specific FSDP2 adapter over the public fully_shard runtime."""

import torch
from collections import defaultdict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from loongforge.embodied.distributed.utils import module_params
from loongforge.embodied.model.lingbot_va.modules.wan_model import WanTransformerBlock


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_str]


def _rank0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _ensure_fsdp_param_compat():
    """Patch FSDP2 grad accumulation for PyTorch builds with stale private access."""
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except Exception:
        return
    if getattr(FSDPParam, "_lingbot_accum_grad_compat", False):
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
    FSDPParam._lingbot_accum_grad_compat = True


def _build_embodied_device_mesh(training_args, ctx):
    shard_size = getattr(training_args, "hsdp_shard_size", None)
    if shard_size is None:
        return init_device_mesh("cuda", (ctx.world_size,), mesh_dim_names=("dp",))
    if shard_size <= 0:
        raise ValueError(f"HSDP shard size must be positive, got {shard_size}.")
    if ctx.world_size % shard_size != 0:
        raise ValueError(
            "HSDP requires world_size to be divisible by hsdp_shard_size, "
            f"got world_size={ctx.world_size}, hsdp_shard_size={shard_size}."
        )
    return init_device_mesh(
        "cuda",
        (ctx.world_size // shard_size, shard_size),
        mesh_dim_names=("replica", "shard"),
    )


def _save_custom_attrs(module):
    return {name: dict(vars(param)) for name, param in module.named_parameters()}


def _restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        for attr_name, attr_value in custom_attrs.get(name, {}).items():
            setattr(param, attr_name, attr_value)


def wrap_lingbot_torch_nested_fsdp2(model, training_args, ctx):
    """Apply the phase4 block+root FSDP2 order and mixed-precision policy."""
    if getattr(training_args, "distributed_strategy", None) != "fsdp":
        raise RuntimeError(
            "LingBot native nested FSDP2 requires embodied FSDP strategy"
        )

    dtype = _resolve_dtype(training_args.dtype)
    if not getattr(ctx, "is_distributed", False):
        return model.to(device=ctx.device)

    _ensure_fsdp_param_compat()
    model.to(device=ctx.device)
    fsdp_kwargs = {
        "mesh": _build_embodied_device_mesh(training_args, ctx),
        "reshard_after_forward": False,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=dtype,
            cast_forward_inputs=False,
        ),
    }

    attrs = _save_custom_attrs(model)
    wrapped_params = set()
    wrapped_param_ids = set()

    def nested_fully_shard(module):
        shard_kwargs = dict(fsdp_kwargs)
        params_before = list(
            module_params(module, excluded_param_ids=wrapped_param_ids)
        )
        if wrapped_params:
            shard_kwargs["ignored_params"] = wrapped_params
        fully_shard(module, **shard_kwargs)
        for param in params_before:
            wrapped_params.add(param)
            wrapped_param_ids.add(id(param))

    wrapped_blocks = 0
    for sub_module in model.modules():
        if isinstance(sub_module, WanTransformerBlock):
            nested_fully_shard(sub_module)
            wrapped_blocks += 1
    nested_fully_shard(model)
    _restore_custom_attrs(model, attrs)

    if _rank0():
        print(
            "LingBot native torch nested FSDP2 wrap enabled "
            f"blocks={wrapped_blocks} child_wrap=none "
            "reshard_after_forward=False keep_fp32_params=True "
            f"mp_policy_param_dtype={dtype} mp_policy_reduce_dtype={dtype} ignored_params=True.",
            flush=True,
        )
    return model


_LINGBOT_FSDP2_SETUP_DONE = "_lingbot_fsdp2_setup_done"
_LINGBOT_DTENSOR_CLIP_LOGGED = False


def _lingbot_optimizer_parameters(optimizer):
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


def _lingbot_local_gradient_groups(optimizer):
    """Collect mutable local DTensor gradients by device and dtype."""
    parameters = _lingbot_optimizer_parameters(optimizer)
    non_dtensor = [
        parameter for parameter in parameters if not isinstance(parameter, DTensor)
    ]
    if non_dtensor:
        raise RuntimeError(
            "LingBot optimizer-owned gradient handling requires pure DTensor parameters; "
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
                "LingBot optimizer-owned gradient handling requires DTensor gradients; "
                f"got {type(gradient).__name__}."
            )
        local_gradient = gradient._local_tensor
        if local_gradient.is_sparse:
            raise RuntimeError(
                "LingBot FSDP2 gradient handling does not support sparse gradients."
            )
        groups[(local_gradient.device, local_gradient.dtype)].append(local_gradient)
        gradient_count += 1
    return parameters, list(groups.values()), gradient_count


def _lingbot_local_norm_sq(gradient_groups, device):
    total_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
    for gradients in gradient_groups:
        norms = torch._foreach_norm(gradients, 2.0, dtype=torch.float32)
        total_norm_sq += torch.stack(norms).square().sum()
    return total_norm_sq


def clip_lingbot_optimizer_gradients(optimizer, max_norm):
    """Clip RAB=false optimizer-owned DTensor gradients by global L2 norm.

    FSDP2 leaves the reduced sharded gradients on the DTensor parameters held
    by the optimizer.  The model's currently materialized parameters may have
    no ``.grad``, so this helper intentionally starts from ``param_groups``.
    """
    global _LINGBOT_DTENSOR_CLIP_LOGGED

    if max_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}.")
    parameters, gradient_groups, gradient_count = _lingbot_local_gradient_groups(
        optimizer
    )
    if parameters:
        device = parameters[0]._local_tensor.device
    else:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    total_norm_sq = _lingbot_local_norm_sq(gradient_groups, device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(total_norm_sq, op=torch.distributed.ReduceOp.SUM)
    total_norm = total_norm_sq.sqrt()
    clip_coefficient = torch.clamp(
        torch.as_tensor(max_norm, device=device, dtype=torch.float32)
        / (total_norm + 1e-6),
        max=1.0,
    )
    for gradients in gradient_groups:
        torch._foreach_mul_(gradients, clip_coefficient)

    if not _LINGBOT_DTENSOR_CLIP_LOGGED and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        print(
            "[lingbot-dtensor-clip] "
            f"optimizer_params={len(parameters)} dtensor_params={len(parameters)} "
            f"gradients={gradient_count} global_grad_norm={total_norm.item():.10g} "
            f"clip_coefficient={clip_coefficient.item():.10g}",
            flush=True,
        )
        _LINGBOT_DTENSOR_CLIP_LOGGED = True
    return total_norm.item()


def clean_lingbot_optimizer_gradients(optimizer):
    """Replace NaN/Inf in optimizer-owned local DTensor gradients with zero."""
    _, gradient_groups, _ = _lingbot_local_gradient_groups(optimizer)
    for gradients in gradient_groups:
        for gradient in gradients:
            torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0, out=gradient)


def register_lingbot_post_step_reshard(model, optimizer):
    """Register LingBot's post-optimizer FSDP2 reshard policy.

    The returned handle must be kept alive by the trainer. The optimizer's
    standard post-step hook guarantees that reshard runs after AdamW and
    before the public scheduler advances.
    """
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

    logged = False

    def post_step_reshard(_optimizer, _args, _kwargs):
        nonlocal logged
        for module in fsdp_modules:
            module.reshard()
        if not logged and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            print(
                f"[lingbot-post-step-reshard] active modules={len(fsdp_modules)}",
                flush=True,
            )
            logged = True

    if not hasattr(optimizer, "register_step_post_hook"):
        raise TypeError(
            "LingBot optimizer must expose register_step_post_hook for post-step reshard"
        )
    return optimizer.register_step_post_hook(post_step_reshard), len(fsdp_modules)


def apply_lingbot_fsdp2_tuning(model):
    """Keep FSDP2 parameters unsharded after backward (final RAB=false stack)."""
    if getattr(model, _LINGBOT_FSDP2_SETUP_DONE, False):
        return
    setattr(model, _LINGBOT_FSDP2_SETUP_DONE, True)

    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        return

    modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
    for module in modules:
        if hasattr(module, "set_reshard_after_backward"):
            module.set_reshard_after_backward(False, recurse=False)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if modules and rank == 0:
        print(
            f"[lingbot-fsdp2] reshard_after_backward=False applied to {len(modules)} modules",
            flush=True,
        )
