# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""LingBot-specific FSDP2 adapter over the public fully_shard runtime."""


import torch
import torch.nn as nn
from collections import defaultdict
from dataclasses import replace as dataclass_replace
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from loongforge.embodied.distributed.fsdp_utils.builders import (
    build_fsdp_device_mesh,
    build_ignored_params,
    build_mp_policy,
)
from loongforge.embodied.model.lingbot_va.features import feature_enabled
from loongforge.embodied.model.lingbot_va.modules.wan_model import WanTransformerBlock


def module_params(
    module: nn.Module,
    recurse: bool = True,
    excluded_param_ids: set[int] | None = None,
) -> list[nn.Parameter]:
    """Return unique parameters, optionally excluding ids already managed."""
    params = []
    seen = set()
    excluded_param_ids = excluded_param_ids or set()
    for param in module.parameters(recurse=recurse):
        param_id = id(param)
        if param_id in seen:
            continue
        if param_id in excluded_param_ids:
            continue
        params.append(param)
        seen.add(param_id)
    return params


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


def _build_lingbot_mp_policy(training_args, model):
    """Upstream mixed-precision policy, with the LingBot BF16-reduce switch on top.

    ``build_mp_policy`` resolves param/output dtypes and ``cast_forward_inputs``
    from the public CLI, so those stay a single source of truth.  Only the
    reduction dtype is LingBot's own decision: reducing gradients in the
    all-gather dtype removes a full fp32 reduce-scatter per step, which is worth
    ~1% of the step time here, so the switch overrides whatever
    ``--fsdp-reduce-dtype`` says instead of relying on the launcher to keep the
    two in sync.
    """
    mp_policy = build_mp_policy(training_args, model)
    use_bf16_reduce = feature_enabled("LINGBOT_FSDP_BF16_REDUCE")
    if use_bf16_reduce:
        mp_policy = dataclass_replace(mp_policy, reduce_dtype=mp_policy.param_dtype)
    return mp_policy, use_bf16_reduce


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

    if not getattr(ctx, "is_distributed", False):
        return model.to(device=ctx.device)

    _ensure_fsdp_param_compat()
    model.to(device=ctx.device)
    mp_policy, use_bf16_reduce = _build_lingbot_mp_policy(training_args, model)
    reshard = feature_enabled("LINGBOT_FSDP_RESHARD")
    # 0-dim parameters cannot be sharded on dim-0; upstream collects them and
    # moves them onto the compute device, and every group must ignore the same
    # set, so this is built once for the whole pass.
    scalar_ignored = build_ignored_params(model, ctx)
    fsdp_kwargs = {
        "mesh": build_fsdp_device_mesh(training_args, ctx),
        "reshard_after_forward": reshard,
        "mp_policy": mp_policy,
    }

    attrs = _save_custom_attrs(model)
    wrapped_params = set(scalar_ignored)
    wrapped_param_ids = {id(param) for param in scalar_ignored}

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

    wrapped_blocks = []
    for sub_module in model.modules():
        if isinstance(sub_module, WanTransformerBlock):
            nested_fully_shard(sub_module)
            wrapped_blocks.append(sub_module)
    nested_fully_shard(model)
    _restore_custom_attrs(model, attrs)

    if _rank0():
        print(
            "LingBot native torch nested FSDP2 wrap enabled "
            f"blocks={len(wrapped_blocks)} child_wrap=none "
            f"reshard_after_forward={reshard} keep_fp32_params=True "
            f"mp_policy_param_dtype={mp_policy.param_dtype} "
            f"mp_policy_reduce_dtype={mp_policy.reduce_dtype} "
            f"mp_policy_cast_forward_inputs={mp_policy.cast_forward_inputs} "
            f"bf16_reduce_enabled={use_bf16_reduce} "
            f"scalar_ignored_params={len(scalar_ignored)} ignored_params=True.",
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


def register_lingbot_post_step_reshard(
    model,
    optimizer,
):
    """Reshard every LingBot FSDP group after AdamW updates its local shard."""
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
            "LingBot optimizer must expose register_step_post_hook for post-step reshard"
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
                "[lingbot-post-step-reshard] "
                f"active modules={len(fsdp_modules)}",
                flush=True,
            )
            logged = True

    optimizer_hook = optimizer.register_step_post_hook(post_step_reshard)
    return optimizer_hook, len(fsdp_modules)


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
    reshard = feature_enabled("LINGBOT_FSDP_RESHARD")
    for module in modules:
        if hasattr(module, "set_reshard_after_backward"):
            module.set_reshard_after_backward(reshard, recurse=False)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if modules and rank == 0:
        print(
            "[lingbot-fsdp2] "
            f"reshard_after_backward={reshard} applied to {len(modules)} modules",
            flush=True,
        )
