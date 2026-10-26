# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-specific FSDP2 adapter over the public fully_shard runtime."""


import torch
import torch.nn as nn
from dataclasses import replace as dataclass_replace
from torch.distributed.fsdp import fully_shard

from loongforge.engines.torch.distributed.fsdp_utils.builders import (
    build_fsdp_device_mesh,
    build_ignored_params,
    build_mp_policy,
)
from loongforge.engines.torch.distributed.fsdp_utils.optimizer_grads import (
    ensure_fsdp_param_compat,
)
from loongforge.engines.torch.quantization.fp8_utils import apply_fp8_linear_conversion
from loongforge.models.embodied.lingbot_va.features import feature_enabled
from loongforge.models.embodied.lingbot_va.modules.wan_model import WanTransformerBlock


def _configure_lingbot_checkpoint(model, training_args):
    """Use TE's FP8-aware checkpoint for LingBot's internal block recompute."""
    if not (
        getattr(training_args, "fp8", False)
        and getattr(training_args, "fp8_backend", None) == "te"
    ):
        return

    backbone = getattr(model, "model", None)
    set_checkpoint = getattr(backbone, "set_block_checkpoint_fn", None)
    if not callable(set_checkpoint):
        raise RuntimeError(
            "LingBot TE FP8 requires a backbone that supports "
            "set_block_checkpoint_fn()."
        )

    from loongforge.engines.torch.quantization.fp8_utils import te_checkpoint_fn

    set_checkpoint(te_checkpoint_fn())


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

    apply_fp8_linear_conversion(model, training_args, ctx.device)
    _configure_lingbot_checkpoint(model, training_args)

    if not getattr(ctx, "is_distributed", False):
        return model.to(device=ctx.device)

    ensure_fsdp_param_compat()
    model.to(device=ctx.device)
    mp_policy, use_bf16_reduce = _build_lingbot_mp_policy(training_args, model)
    reshard = feature_enabled("LINGBOT_FSDP_RESHARD")
    # 0-dim parameters cannot be sharded on dim-0; upstream collects them and
    # moves them onto the compute device, and every group must ignore the same
    # set, so this is built once for the whole pass.
    scalar_ignored = build_ignored_params(training_args, model, ctx)
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
