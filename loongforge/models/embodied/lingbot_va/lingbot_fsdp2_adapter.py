# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-specific FSDP2 adapter over the public fully_shard runtime."""


import torch
import torch.nn as nn
from dataclasses import replace as dataclass_replace
from torch.distributed.fsdp import fully_shard

from loongforge.models.embodied.lingbot_va.features import feature_enabled
from loongforge.models.embodied.lingbot_va.modules.wan_model import WanTransformerBlock


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


def _build_lingbot_mp_policy(mp_policy):
    """Upstream mixed-precision policy, with the LingBot BF16-reduce switch on top.

    ``build_mp_policy`` resolves param/output dtypes and ``cast_forward_inputs``
    from the public CLI, so those stay a single source of truth.  Only the
    reduction dtype is LingBot's own decision: reducing gradients in the
    all-gather dtype removes a full fp32 reduce-scatter per step, which is worth
    ~1% of the step time here, so the switch overrides whatever
    ``--fsdp-reduce-dtype`` says instead of relying on the launcher to keep the
    two in sync.
    """
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


def wrap_lingbot_torch_nested_fsdp2(model, *, mesh, mp_policy, scalar_ignored):
    """Apply the phase4 block+root FSDP2 order and mixed-precision policy."""
    mp_policy, use_bf16_reduce = _build_lingbot_mp_policy(mp_policy)
    reshard = feature_enabled("LINGBOT_FSDP_RESHARD")
    fsdp_kwargs = {
        "mesh": mesh,
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
