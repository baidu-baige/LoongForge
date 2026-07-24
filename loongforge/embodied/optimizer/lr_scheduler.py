# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Per-module LR groups + scheduler factory."""

import logging
from collections.abc import Iterable
from typing import Dict, List

import torch.nn as nn

from loongforge.embodied.distributed.utils import is_rank_zero, unwrap_model
from torch.optim.lr_scheduler import LambdaLR
from loongforge.embodied.optimizer.custom_lr_scheduler import LambdaLinearScheduler
from transformers import get_scheduler

logger = logging.getLogger(__name__)


_NORM_CLASSES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LocalResponseNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


def _iter_no_weight_decay_names(module: nn.Module) -> Iterable[str]:
    hook = getattr(module, "no_weight_decay", None)
    if hook is None or not callable(hook):
        return ()
    names = hook()
    if names is None:
        return ()
    if isinstance(names, str):
        return (names,)
    return names


def _module_is_norm(module: nn.Module) -> bool:
    if isinstance(module, _NORM_CLASSES):
        return True
    return "norm" in module.__class__.__name__.lower()


def _bias_norm_no_weight_decay(model: nn.Module):
    """Return a generic bias/norm no-decay predicate."""
    norm_param_ids: set[int] = set()
    explicit_param_names: set[str] = set()
    for module_name, module in model.named_modules():
        if _module_is_norm(module):
            for param in module.parameters(recurse=False):
                norm_param_ids.add(id(param))
        for name in _iter_no_weight_decay_names(module):
            name = str(name).strip()
            if not name:
                continue
            if module_name and not name.startswith(f"{module_name}."):
                name = f"{module_name}.{name}"
            explicit_param_names.add(name)

    def no_decay(name: str, param) -> bool:
        return (
            name == "bias"
            or name.endswith(".bias")
            or id(param) in norm_param_ids
            or name in explicit_param_names
        )

    logger.info(
        "Bias/norm weight-decay grouping enabled: "
        "norm_param_tensors=%d explicit_param_names=%d",
        len(norm_param_ids),
        len(explicit_param_names),
    )
    return no_decay


def _weight_decay_grouping_predicate(model: nn.Module, training_args):
    grouping = training_args.weight_decay_grouping
    if grouping in (None, "", "all"):
        return None
    if grouping == "bias_norm":
        return _bias_norm_no_weight_decay(model)
    raise ValueError(
        f"Unknown weight decay grouping '{grouping}'. Supported values: all, bias_norm."
    )


def _append_param_group(
    groups: List[Dict],
    named_params: list[tuple[str, nn.Parameter]],
    *,
    lr: float,
    name: str,
    no_decay,
    weight_decay: float,
) -> None:
    if not named_params:
        return
    if no_decay is None:
        groups.append(
            {"params": [param for _, param in named_params], "lr": lr, "name": name}
        )
        return

    decay_params = []
    no_decay_params = []
    for param_name, param in named_params:
        if no_decay(param_name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    if decay_params:
        groups.append(
            {
                "params": decay_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": f"{name}.decay",
            }
        )
    if no_decay_params:
        groups.append(
            {
                "params": no_decay_params,
                "lr": lr,
                "weight_decay": 0.0,
                "name": f"{name}.no_decay",
            }
        )


def _log_model_lr(model: nn.Module, max_depth: int = 3, groups: List[Dict] = None) -> None:
    """Log named submodules with trainable parameter counts, and optionally their LR assignment.

    When ``groups`` is provided, each module row also shows the lr value assigned to
    its parameters (aggregated from the first param found in that module).
    Only logs on rank 0 (or when distributed is not initialized).
    """

    if not is_rank_zero():
        return

    # Build param_id → lr mapping when groups are available
    param_to_lr: dict = {}
    if groups is not None:
        for group in groups:
            for p in group.get("params", []):
                param_to_lr[id(p)] = group["lr"]

    if groups is not None:
        title = "[LR Groups] Model modules with LR assignment:"
    else:
        title = (
            "[LR Groups] Model modules"
            " (use paths below with --lr-group to set per-module LR):"
        )
    lines = [title]
    for name, module in model.named_modules():
        if not name:
            continue
        depth = name.count(".")
        if depth >= max_depth:
            continue
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if trainable == 0:
            continue
        indent = "  " + "  " * depth
        if trainable >= 1_000_000:
            param_str = f"{trainable / 1e6:.1f}M"
        elif trainable >= 1_000:
            param_str = f"{trainable / 1e3:.1f}K"
        else:
            param_str = str(trainable)

        if groups is not None:
            lrs = {param_to_lr[id(p)] for p in module.parameters() if p.requires_grad and id(p) in param_to_lr}
            if not lrs:
                lr_str = "  lr=frozen"
            elif len(lrs) == 1:
                lr_str = f"  lr={lrs.pop():.2e}"
            else:
                lr_str = "  lr=mixed(" + ", ".join(f"{v:.2e}" for v in sorted(lrs)) + ")"
        else:
            lr_str = ""

        lines.append(f"{indent}{name:<60s}  ({param_str} trainable params){lr_str}")
    logger.info("\n".join(lines))


def _parse_lr_group(lr_group_str: str) -> list[tuple[str, float]]:
    """Parse 'path1=lr1,path2=lr2' into an ordered [(path, lr)] list."""
    result = []
    for item in lr_group_str.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid --lr-group entry '{item}': expected 'module.path=lr'"
            )
        path, lr_str = item.rsplit("=", 1)
        result.append((path.strip(), float(lr_str.strip())))
    return result


def build_param_groups(model: nn.Module, training_args) -> List[Dict]:
    """Build optimizer param groups with per-module LR from CLI training_args.

    LR assignment priority (highest to lowest):
      1. ``--lr-group``  — comma-separated ``module.path=lr`` pairs.
         Entries are processed in order; earlier entries consume parameters
         first, so more specific (deeper) paths should be listed before
         broader ancestor paths.
         Example: ``model.paligemma_with_expert.gemma_expert=1e-4,
                   model.paligemma_with_expert=1e-5``
      2. ``--lr-base``  — fallback for all remaining trainable parameters.

    Parameters are never double-counted: once a parameter is assigned to a
    group it is excluded from all subsequent groups.
    """
    raw = unwrap_model(model)
    name_by_id = {id(param): name for name, param in raw.named_parameters()}
    frozen_ids = {id(p) for p in raw.parameters() if not p.requires_grad}
    used_ids = set()
    groups = []
    no_decay = _weight_decay_grouping_predicate(raw, training_args)

    base_lr = training_args.lr_base

    _log_model_lr(raw)

    lr_mappings: list[tuple[str, float]] = []
    lr_group_str = training_args.lr_group

    if lr_group_str:
        lr_mappings = _parse_lr_group(lr_group_str)

    for path, lr_val in lr_mappings:
        module = raw
        try:
            for attr in path.split("."):
                module = getattr(module, attr)
        except AttributeError:
            continue

        parameters = module.parameters() if isinstance(module, nn.Module) else [module]
        named_params = [
            (name_by_id[id(param)], param)
            for param in parameters
            if param.requires_grad
            and id(param) not in frozen_ids
            and id(param) not in used_ids
        ]
        if named_params:
            _append_param_group(
                groups,
                named_params,
                lr=lr_val,
                name=path,
                no_decay=no_decay,
                weight_decay=training_args.weight_decay,
            )
            used_ids.update(id(param) for _, param in named_params)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"LR group '{path}': lr={lr_val}, params={len(named_params)}"
                )

    # Base group: everything else
    other = [
        (name_by_id.get(id(param), f"param_{index}"), param)
        for index, param in enumerate(raw.parameters())
        if param.requires_grad
        and id(param) not in used_ids
        and id(param) not in frozen_ids
    ]
    if other:
        _append_param_group(
            groups,
            other,
            lr=base_lr,
            name="base",
            no_decay=no_decay,
            weight_decay=training_args.weight_decay,
        )

    _log_model_lr(raw, 3, groups=groups)

    return groups


def build_scheduler(optimizer, training_args):
    """Build LR scheduler from CLI training_args."""

    if training_args.lr_decay_style == "lambda_linear":
        cycle_len = training_args.lambda_cycle_length or training_args.train_iters

        _scheduler = LambdaLinearScheduler(
            warm_up_steps=[training_args.lr_warmup_iters],
            f_min=[training_args.lambda_f_min],
            f_max=[training_args.lambda_f_max],
            f_start=[training_args.lambda_f_start],
            cycle_lengths=[cycle_len]
        )

        logger.info(
            f"LambdaLinear scheduler: f_max={training_args.lambda_f_max}, "
            f"f_min={training_args.lambda_f_min}, warmup={training_args.lr_warmup_iters}, "
            f"cycle_len={cycle_len}"
        )

        return LambdaLR(optimizer, _scheduler.schedule)
    else:
        kwargs = {}
        style = training_args.lr_decay_style
        if style in {"cosine_with_min_lr", "cosine_warmup_with_min_lr"}:
            kwargs["min_lr"] = training_args.min_lr
        elif style == "polynomial":
            kwargs["lr_end"] = training_args.lr_end
            kwargs["power"] = training_args.polynomial_power
        elif style == "cosine_with_restarts":
            kwargs["num_cycles"] = training_args.num_cycles

        num_training_steps = int(
            training_args.lr_decay_iters or training_args.train_iters
        )
        return get_scheduler(
            name=style,
            optimizer=optimizer,
            num_warmup_steps=training_args.lr_warmup_iters,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=kwargs,
        )
