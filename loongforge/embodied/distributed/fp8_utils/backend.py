# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Backend dispatch for TransformerEngine and TorchAO FP8 training."""

import logging
from contextlib import nullcontext

import torch.nn as nn

from .targets import resolve_fp8_targets


logger = logging.getLogger(__name__)

FP8_BACKEND_CHOICES = ("te", "torchao")


def apply_fp8_linear_conversion(model: nn.Module, training_args, device) -> int:
    """Apply and verify configured FP8 Linear replacements before wrapping."""
    if not getattr(training_args, "fp8", False):
        return 0

    original_linear_count = sum(
        type(module) is nn.Linear for module in model.modules()
    )
    module_patterns, skip_modules = resolve_fp8_targets(model, training_args)
    fp8_linear_count = convert_linear_for_fp8(
        model,
        module_patterns,
        skip_modules,
        training_args,
        device,
    )
    if fp8_linear_count == 0:
        raise RuntimeError(
            "FP8 is enabled but no nn.Linear modules were converted; "
            "check fp8_module_patterns, fp8_skip_modules, fp8_min_dim, "
            "and backend alignment requirements."
        )

    remaining_linear_count = sum(
        type(module) is nn.Linear for module in model.modules()
    )
    expected_remaining_linear_count = original_linear_count - fp8_linear_count
    if remaining_linear_count != expected_remaining_linear_count:
        raise RuntimeError(
            "FP8 Linear conversion verification failed: "
            f"before={original_linear_count} converted={fp8_linear_count} "
            f"expected_after={expected_remaining_linear_count} "
            f"actual_after={remaining_linear_count}."
        )
    logger.info(
        "FP8 Linear replacement summary: backend=%s original_linear=%d "
        "fp8_linear=%d remaining_linear=%d",
        training_args.fp8_backend,
        original_linear_count,
        fp8_linear_count,
        remaining_linear_count,
    )
    return fp8_linear_count


def convert_linear_for_fp8(
    model: nn.Module,
    module_patterns,
    skip_modules,
    training_args,
    device,
) -> int:
    """Dispatch structural Linear conversion to the selected FP8 backend."""
    if training_args.fp8_backend == "te":
        from .te_fp8 import convert_linear_to_te

        return convert_linear_to_te(
            model,
            module_patterns,
            skip_modules,
            training_args.fp8_min_dim,
            device,
        )
    if training_args.fp8_backend == "torchao":
        from .torchao_fp8 import convert_linear_to_torchao

        return convert_linear_to_torchao(
            model,
            module_patterns,
            skip_modules,
            training_args.fp8_min_dim,
            training_args,
        )
    raise ValueError(
        f"Unknown FP8 backend {training_args.fp8_backend!r}; "
        f"expected one of {', '.join(FP8_BACKEND_CHOICES)}."
    )


def resolve_fp8_forward_ctx(training_args, fp8_group=None):
    """Return the backend's forward context; TorchAO needs no outer context."""
    if not getattr(training_args, "fp8", False):
        return nullcontext()
    if training_args.fp8_backend == "torchao":
        return nullcontext()
    if training_args.fp8_backend == "te":
        from .te_fp8 import resolve_fp8_autocast_ctx

        return resolve_fp8_autocast_ctx(
            training_args,
            fp8_group=fp8_group,
        )
    raise ValueError(f"Unknown FP8 backend {training_args.fp8_backend!r}.")
