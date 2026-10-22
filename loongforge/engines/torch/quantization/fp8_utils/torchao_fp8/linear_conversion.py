# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""TorchAO Float8Linear conversion using the shared model target policy."""

from __future__ import annotations

import logging
from dataclasses import replace

import torch.nn as nn

from ..selection import FP8_GEMM_ALIGNMENT, resolve_linear_selection

logger = logging.getLogger(__name__)


def build_torchao_config(training_args):
    """Build ``Float8LinearConfig`` for the selected TorchAO recipe."""
    try:
        from torchao.float8.config import Float8LinearConfig
    except ImportError as exc:
        raise RuntimeError(
            "--fp8-backend=torchao requires torchao.float8."
        ) from exc

    config = Float8LinearConfig.from_recipe_name(
        training_args.fp8_torchao_recipe
    )
    return replace(
        config,
        pad_inner_dim=training_args.fp8_torchao_pad_inner_dim,
        enable_fsdp_float8_all_gather=(
            training_args.fp8_torchao_fsdp_float8_all_gather
        ),
        # With FSDP2, recompute the FP8 weight in backward instead of retaining
        # a complete unsharded FP8 weight/transpose from the forward pass.
        force_recompute_fp8_weight_in_bwd=(
            training_args.fp8_torchao_fsdp_float8_all_gather
        ),
    )


def convert_linear_to_torchao(
    model: nn.Module,
    raw_module_patterns: str | list[str] | None,
    raw_skip_modules: str | list[str] | None,
    min_dim: int,
    training_args,
) -> int:
    """Replace eligible target Linears with TorchAO ``Float8Linear`` modules."""
    selected_keys, skipped_keys = resolve_linear_selection(
        model,
        raw_module_patterns,
        raw_skip_modules,
    )
    if not selected_keys:
        return 0

    pad_inner_dim = training_args.fp8_torchao_pad_inner_dim
    eligible_keys = set()
    below_min_dim = 0
    unaligned_inner = 0
    unaligned_output = 0
    for module_key in selected_keys:
        if module_key in skipped_keys:
            continue
        linear = model.get_submodule(module_key)
        if max(linear.in_features, linear.out_features) < min_dim:
            below_min_dim += 1
            continue
        if linear.out_features % FP8_GEMM_ALIGNMENT:
            unaligned_output += 1
            continue
        if linear.in_features % FP8_GEMM_ALIGNMENT and not pad_inner_dim:
            unaligned_inner += 1
            continue
        eligible_keys.add(module_key)

    try:
        from torchao.float8 import convert_to_float8_training
    except ImportError as exc:
        raise RuntimeError(
            "--fp8-backend=torchao requires "
            "torchao.float8.convert_to_float8_training."
        ) from exc

    config = build_torchao_config(training_args)
    convert_to_float8_training(
        model,
        config=config,
        module_filter_fn=lambda module, fqn: (
            type(module) is nn.Linear and fqn in eligible_keys
        ),
    )
    converted_keys = [
        module_key
        for module_key in eligible_keys
        if type(model.get_submodule(module_key)) is not nn.Linear
    ]
    failed_keys = sorted(eligible_keys.difference(converted_keys))
    if failed_keys:
        raise RuntimeError(
            "TorchAO FP8 conversion did not replace nn.Linear at: "
            + ", ".join(failed_keys)
        )
    logger.info(
        "Converted nn.Linear to TorchAO Float8Linear: converted=%d "
        "skipped_by_pattern=%d below_min_dim=%d unaligned_inner=%d "
        "unaligned_output=%d (min_dim=%d, alignment=%d, pad_inner_dim=%s)",
        len(eligible_keys),
        len(skipped_keys),
        below_min_dim,
        unaligned_inner,
        unaligned_output,
        min_dim,
        FP8_GEMM_ALIGNMENT,
        pad_inner_dim,
    )
    if converted_keys:
        logger.info(
            "TorchAO FP8 converted modules: %s",
            ", ".join(sorted(converted_keys)),
        )
    else:
        logger.warning(
            "TorchAO FP8 conversion converted no modules; selected=%d skipped=%d.",
            len(selected_keys),
            len(skipped_keys),
        )
    return len(eligible_keys)
