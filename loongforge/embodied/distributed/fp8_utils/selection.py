# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral selection of standard Linear modules for FP8 conversion."""

from __future__ import annotations

import torch.nn as nn

from ..utils import is_module_path_matching_pattern

FP8_GEMM_ALIGNMENT = 16


def resolve_linear_selection(
    model: nn.Module,
    raw_module_patterns: str | list[str] | None,
    raw_skip_modules: str | list[str] | None,
) -> tuple[list[str], set[str]]:
    """Resolve FP8 subtree and skip patterns against exact ``nn.Linear`` keys."""
    from loongforge.embodied.train.training_args import parse_module_key_patterns

    module_patterns = parse_module_key_patterns(
        raw_module_patterns,
        option_name="fp8 module patterns",
    )
    skip_patterns = parse_module_key_patterns(
        raw_skip_modules,
        option_name="fp8 skip modules",
    )
    if not module_patterns and skip_patterns:
        raise ValueError("fp8 skip modules require fp8 module patterns")
    if not module_patterns:
        return [], set()

    linear_keys = [
        module_key
        for module_key, module in model.named_modules()
        if module_key and type(module) is nn.Linear
    ]

    matched_patterns: set[str] = set()
    selected_keys = [
        module_key
        for module_key in linear_keys
        if _select(module_patterns, module_key, matched_patterns)
    ]
    unmatched_patterns = [
        pattern for pattern in module_patterns if pattern not in matched_patterns
    ]
    if unmatched_patterns:
        raise ValueError(
            "fp8 module patterns matched no nn.Linear: "
            + ", ".join(unmatched_patterns)
        )

    matched_skip_patterns: set[str] = set()
    skipped_keys = {
        module_key
        for module_key in selected_keys
        if _select(skip_patterns, module_key, matched_skip_patterns)
    }
    unmatched_skip_patterns = [
        pattern for pattern in skip_patterns if pattern not in matched_skip_patterns
    ]
    if unmatched_skip_patterns:
        raise ValueError(
            "fp8 skip modules were not selected: "
            + ", ".join(unmatched_skip_patterns)
        )
    return selected_keys, skipped_keys


def _select(
    patterns: list[str],
    module_key: str,
    matched_patterns: set[str],
) -> bool:
    """Return whether a pattern selects this module or an ancestor subtree."""
    segments = module_key.split(".")
    candidate_keys = [
        ".".join(segments[:depth]) for depth in range(1, len(segments) + 1)
    ]
    selected = False
    for pattern in patterns:
        if any(
            is_module_path_matching_pattern(pattern, candidate_key)
            for candidate_key in candidate_keys
        ):
            matched_patterns.add(pattern)
            selected = True
    return selected
