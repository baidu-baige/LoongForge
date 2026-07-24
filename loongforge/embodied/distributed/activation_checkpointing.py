# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Framework-managed activation checkpoint selection and wrapping."""

import logging
from collections.abc import Iterable
from fnmatch import fnmatchcase

import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_activation_checkpointing(
    model: nn.Module,
    raw_module_patterns: str | None,
    raw_skip_modules: str | None,
) -> None:
    """Checkpoint selected modules except explicitly skipped module keys."""
    from loongforge.embodied.train.training_args import parse_module_key_patterns

    module_patterns = parse_module_key_patterns(
        raw_module_patterns,
        option_name="activation checkpoint module patterns",
    )
    skip_module_keys = set(
        parse_module_key_patterns(
            raw_skip_modules,
            option_name="activation checkpoint skip modules",
        )
    )
    if not module_patterns and skip_module_keys:
        raise ValueError(
            "activation checkpoint skip modules require checkpoint module patterns"
        )
    if not module_patterns:
        return

    selected_modules = _resolve_module_key_patterns(model, module_patterns)
    unknown_skip_modules = skip_module_keys.difference(selected_modules)
    if unknown_skip_modules:
        raise ValueError(
            "activation checkpoint skip modules were not selected: "
            + ", ".join(sorted(unknown_skip_modules))
        )
    selected_modules = {
        module_key: module
        for module_key, module in selected_modules.items()
        if module_key not in skip_module_keys
    }
    _validate_non_nested_module_keys(selected_modules)

    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            checkpoint_wrapper,
        )
    except ImportError as exc:
        raise RuntimeError(
            "framework-managed activation checkpointing requires PyTorch "
            "checkpoint_wrapper"
        ) from exc

    for module_key, module in selected_modules.items():
        model.set_submodule(
            module_key,
            checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
        )
    logger.info(
        "Applied activation checkpointing: wrapped=%d skipped=%d",
        len(selected_modules),
        len(skip_module_keys),
    )


def _resolve_module_key_patterns(
    model: nn.Module,
    patterns: list[str],
) -> dict[str, nn.Module]:
    """Resolve qualified module-key patterns and reject unmatched patterns."""
    matched_patterns = set()
    selected_modules = {}
    for module_key, module in model.named_modules():
        for pattern in patterns:
            if module_key and _module_key_matches(pattern, module_key):
                matched_patterns.add(pattern)
                selected_modules[module_key] = module
                break

    unmatched_patterns = [
        pattern for pattern in patterns if pattern not in matched_patterns
    ]
    if unmatched_patterns:
        raise ValueError(
            "activation checkpoint module patterns matched no modules: "
            + ", ".join(unmatched_patterns)
        )
    return selected_modules


def _validate_non_nested_module_keys(module_keys: Iterable[str]) -> None:
    """Reject selections containing both a module and one of its descendants."""
    selected_keys = set(module_keys)
    for module_key in selected_keys:
        segments = module_key.split(".")
        for depth in range(1, len(segments)):
            parent_key = ".".join(segments[:depth])
            if parent_key in selected_keys:
                raise ValueError(
                    "activation checkpoint module patterns cannot select both "
                    f"parent {parent_key!r} and nested module {module_key!r}"
                )


def _module_key_matches(pattern: str, module_key: str) -> bool:
    """Match a qualified module key without allowing ``*`` to cross dots."""
    pattern_segments = pattern.split(".")
    module_key_segments = module_key.split(".")
    return len(pattern_segments) == len(module_key_segments) and all(
        fnmatchcase(module_key_segment, pattern_segment)
        for pattern_segment, module_key_segment in zip(
            pattern_segments,
            module_key_segments,
        )
    )
