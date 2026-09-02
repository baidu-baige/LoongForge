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
    raw_module_patterns: str | list[str] | None,
    raw_skip_modules: str | list[str] | None,
    fp8_backend: str | None = None,
) -> None:
    """Checkpoint the modules selected by the patterns, minus those the skip patterns match."""
    from loongforge.embodied.train.training_args import parse_module_key_patterns

    module_patterns = parse_module_key_patterns(
        raw_module_patterns,
        option_name="activation checkpoint module patterns",
    )
    skip_patterns = parse_module_key_patterns(
        raw_skip_modules,
        option_name="activation checkpoint skip modules",
    )
    if not module_patterns and skip_patterns:
        raise ValueError(
            "activation checkpoint skip modules require checkpoint module patterns"
        )
    if not module_patterns:
        return

    selected_modules = _resolve_module_key_patterns(model, module_patterns)
    # Same segment-wise glob as the patterns, so a literal key still means exactly itself.
    matched_skip_patterns = set()
    skip_module_keys = set()
    for module_key in selected_modules:
        for pattern in skip_patterns:
            if _module_key_matches(pattern, module_key):
                matched_skip_patterns.add(pattern)
                skip_module_keys.add(module_key)
                break
    unmatched_skip_patterns = set(skip_patterns).difference(matched_skip_patterns)
    if unmatched_skip_patterns:
        raise ValueError(
            "activation checkpoint skip modules were not selected: "
            + ", ".join(sorted(unmatched_skip_patterns))
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

    # TE's FP8 path saves a different set of tensors during the original forward
    # than during recompute, which makes the non-reentrant PyTorch checkpoint
    # raise CheckpointError. TE ships its own checkpoint that handles this.
    checkpoint_fn = None
    if fp8_backend == "te":
        from loongforge.embodied.distributed.fp8_utils import te_checkpoint_fn

        checkpoint_fn = te_checkpoint_fn()

    checkpoint_wrapper_kwargs = {
        "checkpoint_impl": CheckpointImpl.NO_REENTRANT,
    }
    if checkpoint_fn is not None:
        checkpoint_wrapper_kwargs["checkpoint_fn"] = checkpoint_fn

    for module_key, module in selected_modules.items():
        model.set_submodule(
            module_key,
            checkpoint_wrapper(
                module,
                **checkpoint_wrapper_kwargs,
            ),
        )
    logger.info(
        "Applied activation checkpointing: wrapped=%d skipped=%d impl=%s",
        len(selected_modules),
        len(skip_module_keys),
        "te.checkpoint" if fp8_backend == "te" else "torch non-reentrant",
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
