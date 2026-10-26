# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Value parsers for Torch CLI fields.

CLI definitions live in ``loongforge.training.torch_args``. The engine keeps
these parsers because FSDP, FP8, activation checkpointing, and parallel setup
also use them at runtime.
"""

import argparse

import torch


def parse_reshard_after_forward(value: str):
    """Parse FSDP2 reshard_after_forward from CLI text: true|false|none|int>1."""
    normalized = value.strip().lower()
    if normalized in {"true", "t"}:
        return True
    if normalized in {"false", "f"}:
        return False
    if normalized in {"none", "null"}:
        return None
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected one of: true, false, none, or an integer greater than 1"
        ) from exc
    if int_value <= 1:
        raise argparse.ArgumentTypeError(
            "integer reshard_after_forward must be greater than 1"
        )
    return int_value


def parse_reshard_after_forward_map(raw_map: str):
    """Parse comma-separated ClassName=value pairs for per-module FSDP reshard_after_forward.

    Each pair maps a module class name to a reshard strategy accepted by
    ``parse_reshard_after_forward``.  Unknown keys are passed through as-is.

    Example::

        "TransformerLayer=True,EmbeddingLayer=False"
        # -> {"TransformerLayer": True, "EmbeddingLayer": False}
    """
    class_to_reshard = {}
    for pair in raw_map.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                "expected comma-separated ClassName=value pairs"
            )
        class_name, reshard_value = pair.split("=", 1)
        class_name = class_name.strip()
        if not class_name:
            raise argparse.ArgumentTypeError("empty class name in reshard map")
        class_to_reshard[class_name] = parse_reshard_after_forward(reshard_value)
    return class_to_reshard


def parse_positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if int_value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return int_value


def parse_non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI value."""
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a non-negative integer"
        ) from exc
    if int_value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return int_value


def parse_module_key_patterns(
    value: str | list[str] | None,
    *,
    option_name: str,
) -> list[str]:
    """Parse comma-separated qualified module-key patterns.

    Accepts a raw CLI string or an already-split list so the same helper can
    serve both argparse ``type=`` and the later wrap-time re-parse.
    """
    if isinstance(value, (list, tuple)):
        value = ",".join(value)
    normalized_patterns = (value or "").strip()
    if not normalized_patterns:
        return []

    patterns = [
        pattern.strip()
        for pattern in normalized_patterns.split(",")
        if pattern.strip()
    ]
    if any(
        any(not segment for segment in pattern.split("."))
        for pattern in patterns
    ):
        raise ValueError(f"{option_name} cannot contain empty segments")
    return patterns


def parse_class_names(value: str) -> list[str]:
    """Parse a comma-separated list of class names."""
    return [name.strip() for name in (value or "").split(",") if name.strip()]


def parse_dtype_from_str(dtype_name: str) -> torch.dtype:
    """Map a canonical FSDP CLI dtype name to ``torch.dtype``."""
    DTYPE_MAP_BY_STR = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }

    try:
        return DTYPE_MAP_BY_STR[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported FSDP dtype {dtype_name!r}") from exc


def parse_optional_int_list(raw_value: str | None) -> list[int] | None:
    """Parse a comma-separated string of integers into ``list[int]``.

    Returns ``None`` for empty or whitespace-only input so optional kwargs
    remain unset.

    Example::

        parse_optional_int_list("25,50,100")
        # -> [25, 50, 100]

        parse_optional_int_list(None)
        # -> None
    """
    if not raw_value or not raw_value.strip():
        return None
    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    return [int(item) for item in items] if items else None
