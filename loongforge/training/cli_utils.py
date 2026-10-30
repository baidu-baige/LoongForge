# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse


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


def parse_class_names(value: str) -> list[str]:
    """Parse a comma-separated list of class names."""
    return [name.strip() for name in (value or "").split(",") if name.strip()]


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
