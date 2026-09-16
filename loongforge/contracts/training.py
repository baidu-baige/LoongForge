# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Resolved training invocation passed from CLI to engine dispatch."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainSpec:
    engine: str
    model: str | None
    config_file: str
    args: tuple[str, ...]
