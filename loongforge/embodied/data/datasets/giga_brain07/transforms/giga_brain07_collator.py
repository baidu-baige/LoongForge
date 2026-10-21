# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain07 data collator — xvla-style naming surface.

Layer-A reshell: thin **re-export** mirroring
``data/datasets/xvla/transforms/xvla_collator.py`` and
``data/datasets/pi05/transforms/pi05_collator.py`` so ``giga_brain07``
exposes its collator under the expected ``<name>_collator`` module.

The real collator is **unchanged** and lives in the vendored package
``giga_datasets`` (``giga_datasets.collators.default_collator.DefaultCollator``).
The LoongForge custom trainer (``GigaBrain07Trainer._build_dataloaders``) still
imports it from ``giga_datasets`` directly — this file only provides a
naming-aligned import surface and reimplements nothing.

Import prerequisite (identical to the trainer / run script PYTHONPATH):
    data/datasets/giga_brain07   -> top-level ``giga_datasets`` package
"""

from __future__ import annotations

from giga_datasets import DefaultCollator

__all__ = ["DefaultCollator"]
