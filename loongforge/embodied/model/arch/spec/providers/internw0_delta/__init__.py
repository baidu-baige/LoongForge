# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``internw0_delta`` provider package.

InternW0-Delta: five loss terms, three of them
distillation heads.

Importing this package registers the provider; the spec itself
lives in ``spec.py``.
"""

from __future__ import annotations

from . import spec  # noqa: F401  importing is registering

__all__: list[str] = []
