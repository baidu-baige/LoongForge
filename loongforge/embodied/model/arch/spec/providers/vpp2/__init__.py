# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``vpp2`` provider package.

VPP2: an i2v video stream (two cross-attention slots) plus an
action stream, two loss terms.

Importing this package registers the provider; the spec itself
lives in ``spec.py``.
"""

from __future__ import annotations

from . import spec  # noqa: F401  importing is registering

__all__: list[str] = []
