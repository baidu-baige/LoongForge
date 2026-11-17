# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L4 输出头（Lane C）。"""

from __future__ import annotations

from .head import OutputHead
from .hidden_ref import HiddenStateRef, captures_from_refs
from .spec import HeadSpec

__all__ = ["HiddenStateRef", "captures_from_refs", "OutputHead", "HeadSpec"]
