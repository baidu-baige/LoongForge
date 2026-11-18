# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L4 output heads (Lane C)."""

from __future__ import annotations

from .flow import (IdentityHead, LinearHead, NormLinearHead, PatchFlowHead,
                   build_flow_head)
from .head import OutputHead
from .hidden_ref import HiddenStateRef, captures_from_refs
from .spec import HeadSpec

__all__ = ["HiddenStateRef", "IdentityHead", "LinearHead", "NormLinearHead",
           "OutputHead", "PatchFlowHead", "HeadSpec", "build_flow_head",
           "captures_from_refs"]
