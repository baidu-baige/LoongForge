# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 visibility: declarations, expression primitives and mask materialization."""

from __future__ import annotations

from .combine import ALWAYS_VISIBLE, Predicate, Visibility, from_predicate
from .declarative import ReadGate, ReadPolicy, ReaderSpec, VisibilityTopology
from .mask import (
    AttentionPlan, Capability, CompiledMask, LayerAttentionOverride, Scenario,
)
from .rules import (
    block_coupling, chunk_causal, frame_causal, history, noise_gate,
    prefix_only, same_sample, same_stream, stream_edge,
)

__all__ = [
    # combine
    "Predicate", "Visibility", "from_predicate", "ALWAYS_VISIBLE",
    # declarative
    "ReadGate", "ReadPolicy", "ReaderSpec", "VisibilityTopology",
    # rules
    "same_stream", "stream_edge", "chunk_causal", "frame_causal",
    "prefix_only", "noise_gate", "same_sample", "history", "block_coupling",
    # mask
    "AttentionPlan", "Capability", "Scenario", "CompiledMask",
    "LayerAttentionOverride",
]
