# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 visibility: rules, combinators, presets and materialization (Lane A)."""

from __future__ import annotations

from .combine import ALWAYS_VISIBLE, Predicate, Visibility, from_predicate
from .mask import (
    AttentionPlan, Capability, CompiledMask, LayerAttentionOverride, Scenario,
)
from .presets import PRESETS, get_preset, preset
from .phases import (
    CONDITION, REAL_VIDEO, Phase, condition_phases, phase_labels,
)
from .rules import (
    block_coupling, chunk_causal, frame_causal, history, label_edge,
    noise_gate, prefix_only, same_sample, same_stream, stream_edge,
)

__all__ = [
    # combine
    "Predicate", "Visibility", "from_predicate", "ALWAYS_VISIBLE",
    # rules
    "same_stream", "stream_edge", "label_edge", "chunk_causal", "frame_causal",
    "prefix_only", "noise_gate", "same_sample", "history", "block_coupling",
    # mask
    "AttentionPlan", "Capability", "Scenario", "CompiledMask",
    "LayerAttentionOverride",
    # presets
    "PRESETS", "preset", "get_preset",
    # phases
    "Phase", "CONDITION", "REAL_VIDEO", "condition_phases", "phase_labels",
]
