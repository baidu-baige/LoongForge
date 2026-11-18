# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 streams, their parameters, and the joint attention engine (Lane B)."""

from __future__ import annotations

from .cache import CacheScope, CacheSignature
from .conditions import (
    Condition, ConditionBus, ConditionSpec, InjectionKind,
)
from .engine import (
    AttentionGroup, EngineOutput, StreamEngine, assert_shape_signature,
)
from .registry import (
    STREAMS, StreamAdapter, StreamRegistry, plan_from_legacy_mask,
)
from .routing import KVRoute
from .spec import (
    DEFAULT_WAN_SLOTS, HEAD_SLOTS, LayerMapping, ModulationSlots, StreamSpec,
)
from .state import QKV, StreamModule, StreamState

__all__ = [
    # spec
    "LayerMapping", "StreamSpec", "ModulationSlots",
    "DEFAULT_WAN_SLOTS", "HEAD_SLOTS",
    # state
    "StreamState", "QKV", "StreamModule",
    # conditions
    "InjectionKind", "Condition", "ConditionBus", "ConditionSpec",
    # routing
    "KVRoute",
    # engine
    "AttentionGroup", "EngineOutput", "StreamEngine", "assert_shape_signature",
    # cache
    "CacheSignature", "CacheScope",
    # registry
    "StreamRegistry", "STREAMS", "StreamAdapter", "plan_from_legacy_mask",
]
