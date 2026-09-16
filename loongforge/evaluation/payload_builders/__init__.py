# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Per-model PayloadBuilder registry for the LoongForge eval side.

A PayloadBuilder converts the benchmark-neutral ``canonical dict`` produced by
an adapter into the ``kwargs`` a model's ``predict_action`` accepts. It is the
single collection point for model-specific eval payload assembly (previously
scattered across runners, factories, the policy, and adapters).
"""

from loongforge.evaluation.payload_builders.base import PayloadBuilder
from loongforge.evaluation.payload_builders.registry import (
    PAYLOAD_BUILDER_REGISTRY,
    build_payload_builder,
    register_payload_builder,
)

__all__ = [
    "PayloadBuilder",
    "PAYLOAD_BUILDER_REGISTRY",
    "build_payload_builder",
    "register_payload_builder",
]
