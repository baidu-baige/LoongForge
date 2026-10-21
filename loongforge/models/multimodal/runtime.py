# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Backend bindings a multimodal model receives instead of importing the engine.

The training method builds this object from the engine and passes it to
``omni_model_provider``; the model only calls the supplied callables.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class MultimodalRuntime:
    """Engine capabilities the multimodal model needs at run time."""

    #: Image-encoder data-parallel size decided by the engine's parallel layout.
    encoder_dp_size: int
    #: Number of model-parallel ranks that hold one decoder microbatch.
    model_size: int
    #: Switch the global parallel state to a named encoder/decoder component.
    parallel_state_switch: Callable[[str], None]
    #: DP-balanced vision encoder forward, or None when it is disabled.
    vit_balance: Optional[Callable[..., Any]] = None
    #: Factory building the fine-grained EP schedule plan, or None when unused.
    schedule_plan_factory: Optional[Callable[..., Any]] = None
