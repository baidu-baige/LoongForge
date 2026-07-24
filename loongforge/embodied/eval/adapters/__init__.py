# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from .base import BaseBenchmarkAdapter
from .calvin import CALVIN_MAX_STEPS_PER_SUBTASK, CalvinAdapter
from .libero import LIBERO_DUMMY_ACTION, LiberoAdapter

__all__ = [
    "BaseBenchmarkAdapter",
    "CALVIN_MAX_STEPS_PER_SUBTASK",
    "CalvinAdapter",
    "LIBERO_DUMMY_ACTION",
    "LiberoAdapter",
    "SimplerEnvAdapter",
    "RoboTwinAdapter",
    "ManiSkillAdapter",
]


def __getattr__(name: str):
    """Run __getattr__."""
    if name == "SimplerEnvAdapter":
        from .simplerenv import SimplerEnvAdapter

        return SimplerEnvAdapter
    if name == "RoboTwinAdapter":
        from .robotwin import RoboTwinAdapter

        return RoboTwinAdapter
    if name == "ManiSkillAdapter":
        from .maniskill import ManiSkillAdapter

        return ManiSkillAdapter
    raise AttributeError(name)
