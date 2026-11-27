# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The interface a robot or benchmark implements to be driven by the agent."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, Tuple

import numpy as np

from loongforge.embodied.agent.core.types import EpisodeSpec, MoveResult, Obs


class Embodiment(Protocol):
    """A single-arm robot in a scene, reset to benchmark episodes.

    Toolsets only see this interface: hidden simulator state (object poses,
    reward internals) stays inside the implementation. ``Obs.success`` is the
    benchmark's own verdict, which ends the episode.
    """

    name: str
    cameras: Sequence[str]

    def reset(self, spec: EpisodeSpec) -> Obs:
        """Reset to ``spec``'s initial state, settle the scene and return the first observation."""
        ...

    def observe(self, *, depth: bool = False) -> Obs:
        """Render the current observation; ``depth`` adds metric depth and calibration per camera."""
        ...

    def move(
        self, pos: np.ndarray, quat: np.ndarray, opening: Optional[float], max_steps: Optional[int] = None
    ) -> MoveResult:
        """Drive the tool point to a world pose in closed loop until it rests there or stops short.

        ``pos`` is in metres and ``quat`` is (x, y, z, w), both in the world frame.
        ``opening`` drives the gripper all the way open (+1) or closed (-1); None holds it.
        """
        ...

    def base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """The robot base's world position and 3x3 rotation."""
        ...

    def close(self) -> None:
        """Release simulator resources."""
        ...
