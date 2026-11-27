# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Data shared by embodiments, toolsets and planners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class CameraFrame:
    """One camera's view, upright (not the simulator's raw buffer orientation).

    ``intrinsic`` and ``cam2world`` follow the OpenCV convention for this upright
    image: pixel ``(row, col)`` with depth ``z`` lies at
    ``cam2world @ [(col - cx) z / fx, (row - cy) z / fy, z, 1]``.
    """

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None  # (H, W) float32, metres along the optical axis
    intrinsic: Optional[np.ndarray] = None  # (3, 3)
    cam2world: Optional[np.ndarray] = None  # (4, 4)


@dataclass
class Obs:
    """What the agent may read about the robot and the scene at one moment."""

    instruction: str
    cameras: Dict[str, CameraFrame]
    eef_pos: np.ndarray  # world frame, metres
    eef_quat: np.ndarray  # world frame, (x, y, z, w)
    gripper_gap: float  # distance between the fingers, metres
    success: bool  # the benchmark's own success check
    steps: int  # native actions used so far
    step_limit: int  # native actions the episode may use

    @property
    def remaining_steps(self) -> int:
        """Native actions left in the episode budget."""
        return max(0, self.step_limit - self.steps)


@dataclass
class MoveResult:
    """How one closed-loop motion ended."""

    outcome: str  # reached | stopped | success | budget | timeout
    executed: int  # native actions it took


@dataclass
class EpisodeSpec:
    """Which scene an episode runs: a benchmark suite, a task and an initial state."""

    suite: str
    task: int
    init_idx: int

    @property
    def tag(self) -> str:
        """Directory-safe name of the episode."""
        return f"{self.suite}/task{self.task}/init{self.init_idx}"


@dataclass
class ToolResult:
    """What a tool returns to the planner: JSON-serializable data plus labelled images."""

    data: Dict[str, Any] = field(default_factory=dict)
    images: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """Whether the call failed; the model reads the error instead of an exception."""
        return "error" in self.data


@dataclass
class EpisodeResult:
    """Outcome of one episode, written to result.json."""

    spec: EpisodeSpec
    success: bool
    end_reason: str  # success | finish | budget | max_turns | parse | llm_error | error
    steps: int
    llm_turns: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Flat JSON form."""
        return {
            "suite": self.spec.suite,
            "task": self.spec.task,
            "init_idx": self.spec.init_idx,
            "success": self.success,
            "end_reason": self.end_reason,
            "steps": self.steps,
            "llm_turns": self.llm_turns,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 2),
            "error": self.error,
            **self.extra,
        }
