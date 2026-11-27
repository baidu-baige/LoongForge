# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""A point robot and a scripted LLM for tests that need no simulator or endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.embodied.agent.core.types import CameraFrame, EpisodeSpec, MoveResult, Obs
from loongforge.embodied.agent.planners.llm import LLMConfig, Usage

K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
# camera 1 m above the origin looking straight down: OpenCV x -> world x, y -> world -y, z -> world -z
CAM2WORLD = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 1.0], [0, 0, 0, 1.0]])


class FakeEmbodiment:
    """Moves exactly to each target; the task succeeds once the tool point is within 1 cm of ``goal``."""

    name = "libero"  # reuses LIBERO's prompt conventions
    cameras = ("agentview", "wrist")
    environment_notes = "A point robot above a flat floor at z = 0."

    def __init__(self, goal=(0.1, 0.0, 0.2), step_limit: int = 100, steps_per_move: int = 10) -> None:
        self.goal = np.array(goal)
        self.step_limit = step_limit
        self.steps_per_move = steps_per_move
        self.moves: List[Dict[str, Any]] = []

    def reset(self, spec: EpisodeSpec) -> Obs:
        self.pos = np.array([0.0, 0.0, 0.5])
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.gap = 0.08
        self.steps = 0
        return self.observe()

    def observe(self, *, depth: bool = False) -> Obs:
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[:, :32] = 255  # left half white, to check mirroring
        frame = CameraFrame(rgb, np.ones((64, 64), dtype=np.float32), K, CAM2WORLD) if depth else CameraFrame(rgb)
        return Obs(
            instruction="move to the left target",
            cameras={name: frame for name in self.cameras},
            eef_pos=self.pos.copy(),
            eef_quat=self.quat.copy(),
            gripper_gap=self.gap,
            success=bool(np.linalg.norm(self.pos - self.goal) < 0.01),
            steps=self.steps,
            step_limit=self.step_limit,
        )

    def move(self, pos, quat, opening: Optional[float], max_steps: Optional[int] = None) -> MoveResult:
        self.moves.append({"pos": np.array(pos), "quat": np.array(quat), "opening": opening})
        executed = min(self.steps_per_move, self.step_limit - self.steps)
        self.steps += executed
        self.pos, self.quat = np.array(pos, dtype=float), np.array(quat, dtype=float)
        if opening is not None:
            self.gap = 0.08 if opening > 0 else 0.0
        if self.observe().success:
            return MoveResult("success", executed)
        return MoveResult("reached" if executed == self.steps_per_move else "budget", executed)

    def base_pose(self):
        return np.array([-0.5, 0.0, 0.0]), np.eye(3)

    def close(self) -> None:
        pass


class ScriptedLLM:
    """Returns queued replies in order and records the messages it was sent."""

    def __init__(self, replies: List[Dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.sent: List[List[Dict[str, Any]]] = []
        self.usage = Usage()
        self.config = LLMConfig(model="scripted")

    def chat(self, messages, tools=None) -> Dict[str, Any]:
        self.sent.append(list(messages))
        self.usage.calls += 1
        self.usage.prompt_tokens += 10
        reply = self.replies.pop(0)
        return {"content": reply.get("content", ""), "reasoning": None, "tool_calls": reply.get("tool_calls", [])}
