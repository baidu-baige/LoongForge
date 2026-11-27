# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Absolute pose direct drive through tool calls (RPent-style primitives without a VLA).

Motion tools take world-frame targets and run the embodiment's closed-loop
controller; read-only tools turn pixels of the latest frames into world points
with depth and calibration. After every motion the planner shows the model the
new frames.
"""

from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional

import numpy as np
from pydantic import Field

from loongforge.embodied.agent.core import perception
from loongforge.embodied.agent.core.embodiment import Embodiment
from loongforge.embodied.agent.core.geometry import matrix_to_quat, matrix_to_rpy, quat_to_matrix, rpy_to_matrix
from loongforge.embodied.agent.core.tool import collect_tools, tool
from loongforge.embodied.agent.core.types import MoveResult, Obs, ToolResult

Vec3 = Annotated[List[float], Field(min_length=3, max_length=3)]
Pixel = Annotated[List[int], Field(min_length=2, max_length=2)]

SYSTEM_PROMPT = """You control a robot arm through tools to complete a manipulation task.

## ENVIRONMENT

{environment}

## HOW TO ACT

- Each motion tool moves the tool point to a target pose in closed loop and returns once the robot rests there or stops short (outcome "reached" or "stopped"). Then you receive the new camera frames.
- Poses are world-frame positions in metres and fixed-axis roll, pitch, yaw in radians. Omit rpy to keep the current orientation.
- To locate something, pick pixels on it in the latest frame of a camera and call pixel_to_world. It returns points on the visible surface, not object centres.
- The gripper goes all the way open or closed. Check in the frames that an object moves with the hand before carrying it.
- Each episode has a budget of native control steps; every motion consumes some. The benchmark judges success and ends the episode when the task is done.
- Call finish when you believe the task is done or cannot be done.

## TASK

{instruction}"""


def pose_summary(obs: Obs) -> Dict[str, object]:
    """The robot state as the model reads it."""
    return {
        "eef_xyz": [round(float(v), 4) for v in obs.eef_pos],
        "eef_rpy": [round(float(v), 3) for v in matrix_to_rpy(quat_to_matrix(obs.eef_quat))],
        "gripper_gap_cm": round(obs.gripper_gap * 100.0, 1),
        "steps_used": obs.steps,
        "steps_left": obs.remaining_steps,
        "task_success": obs.success,
    }


class PoseToolset:
    """Tools of one episode; ``obs`` always holds the latest observation, with depth."""

    def __init__(self, embodiment: Embodiment, obs: Obs) -> None:
        self.embodiment = embodiment
        self.obs = obs
        self.finished: Optional[str] = None

    def system_prompt(self) -> str:
        """Prompt with the embodiment's facts and the task instruction."""
        notes = getattr(self.embodiment, "environment_notes", "")
        return SYSTEM_PROMPT.replace("{environment}", notes).replace("{instruction}", self.obs.instruction)

    def tools(self):
        """The tools, bound to this episode."""
        return collect_tools(self)

    def _after_motion(self, result: MoveResult) -> ToolResult:
        self.obs = self.embodiment.observe(depth=True)
        return ToolResult({"outcome": result.outcome, "executed_steps": result.executed, **pose_summary(self.obs)})

    @tool(advances=True)
    def move_to(
        self,
        xyz: Vec3,
        rpy: Optional[Vec3] = None,
        gripper: Optional[Literal["open", "close"]] = None,
    ) -> ToolResult:
        """Move the tool point to a world-frame pose.

        Args:
            xyz: Target position [x, y, z] in metres.
            rpy: Target [roll, pitch, yaw] in radians; omit to keep the current orientation.
            gripper: Open or close the gripper during the motion; omit to keep it as it is.
        """
        quat = self.obs.eef_quat if rpy is None else matrix_to_quat(rpy_to_matrix(rpy))
        opening = None if gripper is None else (1.0 if gripper == "open" else -1.0)
        return self._after_motion(self.embodiment.move(np.asarray(xyz, dtype=np.float64), quat, opening))

    @tool(advances=True)
    def set_gripper(self, state: Literal["open", "close"]) -> ToolResult:
        """Open or close the gripper fully while holding the arm still.

        Args:
            state: "open" or "close".
        """
        opening = 1.0 if state == "open" else -1.0
        return self._after_motion(self.embodiment.move(self.obs.eef_pos, self.obs.eef_quat, opening))

    @tool
    def pixel_to_world(self, camera: str, pixels: List[Pixel]) -> ToolResult:
        """World points of [row, col] pixels in the latest frame of a camera.

        Args:
            camera: Camera name, as labelled on the frames.
            pixels: Up to 16 pixels as [row, col]; row 0 is the top of the image.
        """
        if camera not in self.obs.cameras:
            return ToolResult({"error": f"unknown camera {camera!r}", "cameras": list(self.obs.cameras)})
        if len(pixels) > 16:
            return ToolResult({"error": "at most 16 pixels per call"})
        points = perception.pixel_to_world(self.obs.cameras[camera], pixels)
        return ToolResult({"points": [{"pixel": p, "xyz": xyz} for p, xyz in zip(pixels, points)]})

    @tool
    def finish(self, summary: str) -> ToolResult:
        """End the episode. The benchmark, not this call, decides success.

        Args:
            summary: What was done and what was observed.
        """
        self.finished = summary
        return ToolResult({"finished": True, "task_success": self.obs.success})
