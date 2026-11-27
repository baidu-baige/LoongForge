# Copyright 2026 The LoongForge Authors.
# Copyright contributors to the PEAR project.
# SPDX-License-Identifier: Apache-2.0
#
# The protocol text, state schema and LIBERO image/instruction conventions are
# taken from PEAR (https://github.com/mit-han-lab/pear, commit 31fd1cf):
# pear/envs/system_prompt.md, pear/envs/tool.py, pear/envs/libero/env.py,
# pear/envs/libero/environment_prompt.md and pear/envs/libero/controller.py,
# licensed under the Apache License, Version 2.0.

"""Incremental pose direct drive (PEAR style).

Each turn the model reads the arm's pose in the base frame (cm, radians), the
finger gap and two RGB frames, and answers with one fenced JSON block that
changes every quantity by some amount. No depth, calibration, perception tools
or ``finish``: the episode ends on task success, the step budget, or an answer
without a valid action.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.embodied.agent.core.embodiment import Embodiment
from loongforge.embodied.agent.core.geometry import matrix_to_quat, matrix_to_rpy, quat_to_matrix, rpy_to_matrix
from loongforge.embodied.agent.core.types import MoveResult, Obs
from loongforge.embodied.agent.planners.llm import image_part, text_part

SCHEMA = {"arm": ("x", "y", "z", "roll", "pitch", "yaw"), "gripper": ("width",)}
DECIMALS = {"x": 1, "y": 1, "z": 1, "roll": 2, "pitch": 2, "yaw": 2, "width": 1}
JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
REMINDER = "The current camera frames follow. End your response with one json block containing the next action."

SYSTEM_PROMPT = """You are controlling a robot. Each turn you see the robot's current state and its camera images, and you respond with the next action.

## OBSERVATION

The robot is made of named parts, and each part has named quantities you can set. Every turn begins with the current value of every such quantity, grouped by part, followed by the camera images:

{"<part>": {"<quantity>": <value>, ...}, ...}

Lengths are in centimeters and angles in radians; any other quantity is a plain number. Nothing beyond these names is given about what each quantity corresponds to.

## ACTION

The action changes the listed quantities by these amounts:

{"<part>": {"<quantity>": <change>, ...}, ...}

Write the change of every quantity; 0 leaves a quantity as it is.

## EXECUTION

The action ends once every target is reached and the robot is at rest. If something keeps a target from being reached, the robot stops there and the action ends. Either way, you then receive the next observation.

## RESPONSE FORMAT

End your response with exactly one fenced json block containing the action, and nothing after it:

```json
<action>
```

{environment}## TASK

{instruction}"""

# PEAR shows LIBERO frames rotated by 180 degrees (as LIBERO VLA evals do), which mirrors the
# scene left to right, and swaps "left"/"right" in the instruction to match.
LIBERO_ENVIRONMENT = """- agentview: a fixed camera; the robot's base is at the top of its image, +x points toward the bottom of the image and +y toward its left.
- wrist: a camera on the gripper; with the gripper pointing straight down at yaw 0, +x points toward the top of its image and +y toward its right.
- the episode ends by itself as soon as the task is done: while observations keep coming, the task is not done yet."""
_MIRRORED_SIDE = {"left": "right", "right": "left"}


class InvalidAction(ValueError):
    """The answer has no JSON block, or the block is not an action of the schema."""


def parse_action(text: str) -> Dict[str, Dict[str, float]]:
    """The last fenced JSON block of ``text``, checked against :data:`SCHEMA`."""
    blocks = JSON_BLOCK.findall(text or "")
    if not blocks:
        raise InvalidAction("no fenced json block")
    try:
        action = json.loads(blocks[-1])
    except json.JSONDecodeError as exc:
        raise InvalidAction(f"json block does not parse: {exc}") from exc
    if not isinstance(action, dict) or set(action) != set(SCHEMA):
        raise InvalidAction(f"the json block must map exactly the parts {', '.join(SCHEMA)} to their quantities")
    for part, quantities in action.items():
        if not isinstance(quantities, dict) or set(quantities) != set(SCHEMA[part]):
            raise InvalidAction(f"{part!r} must give exactly its quantities {', '.join(SCHEMA[part])}")
        for name, value in quantities.items():
            if type(value) not in (int, float) or not abs(value) <= sys.float_info.max:
                raise InvalidAction(f"{part}.{name} must be a finite number")
    return action


class PoseDeltaSession:
    """The PEAR protocol on one embodiment episode."""

    def __init__(self, embodiment: Embodiment) -> None:
        if embodiment.name != "libero":
            raise ValueError("pose_delta currently has image conventions for LIBERO only")
        self.embodiment = embodiment

    def system_prompt(self, obs: Obs) -> str:
        """PEAR's protocol with LIBERO's environment facts and the mirrored instruction."""
        instruction = re.sub(r"\b(left|right)\b", lambda m: _MIRRORED_SIDE[m.group(1)], obs.instruction)
        environment = f"## ENVIRONMENT\n\n{LIBERO_ENVIRONMENT}\n\n"
        return SYSTEM_PROMPT.replace("{environment}", environment).replace("{instruction}", instruction)

    def state(self, obs: Obs) -> Dict[str, Dict[str, float]]:
        """Base-frame quantities as the model reads them."""
        values = self._measured(obs)
        return {
            part: {name: round(float(values[name]), DECIMALS[name]) + 0.0 for name in names}
            for part, names in SCHEMA.items()
        }

    def turn(self, obs: Obs, frame_names: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """One user turn: state JSON and reminder, then a label and frame per camera."""
        parts = [text_part(json.dumps(self.state(obs)) + "\n\n" + REMINDER)]
        for camera, rgb in self.frames(obs).items():
            parts.append(text_part(f"[{camera}]"))
            parts.append(image_part(rgb, (frame_names or {}).get(camera)))
        return parts

    @staticmethod
    def frames(obs: Obs) -> Dict[str, np.ndarray]:
        """Frames as PEAR shows them: upright images rotated by 180 degrees, i.e. mirrored left to right."""
        return {camera: np.ascontiguousarray(frame.rgb[:, ::-1]) for camera, frame in obs.cameras.items()}

    def act(self, obs: Obs, action: Dict[str, Dict[str, float]]) -> MoveResult:
        """Change each quantity by the action's amount and move there."""
        targets = self._measured(obs)
        for name, change in action["arm"].items():
            targets[name] += float(change)
        base_pos, base_rot = self.embodiment.base_pose()
        pos = base_pos + base_rot @ (np.array([targets["x"], targets["y"], targets["z"]]) / 100.0)
        quat = matrix_to_quat(base_rot @ rpy_to_matrix([targets["roll"], targets["pitch"], targets["yaw"]]))
        width_change = float(action["gripper"]["width"])
        opening = None if width_change == 0.0 else float(np.sign(width_change))
        return self.embodiment.move(pos, quat, opening)

    def _measured(self, obs: Obs) -> Dict[str, float]:
        base_pos, base_rot = self.embodiment.base_pose()
        pos = base_rot.T @ (obs.eef_pos - base_pos) * 100.0
        rpy = matrix_to_rpy(base_rot.T @ quat_to_matrix(obs.eef_quat))
        values = dict(zip(SCHEMA["arm"], [*pos, *rpy]))
        values["width"] = obs.gripper_gap * 100.0
        return values
