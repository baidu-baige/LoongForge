# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tool declaration, toolkit execution, geometry and perception."""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional

import numpy as np
import pytest
from pydantic import Field

from loongforge.embodied.agent.core import geometry, perception
from loongforge.embodied.agent.core.recorder import Recorder
from loongforge.embodied.agent.core.tool import collect_tools, tool
from loongforge.embodied.agent.core.toolkit import Toolkit
from loongforge.embodied.agent.core.types import CameraFrame, ToolResult
from loongforge.embodied.agent.unit_tests.fakes import CAM2WORLD, K


class Arm:
    @tool(advances=True)
    def move(
        self,
        xyz: Annotated[List[float], Field(min_length=3, max_length=3)],
        gripper: Optional[Literal["open", "close"]] = None,
    ) -> ToolResult:
        """Move somewhere.

        Args:
            xyz: Target in metres,
                world frame.
            gripper: Gripper command.
        """
        return ToolResult({"xyz": xyz, "gripper": gripper})

    @tool
    def broken(self) -> ToolResult:
        """Always fails."""
        raise RuntimeError("boom")


def test_schema_comes_from_annotations_and_docstring() -> None:
    move, broken = collect_tools(Arm())
    schema = move.openai_schema()["function"]
    assert schema["name"] == "move" and schema["description"] == "Move somewhere."
    assert schema["parameters"]["required"] == ["xyz"]
    assert schema["parameters"]["properties"]["xyz"]["description"] == "Target in metres, world frame."
    assert move.advances and not broken.advances


def test_toolkit_reports_errors_instead_of_raising(tmp_path) -> None:
    toolkit = Toolkit(collect_tools(Arm()), Recorder(tmp_path))
    assert toolkit.execute("move", {"xyz": [1, 2, 3]}).data == {"xyz": [1.0, 2.0, 3.0], "gripper": None}
    assert "bad arguments" in toolkit.execute("move", {"xyz": [1, 2]}).data["error"]
    assert "bad arguments" in toolkit.execute("move", {"xyz": ["1", 2, 3]}).data["error"]  # strict: no str -> float
    assert "bad arguments" in toolkit.execute("move", {"xyz": [1, 2, 3], "speed": 1}).data["error"]
    assert "unknown tool" in toolkit.execute("fly", {}).data["error"]
    assert "boom" in toolkit.execute("broken", {}).data["error"]
    assert len((tmp_path / "trace.jsonl").read_text().splitlines()) == 6


def test_duplicate_tool_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        Toolkit(collect_tools(Arm()) * 2)


@pytest.mark.parametrize("rpy", [(0.1, -0.4, 2.0), (np.pi, 0.0, 0.3), (0.0, 0.0, 0.0)])
def test_rotation_roundtrips(rpy) -> None:
    rot = geometry.rpy_to_matrix(rpy)
    assert np.allclose(geometry.rpy_to_matrix(geometry.matrix_to_rpy(rot)), rot)
    assert np.allclose(geometry.quat_to_matrix(geometry.matrix_to_quat(rot)), rot)


def test_rotation_error_is_world_axis_angle() -> None:
    q0 = geometry.matrix_to_quat(np.eye(3))
    q1 = geometry.matrix_to_quat(geometry.rpy_to_matrix((0.0, 0.0, 0.5)))
    assert np.allclose(geometry.rotation_error(q1, q0), [0.0, 0.0, 0.5])
    assert np.allclose(geometry.rotation_error(q0, q0), 0.0)


def test_pixel_world_roundtrip() -> None:
    depth = np.full((64, 64), 1.0, dtype=np.float32)
    depth[0, 0] = 0.0  # invalid
    frame = CameraFrame(np.zeros((64, 64, 3), np.uint8), depth, K, CAM2WORLD)
    center, offset, invalid, outside = perception.pixel_to_world(
        frame, [[32, 32], [32, 42], [0, 0], [99, 0]], neighborhood=0
    )
    assert np.allclose(center, [0.0, 0.0, 0.0]) and np.allclose(offset, [0.1, 0.0, 0.0])
    assert invalid is None and outside is None
    assert perception.world_to_pixel(frame, offset) == [32, 42]


def test_pixel_to_world_needs_depth() -> None:
    with pytest.raises(ValueError):
        perception.pixel_to_world(CameraFrame(np.zeros((4, 4, 3), np.uint8)), [[0, 0]])
