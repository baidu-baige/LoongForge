# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO embodiment: calibration and closed-loop motion. Skipped without LIBERO installed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libero")

from loongforge.embodied.agent.core.geometry import matrix_to_quat, quat_to_matrix, rpy_to_matrix
from loongforge.embodied.agent.core.perception import pixel_to_world, world_to_pixel
from loongforge.embodied.agent.core.types import EpisodeSpec
from loongforge.embodied.agent.embodiments.libero.env import LiberoEmbodiment


@pytest.fixture(scope="module")
def robot():
    embodiment = LiberoEmbodiment(resolution=256)
    yield embodiment
    embodiment.close()


def test_reset_follows_the_eval_protocol(robot) -> None:
    obs = robot.reset(EpisodeSpec("libero_spatial", 0, 0))
    assert obs.instruction.startswith("pick up the black bowl")
    assert (obs.steps, obs.step_limit, obs.success) == (0, 220, False)
    assert obs.cameras["agentview"].rgb.shape == (256, 256, 3) and obs.cameras["agentview"].depth is None


def test_back_projection_lands_on_the_plate(robot) -> None:
    robot.reset(EpisodeSpec("libero_spatial", 0, 0))
    sim = robot._scene.sim
    plate = sim.data.body_xpos[sim.model.body_name2id("plate_1_main")].copy()
    frame = robot.observe(depth=True).cameras["agentview"]
    pixel = world_to_pixel(frame, plate)
    assert pixel[1] > 128  # +y is to the right in the upright agentview image
    surface = pixel_to_world(frame, [pixel])[0]
    assert np.linalg.norm(np.array(surface) - plate) < 0.02


def test_controller_reaches_position_and_orientation_targets(robot) -> None:
    obs = robot.reset(EpisodeSpec("libero_spatial", 0, 0))
    target = obs.eef_pos + np.array([0.08, 0.05, -0.10])
    quat = matrix_to_quat(rpy_to_matrix((0.0, 0.0, 0.4)) @ quat_to_matrix(obs.eef_quat))
    result = robot.move(target, quat, None)
    after = robot.observe()
    assert result.outcome == "reached" and 0 < result.executed <= 100
    assert np.linalg.norm(after.eef_pos - target) < 0.005
    assert after.steps == result.executed


def test_gripper_closes_and_opens(robot) -> None:
    obs = robot.reset(EpisodeSpec("libero_spatial", 0, 0))
    robot.move(obs.eef_pos, obs.eef_quat, -1.0)
    assert robot.observe().gripper_gap < 0.01
    robot.move(obs.eef_pos, obs.eef_quat, 1.0)
    assert robot.observe().gripper_gap > 0.07
