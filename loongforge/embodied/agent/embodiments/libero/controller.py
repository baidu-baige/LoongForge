# Copyright 2026 The LoongForge Authors.
# Copyright contributors to the PEAR project.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from PEAR (https://github.com/mit-han-lab/pear, commit 31fd1cf),
# pear/envs/libero/controller.py, licensed under the Apache License, Version 2.0.
# Changes: targets are given as a world pose instead of base-frame deltas, the
# robosuite 1.4 and 1.5 controller APIs are both supported, and the rotation
# helpers come from loongforge.embodied.agent.core.geometry.

"""Closed-loop motion for LIBERO's Panda: PD on the tool point's pose through robosuite's OSC."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from loongforge.embodied.agent.core.geometry import rotation_error


class MotionController:
    """Step a robosuite scene on native OSC delta actions until a target pose is reached at rest.

    The arm stops early when something keeps it from the target (no progress
    for a few steps); the gripper goes all the way open or closed, as LIBERO's
    own binary gripper does.
    """

    POS_DELTA_M = 0.05  # OSC position change at offset 1.0 (robosuite osc_pose output_max)
    ROT_DELTA_RAD = 0.5  # OSC orientation change at offset 1.0
    GRIP_RATE = 0.25  # gripper command change per control step
    GRIP_SPAN_CM = 8.0  # finger gap from closed to open
    LEAD_S = 0.04  # PD lead: velocity counts against the error as this many seconds of motion
    SPAN_M = 0.015  # position error at which an axis's offset saturates
    SPAN_RAD = 0.15  # orientation error at which an axis's offset saturates
    DECOUPLE_DAMPING = 0.1
    DECOUPLE_MAX = 0.5
    DECOUPLE_JOINT_MARGIN_RAD = 0.2
    DECOUPLE_ROT_ERR_RAD = 0.1
    POS_TOL_M = 0.003
    ROT_TOL_RAD = 0.02
    REST_MPS = 0.03
    REST_RADPS = 0.15
    GRIP_STILL_M = 0.0005
    GRIP_TOL_CM = 0.5
    PROGRESS_M = 0.001
    PROGRESS_RAD = 0.005
    BLOCKED_STEP_N = 2
    STALL_STEP_N = 6
    MAX_ACTION_STEP_N = 100  # native actions one motion may take

    def __init__(
        self,
        scene: Any,
        observation: Dict[str, np.ndarray],
        pos_target: np.ndarray,
        quat_target: np.ndarray,
        opening: Optional[float],
    ) -> None:
        self.scene = scene
        robot = scene.robots[0]
        self.robot = robot
        self.ctrl = getattr(robot, "controller", None) or robot.part_controllers["right"]
        self.gripper = robot.gripper["right"] if isinstance(robot.gripper, dict) else robot.gripper
        self.site = getattr(self.ctrl, "eef_name", None) or self.ctrl.ref_name

        self.pos_target = np.asarray(pos_target, dtype=np.float64)
        self.quat_target = np.asarray(quat_target, dtype=np.float64)
        self.opening_target = None if opening is None else float(np.sign(opening))
        self.width_target = (
            None if self.opening_target is None else self.GRIP_SPAN_CM * (self.opening_target + 1.0) / 2.0
        )

        self.observation = observation
        self.pos, self.quat, self.gap = self.tool_state(observation)
        self.vel, self.ang_vel = self._velocity()
        self.least_pos_err = self.least_rot_err = np.inf
        self.stall_step_n = self.slow_stall_step_n = 0
        self.is_stopping = False

    def move(self, budget: int) -> Tuple[Dict[str, np.ndarray], List[List[float]], bool, str]:
        """Run until rest, stall, task success or budget: (last obs, native actions, success, outcome)."""
        actions: List[List[float]] = []
        for _ in range(min(self.MAX_ACTION_STEP_N, budget)):
            native = np.concatenate([self._arm_offsets(), [self._grip_direction()]])
            self.observation, _, done, _ = self.scene.step(native)
            actions.append(native.tolist())
            if done:
                return self.observation, actions, True, "success"
            outcome = self._step_outcome()
            if outcome is not None:
                return self.observation, actions, False, outcome
        return self.observation, actions, False, "budget" if len(actions) >= budget else "timeout"

    @staticmethod
    def tool_state(observation: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
        """World position, (x, y, z, w) orientation and finger gap (m) of the tool point."""
        qpos = observation["robot0_gripper_qpos"]
        return (
            np.array(observation["robot0_eef_pos"], dtype=np.float64),
            np.array(observation["robot0_eef_quat"], dtype=np.float64),
            float(qpos[0] - qpos[1]),
        )

    def _arm_offsets(self) -> np.ndarray:
        rot_err = rotation_error(self.quat_target, self.quat)
        twist = np.concatenate([self.vel, self.ang_vel])
        span = np.array([self.SPAN_M] * 3 + [self.SPAN_RAD] * 3)
        offsets = np.clip(
            (np.concatenate([self.pos_target - self.pos, rot_err]) - self.LEAD_S * twist) / span, -1.0, 1.0
        )
        if np.linalg.norm(rot_err) > self.DECOUPLE_ROT_ERR_RAD or self._joint_margin() < self.DECOUPLE_JOINT_MARGIN_RAD:
            return offsets
        return self._decouple(offsets, twist)

    def _joint_margin(self) -> float:
        sim = self.scene.sim
        positions = sim.data.qpos[self.robot._ref_joint_pos_indexes]
        lower, upper = sim.model.jnt_range[self.robot._ref_joint_indexes].T
        return float(np.min(np.minimum(positions - lower, upper - positions)))

    def _decouple(self, offsets: np.ndarray, twist: np.ndarray) -> np.ndarray:
        """Offsets that give the uncoupled OSC a decoupled response (PEAR's derivation)."""
        if not getattr(self.ctrl, "uncoupling", False):
            return offsets
        from robosuite.utils.control_utils import opspace_matrices

        ctrl = self.ctrl
        lambda_full, lambda_pos, lambda_ori, _ = opspace_matrices(ctrl.mass_matrix, ctrl.J_full, ctrl.J_pos, ctrl.J_ori)
        block = np.zeros((6, 6))
        block[:3, :3], block[3:, 3:] = lambda_pos, lambda_ori
        scale = np.array([self.POS_DELTA_M] * 3 + [self.ROT_DELTA_RAD] * 3)
        coupling = np.linalg.solve(lambda_full, block) * (scale[None, :] / scale[:, None])
        inverse = np.linalg.solve(coupling.T @ coupling + self.DECOUPLE_DAMPING**2 * np.eye(6), coupling.T)
        decoupled = inverse @ offsets + float(ctrl.kd[0] / ctrl.kp[0]) * (np.eye(6) - inverse) @ (twist / scale)
        correction = np.clip(decoupled - offsets, -self.DECOUPLE_MAX, self.DECOUPLE_MAX)
        return np.clip(offsets + correction, -1.0, 1.0)

    def _opening(self) -> float:
        return float(np.atleast_1d(self.gripper.current_action)[0])  # -1 closed .. 1 open

    def _grip_direction(self) -> float:
        if self.opening_target is None:
            return 0.0
        opening = self._opening()
        if opening > self.opening_target + self.GRIP_RATE / 2:
            return 1.0  # native +1 closes
        if opening < self.opening_target - self.GRIP_RATE / 2:
            return -1.0
        return 0.0

    def _step_outcome(self) -> Optional[str]:
        previous_gap = self.gap
        self.pos, self.quat, self.gap = self.tool_state(self.observation)
        self.vel, self.ang_vel = self._velocity()

        pos_err = float(np.linalg.norm(self.pos_target - self.pos))
        rot_err = float(np.linalg.norm(rotation_error(self.quat_target, self.quat)))
        is_slow = np.linalg.norm(self.vel) < self.REST_MPS and np.linalg.norm(self.ang_vel) < self.REST_RADPS
        fingers_done = (
            self.width_target is None
            or abs(self.gap - previous_gap) < self.GRIP_STILL_M
            or abs(self.gap * 100.0 - self.width_target) < self.GRIP_TOL_CM
        )
        opening_reached = self.opening_target is None or abs(self._opening() - self.opening_target) < self.GRIP_RATE / 2
        is_at_rest = is_slow and fingers_done and opening_reached
        arm_reached = pos_err < self.POS_TOL_M and rot_err < self.ROT_TOL_RAD

        if is_at_rest and (arm_reached or self.is_stopping):
            return "stopped" if self.is_stopping else "reached"
        if self.is_stopping:
            return None

        if (
            arm_reached
            or pos_err < self.least_pos_err - self.PROGRESS_M
            or rot_err < self.least_rot_err - self.PROGRESS_RAD
        ):
            self.least_pos_err, self.least_rot_err = min(self.least_pos_err, pos_err), min(self.least_rot_err, rot_err)
            self.stall_step_n = self.slow_stall_step_n = 0
        else:
            self.stall_step_n += 1
            self.slow_stall_step_n = self.slow_stall_step_n + 1 if is_slow else 0

        if not arm_reached and (
            self.slow_stall_step_n >= self.BLOCKED_STEP_N or self.stall_step_n >= self.STALL_STEP_N
        ):
            self.is_stopping = True  # stop pushing: come to rest where it is
            self.pos_target, self.quat_target = self.pos.copy(), self.quat.copy()
            if is_at_rest:
                return "stopped"
        return None

    def _velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        data = self.scene.sim.data
        return (
            np.array(data.get_site_xvelp(self.site), dtype=np.float64),
            np.array(data.get_site_xvelr(self.site), dtype=np.float64),
        )
