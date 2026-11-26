# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks using a small redundant-branch planar arm fixture."""

import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .continuity import ContinuousProblem, ContinuityConfig, reconstruct_continuous_trajectory


def fixture(n=12, angular_step=0.01):
    bodies = ""
    for side, y in (("left", -1), ("right", 1)):
        bodies += f"""<body name="{side}_base" pos="0 {y} 0">
        <joint type="hinge" axis="0 0 1" range="-2.8 2.8"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size=".02" contype="0" conaffinity="0"/>
        <body name="{side}_link" pos="1 0 0">
        <joint type="hinge" axis="0 0 1" range="-2.8 2.8"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size=".02" contype="0" conaffinity="0"/>
        <site name="{side}_tcp" pos="1 0 0"/></body></body>"""
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><compiler angle="radian"/><worldbody>{bodies}</worldbody></mujoco>'
    )
    refs = [("site", mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s + "_tcp")) for s in ("left", "right")]
    # Alternating elbow branches have exactly the same endpoint position.
    q = np.tile([0.5, -1, 0.5, -1], (n, 1))
    q[::2] *= -1
    q[:, [0, 2]] += np.arange(n)[:, None] * angular_step
    data = mujoco.MjData(model)
    positions, rotations = [], []
    for t in range(n):
        data.qpos[:] = [0.5 + t * angular_step, -1, 0.5 + t * angular_step, -1]
        mujoco.mj_forward(model, data)
        positions.append([data.site_xpos[idx].copy() for _, idx in refs])
        rotations.append([data.site_xmat[idx].reshape(3, 3).copy() for _, idx in refs])
    return model, q, refs, np.array(positions), np.array(rotations)


class ContinuityTests(unittest.TestCase):
    def test_penetration_residual_keeps_collision_gradient(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
        <worldbody>
          <body name="left_link"><joint axis="0 0 1" range="-2 2"/>
            <geom type="sphere" size=".15" pos="1 0 0"/><site pos="1 0 0"/></body>
          <body name="right_link" pos="1.8 0 0"><joint axis="0 0 1" range="-2 2"/>
            <geom type="sphere" size=".15" pos="-1 0 0"/><site pos="-1 0 0"/></body>
        </worldbody></mujoco>""")
        q = np.array([0.1, -0.12])
        data = mujoco.MjData(model)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        problem = ContinuousProblem(
            model,
            [0, 1],
            [("site", 0), ("site", 1)],
            data.site_xpos[None].copy(),
            data.site_xmat.reshape(1, 2, 3, 3).copy(),
            np.ones((1, 2), bool),
            30,
        )
        problem.collision_residual_clearance = 0.0
        problem.frame(0, q)
        residual, jac = problem.evaluate(q)
        self.assertGreater(residual[16:].max(), 0.0)
        numeric = np.column_stack(
            [
                (problem.evaluate(q + np.eye(2)[i] * 1e-6)[0] - problem.evaluate(q - np.eye(2)[i] * 1e-6)[0]) / 2e-6
                for i in range(2)
            ]
        )
        np.testing.assert_allclose(jac, numeric, atol=1e-3, rtol=1e-4)
        self.assertFalse(problem.measure(q[None])[0]["accepted"])

    def test_branch_changes_removed_without_losing_moving_target(self):
        m, q, refs, p, R = fixture()
        original_margin = m.geom_margin.copy()
        result, report, _, _ = reconstruct_continuous_trajectory(
            m, q, np.arange(4), refs, p, R, np.ones((len(q), 2), bool), 30, ContinuityConfig(max_seeds=1)
        )
        self.assertTrue(report["accepted"])
        self.assertGreater(np.max(abs(np.diff(q, axis=0))), 1)
        self.assertLessEqual(np.max(abs(np.diff(result, axis=0))), 0.100001)
        self.assertLess(report["position_max_m"], 0.005)
        np.testing.assert_array_equal(original_margin, m.geom_margin)

    def test_fps_changes_feasibility_under_same_velocity_bound(self):
        m, q, refs, p, R = fixture(n=6, angular_step=0.12)
        for fps, expected in ((15, True), (60, False)):
            result, report, _, _ = reconstruct_continuous_trajectory(
                m, q, np.arange(4), refs, p, R, np.ones((len(q), 2), bool), fps, ContinuityConfig(max_seeds=1)
            )
            self.assertEqual(report["accepted"], expected)
            self.assertLessEqual(np.max(abs(np.diff(result, axis=0))), 3 / fps + 1e-6)

    def test_unreachable_target_rejected_and_model_restored(self):
        m, q, refs, p, R = fixture(n=6)
        margins = m.geom_margin.copy()
        p[:, :, 0] += 5
        _, report, _, _ = reconstruct_continuous_trajectory(
            m, q, np.arange(4), refs, p, R, np.ones((len(q), 2), bool), 30, ContinuityConfig(max_seeds=1, max_nfev=5)
        )
        self.assertFalse(report["accepted"])
        self.assertEqual(report["position_failed_frames"], len(q))
        np.testing.assert_array_equal(margins, m.geom_margin)

    def test_analytic_pose_jacobian(self):
        m, q, refs, p, R = fixture()
        R[:] = Rotation.from_rotvec([0.2, -0.15, 0.1]).as_matrix()
        problem = ContinuousProblem(m, np.arange(4), refs, p, R, np.ones((len(q), 2), bool), 30)
        problem.frame(0, q[0])
        _, jac = problem.evaluate(q[0])
        numeric = np.column_stack(
            [
                (problem.evaluate(q[0] + np.eye(4)[i] * 1e-6)[0] - problem.evaluate(q[0] - np.eye(4)[i] * 1e-6)[0])
                / 2e-6
                for i in range(4)
            ]
        )
        np.testing.assert_allclose(jac, numeric, atol=2e-5, rtol=2e-5)

    def test_invalid_targets_rejected(self):
        m, q, refs, p, R = fixture()
        p[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            ContinuousProblem(m, np.arange(4), refs, p, R, np.ones((len(q), 2), bool), 30)


if __name__ == "__main__":
    unittest.main()
