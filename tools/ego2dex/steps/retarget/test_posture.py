# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
import mujoco
from .test_continuity import fixture
from .posture import TrajectoryProblem, refine_posture, refine_recovery_stages


class PostureTests(unittest.TestCase):
    def test_full_trajectory_jacobian_including_temporal_edges(self):
        m, q, refs, p, R = fixture(n=5)
        objective = TrajectoryProblem(m, q, np.arange(4), refs, p, R, np.ones((5, 2), bool), 30)
        x = q.ravel() + 0.02
        _, jac = objective.evaluate(x)
        numeric = np.column_stack(
            [
                (
                    objective.evaluate(x + np.eye(len(x))[i] * 1e-6)[0]
                    - objective.evaluate(x - np.eye(len(x))[i] * 1e-6)[0]
                )
                / 2e-6
                for i in range(len(x))
            ]
        )
        np.testing.assert_allclose(jac.toarray(), numeric, atol=3e-5, rtol=3e-5)

    def test_perfect_static_motion_preserved_and_margins_restored(self):
        m, q, refs, p, R = fixture(n=5, angular_step=0)
        q[:] = [0.5, -1, 0.5, -1]
        margins = m.geom_margin.copy()
        result, report, _, _ = refine_posture(m, q, np.arange(4), refs, p, R, np.ones((5, 2), bool), 30, max_nfev=3)
        self.assertTrue(report["after"]["accepted"])
        np.testing.assert_allclose(result, q, atol=1e-8)
        np.testing.assert_array_equal(margins, m.geom_margin)

    def test_midpoint_collision_jacobian(self):
        m = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
        <worldbody>
          <body name="left_link"><joint axis="0 0 1" range="-2 2"/>
            <geom type="sphere" size=".15" pos="1 0 0"/><site pos="1 0 0"/></body>
          <body name="right_link" pos="1.8 0 0"><joint axis="0 0 1" range="-2 2"/>
            <geom type="sphere" size=".15" pos="-1 0 0"/><site pos="-1 0 0"/></body>
        </worldbody></mujoco>""")
        q = np.tile([0.1, -0.12], (5, 1)) + np.arange(5)[:, None] * 0.005
        data = mujoco.MjData(m)
        data.qpos[:] = q[0]
        mujoco.mj_forward(m, data)
        p = np.tile(data.site_xpos[None], (5, 1, 1))
        R = np.tile(data.site_xmat.reshape(1, 2, 3, 3), (5, 1, 1, 1))
        objective = TrajectoryProblem(
            m, q, np.arange(2), [("site", 0), ("site", 1)], p, R, np.ones((5, 2), bool), 30, midpoint_collisions=True
        )
        x = q.ravel() + 0.02
        r, j = objective.evaluate(x)
        start = len(r) - (len(q) - 1) * len(objective.problem.pairs)
        self.assertTrue(np.any(r[start:] > 0), "Fixture must exercise active midpoint contacts")
        numeric = np.column_stack(
            [
                (
                    objective.evaluate(x + np.eye(len(x))[i] * 1e-6)[0][start:]
                    - objective.evaluate(x - np.eye(len(x))[i] * 1e-6)[0][start:]
                )
                / 2e-6
                for i in range(len(x))
            ]
        )
        np.testing.assert_allclose(j.toarray()[start:], numeric, atol=3e-5, rtol=3e-5)

    def test_invalid_initial_motion_rejected(self):
        m, q, refs, p, R = fixture(n=5)
        with self.assertRaises(ValueError):
            refine_posture(m, q, np.arange(4), refs, p, R, np.ones((5, 2), bool), 30)

    def test_local_target_error_can_be_recovered_without_relaxing_guards(self):
        m, q, refs, p, R = fixture(n=9, angular_step=0.01)
        q[:] = [0.5, -1, 0.5, -1]
        q[:, [0, 2]] += np.arange(9)[:, None] * 0.01
        q[4, 0] += 0.025
        margins = m.geom_margin.copy()
        _, report, _, _ = refine_posture(
            m, q, np.arange(4), refs, p, R, np.ones((9, 2), bool), 30, max_nfev=60, allow_infeasible=True
        )
        self.assertFalse(report["before"]["accepted"])
        self.assertTrue(report["after"]["accepted"])
        self.assertLessEqual(report["after"]["position_max_m"], 0.005)
        self.assertLessEqual(report["after"]["max_velocity_rad_s"], 3.00001)
        np.testing.assert_array_equal(margins, m.geom_margin)

    def test_recovery_does_not_accept_unreachable_targets(self):
        m, q, refs, p, R = fixture(n=5, angular_step=0)
        q[:] = [0.5, -1, 0.5, -1]
        p[:, :, 0] += 5
        _, report, _, _ = refine_posture(
            m, q, np.arange(4), refs, p, R, np.ones((5, 2), bool), 30, max_nfev=5, allow_infeasible=True
        )
        self.assertFalse(report["after"]["accepted"])

    def test_staged_recovery_and_margin_restoration(self):
        m, q, refs, p, R = fixture(n=9, angular_step=0.01)
        q[:] = [0.5, -1, 0.5, -1]
        q[:, [0, 2]] += np.arange(9)[:, None] * 0.01
        q[4, 0] += 0.025
        margins = m.geom_margin.copy()
        seen = []
        _, report, _, _ = refine_recovery_stages(
            m,
            q,
            np.arange(4),
            refs,
            p,
            R,
            np.ones((9, 2), bool),
            30,
            max_nfev=60,
            candidate_callback=lambda q, r: seen.append(r),
        )
        self.assertTrue(any(r["accepted"] for r in seen))
        self.assertEqual(len(seen), 4)
        self.assertEqual(report["total_nfev"], report["nfev"] + report["frame_phase"]["nfev"])
        np.testing.assert_array_equal(margins, m.geom_margin)


if __name__ == "__main__":
    unittest.main()
