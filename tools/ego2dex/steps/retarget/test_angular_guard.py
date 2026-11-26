# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .test_continuity import fixture
from .posture import TrajectoryProblem


class AngularGuardTests(unittest.TestCase):
    def problem(self):
        m, q, refs, p, R = fixture(n=5)
        baseline = {
            key: [0.0, 0.0] for key in ("symmetry_orientation_mean_deg", "approach_mean_deg", "opening_line_mean_deg")
        }
        baseline["symmetry_orientation_p95_deg"] = 0.0
        tcp = Rotation.from_rotvec([[0.3, 0.2, 0.1], [-0.2, 0.1, 0.4]]).as_matrix()
        objective = TrajectoryProblem(
            m,
            q,
            np.arange(4),
            refs,
            p,
            R,
            np.ones((5, 2), bool),
            30,
            angular_guard=dict(targets=R, tcp=tcp, baseline=baseline),
        )
        return objective, q

    def test_analytic_gradient_and_quantile(self):
        obj, q = self.problem()
        x = q.ravel() + np.linspace(0.07, 0.23, q.size)
        r, j = obj.evaluate(x)
        numeric = np.column_stack(
            [
                (
                    obj.evaluate(x + np.eye(len(x))[i] * 1e-6)[0][-7:]
                    - obj.evaluate(x - np.eye(len(x))[i] * 1e-6)[0][-7:]
                )
                / 2e-6
                for i in range(len(x))
            ]
        )
        np.testing.assert_allclose(j.toarray()[-7:], numeric, atol=2e-4, rtol=2e-5)
        self.assertTrue(np.all(r[-7:] > 0))

    def test_parallel_jaw_symmetry(self):
        obj, q = self.problem()
        guard = obj.angular_guard
        obj.problem.frame(0, q[0])
        obj.problem.evaluate(q[0])
        a, b = guard.frame(0)
        # A half turn about TCP approach leaves all three accepted metrics unchanged.
        for s in range(2):
            tcp = guard.tcp[s]
            guard.targets[:, s] = guard.targets[:, s] @ tcp @ np.diag([-1.0, -1.0, 1.0]) @ tcp.T
        c, d = guard.frame(0)
        np.testing.assert_allclose(a, c, atol=1e-7)
        np.testing.assert_allclose(b, d, atol=1e-6)

    def test_missing_hand_and_quantile_interpolation(self):
        obj, _ = self.problem()
        guard = obj.angular_guard
        guard.limits[:] = 0
        guard.limits[[1, 3, 5]] = np.nan
        values = np.full((5, 3, 2), np.nan)
        values[:, :, 0] = np.arange(1, 6)[:, None] * 0.1
        gradients = np.ones((5, 3, 2, 4))
        r, j = guard.aggregate(values, gradients)
        np.testing.assert_allclose(r[[0, 2, 4]], 300.0)
        np.testing.assert_allclose(r[[1, 3, 5]], 0.0)
        self.assertAlmostEqual(r[6], 480.0)
        np.testing.assert_allclose(j.toarray()[6].reshape(5, 4)[:, 0], [0.0, 0.0, 0.0, 200.0, 800.0])


if __name__ == "__main__":
    unittest.main()
