# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .box_clearance import box_gap_gradient


class BoxClearanceTests(unittest.TestCase):
    def test_axis_aligned_gap_and_overlap(self):
        zeros = np.zeros((3, 1))
        for x, expected in ((3.0, 1.0), (1.5, -0.5)):
            gap, _ = box_gap_gradient(
                np.zeros(3),
                np.eye(3),
                np.ones(3),
                np.array([x, 0, 0]),
                np.eye(3),
                np.ones(3),
                zeros,
                zeros,
                zeros,
                zeros,
            )
            self.assertAlmostEqual(gap, expected)

    def test_rotating_boxes_directional_derivative(self):
        rng = np.random.default_rng(417)
        for _ in range(30):
            p1, p2 = rng.normal(size=(2, 3))
            R1, R2 = Rotation.random(2, random_state=rng).as_matrix()
            h1, h2 = rng.uniform(0.03, 0.2, size=(2, 3))
            jp1, jr1, jp2, jr2 = rng.normal(size=(4, 3, 1))
            gap, gradient = box_gap_gradient(p1, R1, h1, p2, R2, h2, jp1, jr1, jp2, jr2)
            vals = []
            for eps in (-1e-6, 1e-6):
                vals.append(
                    box_gap_gradient(
                        p1 + eps * jp1[:, 0],
                        Rotation.from_rotvec(eps * jr1[:, 0]).as_matrix() @ R1,
                        h1,
                        p2 + eps * jp2[:, 0],
                        Rotation.from_rotvec(eps * jr2[:, 0]).as_matrix() @ R2,
                        h2,
                        jp1,
                        jr1,
                        jp2,
                        jr2,
                    )[0]
                )
            self.assertAlmostEqual(gradient[0], (vals[1] - vals[0]) / 2e-6, places=5)
