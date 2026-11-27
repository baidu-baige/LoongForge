# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from steps.robot_registry import get_robot_spec
from .joint_pose import symmetric_targets, eligible, joint_pose_search


class JointPoseTests(unittest.TestCase):
    def test_symmetry_preserves_tcp_and_approach_but_swaps_fingers(self):
        spec = get_robot_spec("panda")
        R = Rotation.from_rotvec(np.arange(18).reshape(6, 3) * 0.1).as_matrix().reshape(3, 2, 3, 3)
        p = np.arange(18).reshape(3, 2, 3) * 0.01
        p2, R2 = symmetric_targets(p, R, spec, [True, True])
        offset = np.array(spec.tcp_pos_body)
        np.testing.assert_allclose(
            p + np.einsum("nsij,j->nsi", R, offset), p2 + np.einsum("nsij,j->nsi", R2, offset), atol=1e-12
        )
        np.testing.assert_allclose(R[:, :, :, 2], R2[:, :, :, 2], atol=1e-12)
        np.testing.assert_allclose(R[:, :, :, 1], -R2[:, :, :, 1], atol=1e-12)
        p3, R3 = symmetric_targets(p2, R2, spec, [True, True])
        np.testing.assert_allclose(p3, p, atol=1e-12)
        np.testing.assert_allclose(R3, R, atol=1e-12)

    def test_unsupported_asymmetric_gripper_rejected(self):
        spec = replace(get_robot_spec("panda"), gripper_mode="asymmetric")
        with self.assertRaises(ValueError):
            symmetric_targets(np.zeros((3, 2, 3)), np.tile(np.eye(3), (3, 2, 1, 1)), spec, [True, False])

    def test_better_mean_cannot_hide_other_arm_or_tail_regression(self):
        b = dict(
            accepted=True,
            velocity_rms_rad_s=1.0,
            acceleration_rms_rad_s2=10.0,
            symmetry_orientation_mean_deg=[2.0, 50.0],
            approach_mean_deg=[1.0, 20.0],
            opening_line_mean_deg=[1.0, 20.0],
            symmetry_orientation_p95_deg=80.0,
        )
        self.assertTrue(eligible(b, b))
        self.assertFalse(eligible({**b, "symmetry_orientation_mean_deg": [8.0, 20.0]}, b))
        self.assertFalse(eligible({**b, "accepted": False}, b))
        self.assertFalse(eligible({**b, "symmetry_orientation_p95_deg": 85.0}, b))

    def test_bounded_tradeoff_requires_large_pose_gain_and_smoothness(self):
        b = dict(
            accepted=True,
            velocity_rms_rad_s=1.0,
            acceleration_rms_rad_s2=10.0,
            max_acceleration_rad_s2=100.0,
            symmetry_orientation_mean_deg=[2.0, 50.0],
            approach_mean_deg=[1.0, 20.0],
            opening_line_mean_deg=[1.0, 20.0],
            symmetry_orientation_p95_deg=80.0,
        )
        good = {
            **b,
            "velocity_rms_rad_s": 1.15,
            "acceleration_rms_rad_s2": 8.0,
            "max_acceleration_rad_s2": 70.0,
            "symmetry_orientation_mean_deg": [3.5, 20.0],
        }
        self.assertTrue(eligible(good, b))
        self.assertFalse(eligible({**good, "velocity_rms_rad_s": 1.26}, b))
        self.assertFalse(eligible({**good, "symmetry_orientation_mean_deg": [3.5, 49.0]}, b))
        self.assertFalse(eligible({**good, "max_acceleration_rad_s2": 110.0}, b))

    def test_invalid_seed_is_not_a_relative_quality_baseline(self):
        # Recovery cannot inherit artificially low velocities from a stuck
        # seed, but must never admit a candidate failing the absolute guards.
        invalid = dict(
            accepted=False,
            velocity_rms_rad_s=0.0,
            symmetry_orientation_mean_deg=[3.0, 4.0],
            approach_mean_deg=[2.0, 3.0],
            opening_line_mean_deg=[1.0, 2.0],
            symmetry_orientation_p95_deg=8.0,
        )
        self.assertTrue(eligible({**invalid, "accepted": True}, invalid))
        self.assertFalse(eligible(invalid, invalid))
        self.assertFalse(eligible({**invalid, "accepted": True, "opening_line_mean_deg": [60.0, 2.0]}, invalid))

    def test_recovery_searches_past_shortlist_and_never_exports_failed_seed(self):
        model = SimpleNamespace(geom_margin=np.zeros(1), cam_fovy=np.array([80.0]))
        track = np.zeros((4, 2))
        invalid = dict(
            accepted=False,
            self_collision_frames=0,
            cross_arm_collision_frames=0,
            midpoint_collision_intervals=0,
            position_max_m=0.1,
            position_failed_frames=4,
        )
        problem = SimpleNamespace(
            measure=lambda q: (dict(invalid), None, None),
            lower=np.full(2, -2.0),
            upper=np.full(2, 2.0),
            frame=lambda *a: None,
            solve=lambda q, **kw: q,
            evaluate=lambda q: (np.zeros(16), None),
        )
        metrics = dict(
            symmetry_orientation_mean_deg=[1.0, 1.0],
            approach_mean_deg=[1.0, 1.0],
            opening_line_mean_deg=[1.0, 1.0],
            symmetry_orientation_p95_deg=1.0,
        )
        args = (
            model,
            track,
            [0, 1],
            [],
            np.zeros((4, 2, 3)),
            np.tile(np.eye(3), (4, 2, 1, 1)),
            np.ones((4, 2), bool),
            30,
            get_robot_spec("panda"),
            np.array([[-0.3, 0, 0], [0.3, 0, 0]]),
            np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
        )
        for success_at in (6, None):
            calls = []

            def reconstruct(*a, **kw):
                calls.append(1)
                return track.copy(), {**invalid, "accepted": len(calls) == success_at}, None, None

            with (
                patch("steps.retarget.joint_pose.ContinuousProblem", return_value=problem),
                patch("steps.retarget.joint_pose.pose_metrics", return_value=metrics),
                patch("steps.retarget.joint_pose.build_dual_model", return_value=model),
                patch("steps.retarget.joint_pose.mujoco.mj_name2id", return_value=0),
                patch("steps.retarget.joint_pose.reconstruct_continuous_trajectory", side_effect=reconstruct),
                patch("steps.retarget.joint_pose.refine_posture", return_value=(track.copy(), {}, None, None)),
            ):
                selected, report = joint_pose_search(*args)
            self.assertEqual(len(calls), success_at or 20)
            self.assertEqual(report["recovered"], success_at is not None)
            self.assertEqual(report["search_exhausted"], success_at is None)
            if success_at:
                self.assertTrue(selected["report"]["accepted"])
            else:
                self.assertIsNone(selected)
                self.assertIsNone(report["after"])


if __name__ == "__main__":
    unittest.main()
