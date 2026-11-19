# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Partial Path B camera localization must not poison retarget base search."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from steps import path_b
from steps.retarget import episode as retarget
from steps.robot_registry import ROBOT_SPECS
from tests.test_path_b_geometry import hand_shape


class RetargetCameraValidity(unittest.TestCase):
    def write_episode(self, root):
        n = 12
        points = np.tile(hand_shape() + [0., 0., .7], (n, 1, 1))
        camera = np.tile(np.eye(4), (n, 1, 1))
        camera[:, 0, 3] = np.arange(n) * .01
        valid = np.ones(n, bool)
        valid[[0, 5, 11]] = False
        episode = path_b.write_episode(
            root, "episode", [np.zeros((48, 64, 3), np.uint8)] * n, 30.,
            dict(left=points, right=np.full_like(points, np.nan)),
            dict(left=np.ones(n), right=np.zeros(n)), c2w=camera,
            camera_valid=valid)
        return episode, valid

    def test_world_loader_honors_camera_flags_even_for_finite_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode, valid = self.write_episode(root)
            group = zarr.open_group(str(episode), mode="a")
            group["obs_head_pose"][0] = [100., 200., 300., 1., 0., 0., 0.]
            group["left.obs_keypoints"][0] = np.ones(63)
            head, left, right, actual = retarget.load_episode_world(
                root, "episode", 12, return_camera_valid=True)
            np.testing.assert_array_equal(actual, valid)
            self.assertEqual(len(head), 12)
            self.assertTrue(np.isnan(head[~valid]).all())
            self.assertTrue(np.isnan(left[~valid]).all())
            self.assertTrue(np.isnan(right).all())
            prepared = retarget.prepare_camera_poses(head, actual)
            self.assertTrue(np.isfinite(prepared).all())
            np.testing.assert_array_equal(prepared[valid], head[valid])
            np.testing.assert_array_equal(prepared[0], head[1])
            np.testing.assert_array_equal(prepared[-1], head[-2])

    def test_partial_localization_reaches_base_search_with_finite_geometry(self):
        class ReachedBaseSearch(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, valid = self.write_episode(root)
            expected_position = [np.arange(12)[valid].mean() * .01, 0., 0.]
            spec = ROBOT_SPECS["panda"]

            def check_search(*args, **kwargs):
                self.assertTrue(np.isfinite(args[5]).all())  # target positions
                self.assertTrue(np.isfinite(args[6]).all())  # target orientations
                self.assertTrue(np.isfinite(args[7]).all())  # nominal base rotation
                self.assertTrue(np.isfinite(args[8]).all())  # camera forward
                np.testing.assert_allclose(kwargs["camera_pos"], expected_position, atol=1e-7)
                raise ReachedBaseSearch

            # Run the real episode loading, validity, target conversion, and
            # base-frame calculations; stop before model-dependent IK search.
            with patch.object(retarget, "build_single_arm_model", return_value=(object(), spec)), \
                    patch.object(retarget, "name_to_dof", return_value=([], [], [])), \
                    patch.object(retarget, "resolve_ee_ref", return_value=("site", 0)), \
                    patch.object(retarget, "search_base_pose", side_effect=check_search), \
                    self.assertRaises(ReachedBaseSearch):
                retarget.process_episode(
                    "episode", root, "unused.mp4", np.zeros((12, 16)),
                    root / "out", 30., 2, 100., 48, False, ik_solver="dls",
                    use_scene_support=False, trajectory_refine=False, spec=spec)

    def test_legacy_episode_without_validity_flags_detects_nonfinite_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode, valid = self.write_episode(root)
            group = zarr.open_group(str(episode), mode="a")
            del group["path_b_camera_pose_valid"]
            _, _, _, actual = retarget.load_episode_world(
                root, "episode", 12, return_camera_valid=True)
            np.testing.assert_array_equal(actual, valid)

    def test_all_invalid_camera_poses_fail_explicitly(self):
        head = np.full((12, 7), np.nan)
        with self.assertRaisesRegex(ValueError, "no valid camera poses"):
            retarget.prepare_camera_poses(head, np.zeros(12, bool))
        with self.assertRaisesRegex(ValueError, "without valid camera poses"):
            retarget.head_forward_flat(head)

    def test_gaze_ignores_missing_poses(self):
        head = np.array([[0., 0., 0., 1., 0., 0., 0.], [np.nan] * 7])
        np.testing.assert_allclose(
            retarget.head_forward_flat(head), retarget.head_forward_flat(head[:1]))

    def test_support_surface_ignores_unlocalized_cameras(self):
        depth = np.ones((12, 48, 64), np.float32)
        head = np.tile([0., 0., 0., 1., 0., 0., 0.], (12, 1))
        head[[0, 5, 11]] = np.nan
        camera = np.array([[80., 0., 32.], [0., 80., 24.], [0., 0., 1.]])
        targets = np.array([[-.08, -.05, .80], [.08, .05, .85]])
        surface = retarget.estimate_scene_support_surface(
            depth, head, camera, targets, reach=.42)
        self.assertIsNotNone(surface)
        np.testing.assert_allclose(surface["coefficients"], [0., 0., 1.], atol=1e-6)
        head[:] = np.nan
        self.assertIsNone(retarget.estimate_scene_support_surface(
            depth, head, camera, targets, reach=.42))


if __name__ == "__main__":
    unittest.main()
