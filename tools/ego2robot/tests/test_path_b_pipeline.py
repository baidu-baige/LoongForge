# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check reconstruction validity and frame mapping through load and align."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from steps import action_alignment, loader, path_b
from tests.test_path_b_geometry import hand_shape


class PathBPipeline(unittest.TestCase):
    def write_episode(self, root, fps=30.):
        n = 12
        points = np.tile(hand_shape() + [0., 0., .7], (n, 1, 1))
        scores = dict(left=np.ones(n, np.float32), right=np.zeros(n, np.float32))
        scores["left"][4] = 0
        camera_valid = np.ones(n, bool)
        camera_valid[-2:] = False
        return path_b.write_episode(
            root, "episode", [np.zeros((48, 64, 3), np.uint8)] * n, fps,
            dict(left=points, right=np.full_like(points, np.nan)), scores,
            c2w=np.tile(np.eye(4), (n, 1, 1)),
            camera_valid=camera_valid, camera_observed=camera_valid)

    def test_load_and_align_persist_frame_mapping_and_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_episode(root / "zarr")
            with contextlib.redirect_stdout(io.StringIO()):
                loader.run(SimpleNamespace(
                    input_dir=str(root / "zarr"), output_dir=str(root / "load"),
                    target_fps=30.))
                action_alignment.run(SimpleNamespace(
                    input_dir=str(root / "load"), output_dir=str(root / "align"),
                    skip=5, no_smooth=False, smooth_window=11))
            with np.load(root / "load/episode.npz") as loaded, np.load(
                    root / "align/episode.npz") as aligned:
                self.assertEqual(int(aligned["total_frames"]), 12)
                self.assertEqual(aligned["state"].shape, (12, 16))
                np.testing.assert_array_equal(
                    aligned["source_frame_indices"], np.arange(12))
                expected_valid = np.ones(12, bool)
                expected_valid[[4, 10, 11]] = False
                np.testing.assert_array_equal(aligned["frame_valid"], expected_valid)
                for key in loaded.files:
                    if key.startswith("path_b_"):
                        np.testing.assert_array_equal(aligned[key], loaded[key])
                self.assertTrue(np.isnan(aligned["state"][4]).all())

    def test_resampling_keeps_missing_hands_and_camera_flags_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = self.write_episode(Path(tmp), fps=15.)
            with contextlib.redirect_stdout(io.StringIO()):
                data = loader.process_episode(str(episode), target_fps=30.)
            self.assertEqual(data["total_frames"], 23)
            nearest = np.rint(np.linspace(0, 11, 23)).astype(int)
            np.testing.assert_array_equal(data["source_frame_indices"], nearest)
            for side in ("left", "right"):
                observed = data[f"path_b_{side}_observed"]
                self.assertEqual(observed.shape, (23,))
                self.assertTrue(np.isnan(
                    data[f"{side}_obs_keypoints"][~observed]).all())
                np.testing.assert_array_equal(
                    data[f"path_b_{side}_confidence"][~observed], 0)
            self.assertFalse(data["path_b_right_observed"].any())
            invalid_camera = ~data["path_b_camera_pose_valid"]
            self.assertFalse(data["frame_valid"][invalid_camera].any())
            self.assertTrue(np.isnan(data["obs_head_pose"][invalid_camera]).all())


if __name__ == "__main__":
    unittest.main()
