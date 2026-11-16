# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Geometric and inference-boundary checks for metric hand translation."""

import copy
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from steps import camera_motion, hand_camera, path_b
from tests.test_path_b_geometry import hand_shape


class HandDepthAlignment(unittest.TestCase):
    def setUp(self):
        self.camera = dict(fl_x=260., fl_y=280., cx=160., cy=120.,
                           w=320, h=240, source="video:test", estimated=True)
        self.k = hand_camera.camera_matrix(self.camera)
        self.shape = hand_shape()
        self.target = self.shape + [0., 0., .7]
        self.uv = hand_camera.project(self.target, self.k)
        self.frame = np.zeros((240, 320, 3), np.uint8)
        self.depth = np.zeros((240, 320), np.float32)
        for uv, xyz in zip(self.uv, self.target):
            x, y = np.rint(uv).astype(int)
            self.depth[max(0, y-1):y+2, max(0, x-1):x+2] = xyz[2]
        self.item = dict(frame=0, right=1., score=.9,
                         keypoints=(self.shape + [0., 0., .76]).astype(np.float32),
                         keypoints_2d=self.uv.copy(),
                         _wilor_extra={"cam_trans": np.array([0., 0., .76])})

    def align(self, predictions, depth=None, confidence=None, **kwargs):
        depth = self.depth if depth is None else depth
        confidence = np.ones_like(depth) if confidence is None else confidence
        with patch.object(camera_motion, "infer_metric_depth",
                          return_value=(depth, confidence)) as infer:
            result = camera_motion.align_hand_predictions_to_metric_depth(
                [self.frame] * 4, predictions, object(), self.camera, "da3_metric",
                patch_size=1, **kwargs)
        return result, infer

    def test_recovers_translation_preserves_shape_and_updates_mano(self):
        item = copy.deepcopy(self.item)
        old = item["keypoints"].copy()
        result, _ = self.align([item])
        self.assertEqual(result["hands_aligned"], 1)
        np.testing.assert_allclose(item["keypoints"] - item["keypoints"][:1],
                                   old - old[:1], atol=1e-7)
        np.testing.assert_allclose(item["keypoints"][0], self.target[0], atol=2e-3)
        delta = item["keypoints"][0] - old[0]
        np.testing.assert_allclose(item["_wilor_extra"]["cam_trans"],
                                   [0., 0., .76] + delta, atol=1e-7)
        self.assertFalse(result["reference_data_used"])
        self.assertFalse(result["confidence_informative"])

    def test_invalid_depth_and_large_correction_leave_input_unchanged(self):
        for depth in (np.zeros_like(self.depth), np.full_like(self.depth, np.inf),
                      np.full_like(self.depth, 2.)):
            item = copy.deepcopy(self.item)
            old = item["keypoints"].copy()
            result, _ = self.align([item], depth=depth)
            self.assertEqual(result["hands_aligned"], 0)
            np.testing.assert_array_equal(item["keypoints"], old)

    def test_samples_once_per_frame_and_respects_stride(self):
        items = [dict(copy.deepcopy(self.item), frame=t) for t in range(4)]
        items.append(dict(copy.deepcopy(self.item), frame=0, right=0.))
        old = items[1]["keypoints"].copy()
        result, infer = self.align(items, frame_stride=2)
        self.assertEqual(result["frames_considered"], 3)  # 0, 2, endpoint 3
        self.assertEqual(infer.call_count, 3)
        np.testing.assert_array_equal(items[1]["keypoints"], old)

    def test_cannot_hide_inference_failure(self):
        with patch.object(camera_motion, "infer_metric_depth",
                          side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                camera_motion.align_hand_predictions_to_metric_depth(
                    [self.frame], [self.item], object(), self.camera, "da3_metric")

    def test_missing_2d_and_nonfinite_points_are_rejected(self):
        a = copy.deepcopy(self.item)
        del a["keypoints_2d"]
        b = copy.deepcopy(self.item)
        b["keypoints"][0] = np.nan
        result, _ = self.align([a, b])
        self.assertEqual(result["hands_aligned"], 0)
        self.assertEqual(result["rejection_reasons"],
                         {"missing_2d": 1, "nonfinite_observation": 1})

    def test_invalid_options_and_cli_default(self):
        for stride, patch_size, mad, delta in [(0, 7, .03, .15), (1, 2, .03, .15),
                                               (1, 7, -1., .15), (1, 7, .03, np.nan)]:
            with self.assertRaises(ValueError):
                camera_motion.validate_hand_depth_options(stride, patch_size, mad, delta)
        args = path_b.build_arg_parser().parse_args(
            ["--video", "rgb.mp4", "--output_dir", "out"])
        self.assertFalse(args.hand_depth_alignment)

    def test_static_pipeline_depth_alignment_can_only_write_zarr(self):
        """Model depth is the only external 3D input in the enabled path."""
        with tempfile.TemporaryDirectory() as tmp:
            args = path_b.build_arg_parser().parse_args([
                "--video", "only_rgb.mp4", "--output_dir", tmp,
                "--camera_motion", "static", "--hand_depth_alignment",
                "--hand_depth_patch_size", "1",
                "--metric_depth_backend", "da3_metric",
                "--da3_metric_model", "/mock/video_model",
            ])
            original_open = zarr.open_group
            def write_only(*args, **kwargs):
                self.assertEqual(kwargs.get("mode"), "w")
                return original_open(*args, **kwargs)
            items = [dict(copy.deepcopy(self.item), frame=i) for i in range(12)]
            for item in items:
                del item["_wilor_extra"]
            with patch.object(path_b, "read_video",
                              return_value=([self.frame]*12, 30.)), \
                 patch.object(path_b, "estimate_camera", return_value=self.camera), \
                 patch.object(path_b, "run_wilor", return_value=items), \
                 patch.object(camera_motion, "load_metric_depth", return_value=object()), \
                 patch.object(camera_motion, "infer_metric_depth",
                              return_value=(self.depth, np.ones_like(self.depth))), \
                 patch.object(zarr, "open_group", side_effect=write_only), \
                 contextlib.redirect_stdout(io.StringIO()):
                path_b.process_video(args)
            group = original_open(str(Path(tmp)/"video_000"), mode="r")
            self.assertTrue(group.attrs["hand_depth_alignment"]["enabled"])
            self.assertFalse(group.attrs["reference_data_used_for_reconstruction"])


if __name__ == "__main__":
    unittest.main()
