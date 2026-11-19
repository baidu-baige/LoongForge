# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Geometry and missing-data regressions; all reference points are synthetic."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from steps import hand_camera, loader, path_b


def hand_shape():
    points = np.random.default_rng(412).normal(size=(21, 3)) * [.03, .05, .015]
    points -= points[:1]
    return points.astype(np.float32)


class CameraGeometry(unittest.TestCase):
    def test_pnp_recovers_pose_with_offcenter_anisotropic_camera(self):
        k = np.array([[280., 0, 310], [0, 310, 220], [0, 0, 1]])
        shape = hand_shape().astype(np.float64)
        rotation = Rotation.from_euler("xyz", [.3, -.4, .2]).as_matrix()
        expected = shape @ rotation.T + [.07, -.02, .6]
        uv = hand_camera.project(expected, k)
        fitted, r, t, accepted = hand_camera.fit_camera_pose(shape + [1., 1., 3.], uv, k)
        self.assertTrue(accepted)
        np.testing.assert_allclose(fitted, expected, atol=1e-6)
        np.testing.assert_allclose(r, rotation, atol=1e-6)
        np.testing.assert_allclose(t, expected[0], atol=1e-6)

    def test_pnp_rejects_one_2d_outlier(self):
        k = np.array([[320., 0, 320], [0, 320, 240], [0, 0, 1]])
        shape = hand_shape().astype(np.float64)
        expected = shape @ Rotation.from_euler("xyz", [.12, -.08, .05]).as_matrix().T
        expected += [.04, -.02, .72]
        uv = hand_camera.project(expected, k)
        uv[8] += [85., -60.]
        fitted, _, _, accepted = hand_camera.fit_camera_pose(shape, uv, k, "pnp_ransac")
        self.assertTrue(accepted)
        inlier_error = np.linalg.norm(
            hand_camera.project(fitted, k) - hand_camera.project(expected, k), axis=1)
        self.assertLess(np.median(inlier_error), 1e-3)
        self.assertLess(inlier_error[np.arange(21) != 8].max(), 1e-2)

    def test_translation_fit_has_positive_depth_without_clipping_to_old_range(self):
        k = np.array([[290., 0, 320], [0, 290, 240], [0, 0, 1]])
        shape = hand_shape()
        for depth in [.15, 3.5]:
            expected = shape + [.01, -.01, depth]
            uv = hand_camera.project(expected, k)
            fitted, _, _, _ = hand_camera.fit_camera_pose(shape, uv, k, "translation")
            np.testing.assert_allclose(fitted, expected, atol=1e-5)
            self.assertGreater(fitted[:, 2].min(), .02)

    def test_video_focal_prior_only_clamps_gross_outliers(self):
        camera = hand_camera.assumed_camera(640, 480)
        camera.update(estimated=True, source="video:GeoCalib", fl_x=640., fl_y=640.)
        guarded = hand_camera.stabilize_camera_estimate(camera, max_relative_deviation=.35)
        self.assertTrue(guarded["focal_prior_applied"])
        self.assertAlmostEqual(guarded["fl_x"], 256.)
        self.assertEqual(guarded["raw_fl_x"], 640.)
        camera.update(fl_x=280., fl_y=280.)
        guarded = hand_camera.stabilize_camera_estimate(camera, max_relative_deviation=.35)
        self.assertFalse(guarded["focal_prior_applied"])
        self.assertAlmostEqual(guarded["fl_x"], 280.)

    def test_degenerate_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            hand_camera.fit_camera_pose(np.zeros((21, 3)), np.zeros((21, 2)), np.eye(3))

    def test_pnp_rejection_preserves_translation_solution(self):
        shape = hand_shape()
        k = hand_camera.camera_matrix(hand_camera.assumed_camera(640, 480))
        uv = hand_camera.project(shape + [.1, .02, .7], k)
        with patch.object(cv2, "solvePnP", return_value=(True, np.zeros((3, 1)),
                                                       np.array([[0.], [0.], [-1.]]))):
            fitted, _, _, accepted = hand_camera.fit_camera_pose(shape, uv, k)
        self.assertFalse(accepted)
        np.testing.assert_allclose(fitted, shape + [.1, .02, .7], atol=1e-6)

    def test_mano_correction_preserves_mirrored_wrist_pivot(self):
        canonical = hand_shape().astype(np.float64) + [.02, .01, -.01]
        root = Rotation.from_euler("xyz", [.1, .2, -.3]).as_matrix()
        correction = Rotation.from_euler("xyz", [.3, -.2, .1]).as_matrix()
        camera = hand_camera.assumed_camera(640, 480)
        camera["estimated"] = True
        k = hand_camera.camera_matrix(camera)
        for side in [0, 1]:
            mirror = np.diag([2 * side - 1, 1, 1])
            old_t = np.array([.1, .02, 1.])
            old = ((canonical - canonical[:1]) @ root.T + canonical[:1]) @ mirror + old_t
            expected = (old - old[:1]) @ correction.T + [.05, .03, .7]
            extra = {"global_orient": root.copy(), "cam_trans": old_t.copy()}
            item = {"keypoints": old.astype(np.float32), "keypoints_2d": hand_camera.project(expected, k),
                    "right": side, "score": 1., "_wilor_extra": extra}
            hand_camera.calibrate_predictions([item], camera)
            reconstructed = ((canonical - canonical[:1]) @ extra["global_orient"].T
                             + canonical[:1]) @ mirror + extra["cam_trans"]
            np.testing.assert_allclose(reconstructed, item["keypoints"], atol=1e-6)

    def test_reference_calibration_arguments_are_not_accepted(self):
        parser = path_b.build_arg_parser()
        for arg in ["--camera_source", "--intrinsics", "--source_zarr"]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--video", "rgb.mp4", "--output_dir", "out", arg, "gt.zarr"])

    def test_video_pipeline_needs_no_zarr_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = [np.zeros((48, 64, 3), np.uint8) for _ in range(12)]
            shape = hand_shape() + [0, 0, .7]
            camera = hand_camera.assumed_camera(64, 48)
            camera.update(estimated=True, source="video:GeoCalib")
            observations = hand_camera.project(shape, hand_camera.camera_matrix(camera))
            predictions = [{"keypoints": shape.copy(), "keypoints_2d": observations,
                            "right": 1., "frame": i, "score": .9} for i in range(12)]
            args = path_b.build_arg_parser().parse_args(
                ["--video", "only_input.mp4", "--output_dir", tmp, "--camera_motion", "static"])
            original_open = zarr.open_group
            def write_only(*args, **kwargs):
                self.assertEqual(kwargs.get("mode"), "w", "inference tried to read a Zarr")
                return original_open(*args, **kwargs)
            with patch.object(path_b, "read_video", return_value=(frames, 30.)), \
                 patch.object(path_b, "estimate_camera", return_value=camera), \
                 patch.object(path_b, "run_wilor", return_value=predictions), \
                 patch.object(zarr, "open_group", side_effect=write_only), \
                 contextlib.redirect_stdout(io.StringIO()):
                path_b.process_video(args)
            self.assertTrue((Path(tmp) / "video_000/zarr.json").is_file())


class MissingDataAndPoses(unittest.TestCase):
    def test_short_gap_interpolation_restores_reduced_confidence(self):
        detections = [
            {"frame": 1, "right": 1., "score": .8,
             "keypoints": hand_shape() + [0, 0, .7]},
            {"frame": 3, "right": 1., "score": .6,
             "keypoints": hand_shape() + [0, 0, .7]},
        ]
        tracks, scores = path_b.detections_to_tracks(detections, 5, 640, max_gap=1)
        self.assertTrue(np.isfinite(tracks["right"][2]).all())
        self.assertGreater(scores["right"][2], 0.)
        self.assertLess(scores["right"][2], min(scores["right"][1], scores["right"][3]))

    def test_tcp_rotation_roundtrip_including_negative_trace(self):
        base = hand_shape() + [0, 0, .7]
        angles = np.linspace(0, 2 * np.pi, 181)
        rotations = Rotation.from_rotvec(np.column_stack(
            [np.zeros_like(angles), np.zeros_like(angles), angles])).as_matrix()
        kp = np.einsum("nij,kj->nki", rotations, base)
        for sign in [-1., 1.]:
            pose, _ = path_b._tcp_pose(kp, sign)
            vf = .7 * kp[:, 8] + .3 * kp[:, 12]
            z = sign * (kp[:, 4] - vf)
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            y = np.cross(z, vf - kp[:, 0])
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            expected = np.stack([np.cross(y, z), y, z], axis=2)
            self.assertTrue(np.any(np.trace(expected, axis1=1, axis2=2) < 0))
            actual = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            self.assertTrue((np.sum(pose[1:, 3:] * pose[:-1, 3:], axis=1) >= 0).all())

    def test_degenerate_tcp_is_invalid(self):
        pose, _ = path_b._tcp_pose(np.ones((2, 21, 3)), 1.)
        self.assertTrue(np.isnan(pose).all())

    def test_only_short_internal_gaps_are_interpolated(self):
        arr = np.full((12, 21, 3), np.nan, np.float32)
        arr[1], arr[3], arr[10] = 1, 3, 10
        path_b._interpolate_short_gaps(arr, max_gap=2)
        np.testing.assert_allclose(arr[2], 2)
        self.assertTrue(np.isnan(arr[[0, 4, 5, 6, 7, 8, 9, 11]]).all())

    def test_smoothing_does_not_cross_long_gap(self):
        arr = np.tile(hand_shape(), (28, 1, 1))
        arr[11:17] = np.nan
        arr[17:] += 2
        one = path_b.smooth_tracks({"left": arr})["left"]
        arr[17:] += 20
        two = path_b.smooth_tracks({"left": arr})["left"]
        np.testing.assert_array_equal(one[:11], two[:11])
        self.assertTrue(np.isnan(two[11:17]).all())

    def test_smoothing_respects_full_image_reprojection(self):
        n = 21
        base = hand_shape().astype(np.float32) + [0., 0., .8]
        truth = np.stack([base + [0.01 * t, 0., 0.] for t in range(n)])
        noisy = truth.copy()
        noisy[10] += [0.08, -0.04, 0.12]
        camera = dict(fl_x=320., fl_y=320., cx=320., cy=240.,
                      w=640, h=480, estimated=True, source="synthetic")
        tracks_2d = {"left": np.stack([
            hand_camera.project(truth[t], hand_camera.camera_matrix(camera))
            for t in range(n)]).astype(np.float32)}
        scores = {"left": np.full(n, .2, np.float32)}
        result = path_b.smooth_tracks(
            {"left": noisy}, window=11, polyorder=3, scores=scores,
            tracks_2d=tracks_2d, camera=camera, reprojection_gate_px=3.)
        original_error = np.linalg.norm(noisy - truth, axis=-1).mean()
        result_error = np.linalg.norm(result["left"] - truth, axis=-1).mean()
        self.assertLess(result_error, original_error)
        projected = hand_camera.project(result["left"][10],
                                         hand_camera.camera_matrix(camera))
        self.assertLess(np.linalg.norm(projected - tracks_2d["left"][10], axis=-1).mean(), 8.)

    def test_zarr_masks_complete_pose_and_loader_preserves_video_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = [np.full((48, 64, 3), [20, 80, 180], np.uint8) for _ in range(12)]
            points = np.tile(hand_shape() + [0, 0, .7], (12, 1, 1))
            points[6, 15] = np.nan  # Invalid joint outside the TCP's fingertips.
            scores = {"left": np.ones(12, np.float32), "right": np.zeros(12, np.float32)}
            scores["left"][4] = 0
            episode = path_b.write_episode(Path(tmp), "episode", frames, 30.,
                {"left": points, "right": np.full_like(points, np.nan)}, scores,
                tracks_2d={"left": np.ones((12, 21, 2)), "right": np.full((12, 21, 2), np.nan)})
            store = zarr.open_group(str(episode), mode="r")
            for suffix in ["obs_keypoints", "obs_ee_pose", "obs_wrist_pose", "obs_keypoints_2d"]:
                self.assertTrue(np.isnan(store["left." + suffix][4]).all())
                self.assertTrue(np.isnan(store["left." + suffix][6]).all())
                self.assertTrue(np.isnan(store["right." + suffix][:]).all())
            rgb = loader.decode_frame_jpeg(str(episode), 4)
            self.assertEqual(rgb.shape, (48, 64, 3))
            np.testing.assert_allclose(rgb[10, 10], [180, 80, 20], atol=3)
            with contextlib.redirect_stdout(io.StringIO()):
                data = loader.process_episode(str(episode))
            self.assertEqual(data["total_frames"], 12)
            np.testing.assert_array_equal(data["source_frame_indices"], np.arange(12))
            self.assertFalse(data["frame_valid"][4])
            self.assertEqual(int(data["frame_valid"].sum()), 10)

    def test_partial_side_fields_are_invalid(self):
        data = {"total_frames": 1, "obs_head_pose": np.array([[0, 0, 0, 1, 0, 0, 0]])}
        for side in ["left", "right"]:
            for suffix, width in [("obs_ee_pose", 7), ("obs_wrist_pose", 7), ("obs_keypoints", 63)]:
                data[f"{side}_{suffix}"] = np.full((1, width), np.nan)
        data["left_obs_ee_pose"][:] = [0, 0, 1, 1, 0, 0, 0]
        self.assertFalse(loader.find_invalid_frames(data)[0])

    def test_scalar_object_jpeg_blob_unwraps_without_pointer_bytes(self):
        blob = b"\xff\xd8real_jpeg_payload"
        self.assertEqual(loader._blob_bytes(np.array(blob, dtype=object)), blob)
        self.assertEqual(loader._blob_bytes(np.array([np.array(blob, dtype=object)], dtype=object)), blob)


if __name__ == "__main__":
    unittest.main()
