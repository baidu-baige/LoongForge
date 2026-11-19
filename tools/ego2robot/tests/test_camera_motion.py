# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Geometry, scale and video-only boundaries for camera motion reconstruction."""

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

from steps import camera_geometry, camera_motion, dynhamr_official, hand_camera, path_b, loader
from tests.test_dynhamr_bridge import prediction_fields
from tests.test_path_b_geometry import hand_shape


class CameraMotionGeometry(unittest.TestCase):
    def test_first_camera_gauge_and_interpolation(self):
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[:, :3, :3] = Rotation.from_euler("z", [[30], [90]], degrees=True).as_matrix()
        poses[:, :3, 3] = [[3, 1, 2], [4, 2, 2]]
        out = camera_geometry.interpolate_c2w(poses, [0, 4], 5)
        np.testing.assert_allclose(out[0], np.eye(4), atol=1e-7)
        np.testing.assert_allclose(out[-1], np.linalg.inv(poses[0]) @ poses[1], atol=1e-6)
        self.assertAlmostEqual(Rotation.from_matrix(out[2, :3, :3]).magnitude(), np.pi/6, places=6)
        np.testing.assert_allclose(out[2, :3, 3], out[-1, :3, 3] / 2, atol=1e-7)

    def test_missing_endpoints_and_long_gaps_fail(self):
        poses = np.tile(np.eye(4), (2, 1, 1))
        for indices, n in [([1, 4], 5), ([0, 3], 5), ([0, 50], 51)]:
            with self.assertRaises(ValueError):
                camera_geometry.interpolate_c2w(poses, indices, n, max_gap=12)

    def test_partial_camera_tracking_has_explicit_invalid_ends_and_long_gaps(self):
        poses = np.tile(np.eye(4), (4, 1, 1))
        poses[:, 0, 3] = [1., 2., 3., 4.]
        full, valid = camera_geometry.complete_c2w(poses, [2, 4, 12, 14], 17, max_gap=3)
        expected = np.zeros(17, bool)
        expected[2:5], expected[12:15] = True, True
        np.testing.assert_array_equal(valid, expected)
        np.testing.assert_allclose(full[2], np.eye(4))
        self.assertTrue(np.isfinite(full).all())  # Optimizer placeholders only.

    def test_improper_rotation_rejected(self):
        bad = np.eye(4)[None]
        bad[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            camera_geometry.validate_trajectory(bad, 1)

    def test_metric_scale_rejects_outliers_and_weights_frames_equally(self):
        ratios = [np.r_[np.full(200, .2), 8., 30.],
                  np.full(25, .21), np.full(50, .19)]
        scale, info = camera_motion.robust_metric_scale(ratios)
        self.assertAlmostEqual(scale, .2)
        self.assertEqual(info["inlier_counts"], [200, 25, 50])
        self.assertFalse(info["reference_data_used"])
        self.assertFalse(info["physically_calibrated"])
        self.assertEqual(info["temporal_block_count"], 3)
        self.assertEqual(len(info["temporal_block_scales"]), 3)

    def test_inconsistent_or_insufficient_metric_scale_fails(self):
        for ratios in [[np.ones(10)] * 8, [np.ones(30)],
                       [np.full(30, .1), np.full(30, 1.), np.full(30, 10.)]]:
            with self.assertRaises(RuntimeError):
                camera_motion.robust_metric_scale(ratios)

    def test_temporal_scale_blocks_reject_a_biased_video_section(self):
        ratios = [np.full(30, .2)] * 3 + [np.full(30, .8)] * 3
        with self.assertRaisesRegex(RuntimeError, "temporally blocked"):
            camera_motion.robust_metric_scale(ratios)

    def test_mask_uses_video_hand_pixels_and_temporal_neighborhood(self):
        image = np.zeros((100, 120, 3), np.uint8)
        pred = [dict(frame=3, score=.9, keypoints_2d=np.tile([[40, 40], [60, 40], [50, 60]], (7, 1)))]
        mask = camera_motion.background_mask(image, pred, 4, margin=5)
        self.assertEqual(mask[48, 50], 0)
        self.assertEqual(mask[5, 5], 255)
        far = camera_motion.background_mask(image, pred, 9, margin=5)
        self.assertTrue((far == 255).all())

    def test_metric_depth_receives_rgb_and_the_fixed_geocalib_k(self):
        import torch

        frame = np.full((48, 64, 3), [20, 50, 180], np.uint8)
        k = np.array([[30., 0, 32], [0, 35., 24], [0, 0, 1]])
        outer = self
        class Model:
            def infer(self, rgb, camera):
                np.testing.assert_array_equal(rgb[:, 0, 0].numpy(), [180, 50, 20])
                np.testing.assert_allclose(camera.numpy(), k[None])
                outer.assertEqual(rgb.dtype, torch.uint8)
                return dict(depth=torch.full((1, 1, 48, 64), 1.2),
                            confidence=torch.ones((1, 1, 48, 64)),
                            intrinsics=torch.eye(3)[None] * 999)
        depth, confidence = camera_motion.infer_metric_depth(Model(), frame, k)
        np.testing.assert_allclose(depth, 1.2)
        np.testing.assert_allclose(confidence, 1.)

    def test_da3_metric_depth_uses_processed_focal_and_restores_video_size(self):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[..., 0] = 20  # B
        frame[..., 1] = 50  # G
        frame[..., 2] = 180  # R
        k = np.array([[300., 0, 32], [0, 360., 24], [0, 0, 1]])

        class Prediction:
            depth = np.full((1, 24, 32), 2., np.float32)
            conf = np.full((1, 24, 32), .8, np.float32)

        class Model:
            def inference(self, images, **kwargs):
                np.testing.assert_array_equal(images[0][0, 0], [180, 50, 20])
                self.kwargs = kwargs
                return Prediction()

        model = Model()
        depth, confidence = camera_motion.infer_metric_depth(
            {"backend": "da3_metric", "model": model}, frame, k,
            process_res=504)
        # The fake output has a 2x2 processed scale, so focal_processed is
        # (300+360)/2 * ((32/64)+(24/48))/2 = 165.
        np.testing.assert_allclose(depth, 2. * 165. / 300.)
        np.testing.assert_allclose(confidence, .8)
        self.assertEqual(model.kwargs["process_res"], 504)

    def test_background_pnp_localizes_camera_not_hand(self):
        rng = np.random.default_rng(7)
        xyz = rng.uniform([-.5, -.3, 2.], [.5, .3, 4.], (80, 3))
        k = np.array([[280., 0, 320], [0, 280., 240], [0, 0, 1]])
        rot = Rotation.from_euler("xyz", [.015, -.025, .01]).as_matrix()
        translation = np.array([.03, -.01, .015])
        a = hand_camera.project(xyz, k).astype(np.float32)
        b = hand_camera.project(xyz @ rot.T + translation, k).astype(np.float32)
        ok = np.ones((len(a), 1), np.uint8)
        frame = np.zeros((480, 640, 3), np.uint8)
        with patch.object(cv2, "calcOpticalFlowPyrLK",
                          side_effect=[(b[:, None], ok, None), (a[:, None], ok, None)]):
            result = camera_motion.localize_frame(frame, frame, a, xyz, k, np.eye(4),
                                                  np.full((480, 640), 255, np.uint8))
        self.assertIsNotNone(result)
        c2w, count, error = result
        np.testing.assert_allclose(c2w[:3, :3], rot.T, atol=1e-6)
        np.testing.assert_allclose(c2w[:3, 3], -rot.T @ translation, atol=1e-6)
        self.assertEqual(count, 80)
        self.assertLess(error, 1e-3)

    def test_multi_anchor_localization_rejects_disagreement(self):
        good = np.eye(4)
        other = np.eye(4)
        other[0, 3] = .5
        candidates = [
            (good, 40, .8, 3, np.zeros((40, 2)), np.ones((40, 3))),
            (other, 42, .7, 9, np.zeros((42, 2)), np.ones((42, 3))),
        ]
        selected, info = camera_motion.select_localization_candidate(
            candidates, scene_scale=2.)
        self.assertIsNone(selected)
        self.assertTrue(info["rejected"])
        self.assertEqual(info["candidate_count"], 2)

    def test_multi_anchor_localization_uses_consensus_then_reprojection(self):
        poses = [np.eye(4) for _ in range(3)]
        poses[1][0, 3] = .01
        poses[2][0, 3] = .012
        candidates = [
            (poses[0], 30, 1.2, 3, np.zeros((30, 2)), np.ones((30, 3))),
            (poses[1], 30, .8, 4, np.zeros((30, 2)), np.ones((30, 3))),
            (poses[2], 35, 1.0, 5, np.zeros((35, 2)), np.ones((35, 3))),
        ]
        selected, info = camera_motion.select_localization_candidate(
            candidates, scene_scale=2.)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[3], 4)
        self.assertEqual(info["consensus_count"], 3)

    def test_world_export_and_camera_roundtrip_preserve_missing(self):
        n = 12
        poses = np.tile(np.eye(4), (n, 1, 1))
        poses[:, :3, :3] = Rotation.from_euler("y", np.linspace(0, .7, n)[:, None]).as_matrix()
        poses[:, :3, 3] = np.linspace([0, 0, 0], [.1, .2, -.1], n)
        points = np.tile(hand_shape() + [0, 0, .7], (n, 1, 1))
        tracks = dict(left=points.copy(), right=np.full_like(points, np.nan))
        scores = dict(left=np.ones(n), right=np.zeros(n))
        scores["left"][3] = 0
        with tempfile.TemporaryDirectory() as tmp:
            episode = path_b.write_episode(Path(tmp), "ep",
                [np.zeros((48, 64, 3), np.uint8)]*n, 30, tracks, scores, c2w=poses)
            group = zarr.open_group(str(episode), mode="r")
            head = group["obs_head_pose"][:]
            np.testing.assert_allclose(head[:, :3], poses[:, :3, 3], atol=1e-6)
            np.testing.assert_allclose(Rotation.from_quat(head[:, [4, 5, 6, 3]]).as_matrix(),
                                       poses[:, :3, :3], atol=1e-6)
            world = {s: group[s+".obs_keypoints"][:].reshape(n, 21, 3) for s in tracks}
            restored = path_b.transform_tracks(world, poses, inverse=True)
            keep = scores["left"] > 0
            np.testing.assert_allclose(restored["left"][keep], points[keep], atol=1e-6)
            self.assertTrue(np.isnan(restored["left"][~keep]).all())
            self.assertTrue(np.isnan(restored["right"]).all())
            self.assertEqual(group.attrs["camera_pose_source"], "video:COLMAP")

    def test_moving_camera_bridge_writes_w2c(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = cv2.VideoWriter(str(root/"video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            for _ in range(6):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            np.savez(root/"pred.npz", **prediction_fields())
            poses = np.tile(np.eye(4), (6, 1, 1))
            poses[:, :3, 3] = np.linspace([0, 0, 0], [.1, 0, 0], 6)
            poses[:, :3, :3] = Rotation.from_euler("y", np.linspace(0, .2, 6)[:, None]).as_matrix()
            dynhamr_official._write_dynhamr_inputs(root/"inputs", root/"video.mp4",
                "seq", root/"pred.npz", camera_trajectory=poses)
            with np.load(root/"inputs/dynhamr/cameras/seq/shot-0/cameras.npz") as data:
                np.testing.assert_allclose(data["w2c"], np.linalg.inv(poses), atol=1e-7)

    def test_unlocalized_camera_masks_head_hands_and_loader_frame_valid(self):
        n = 12
        camera_valid = np.ones(n, bool)
        camera_valid[-3:] = False
        points = np.tile(hand_shape()+[0, 0, .7], (n, 1, 1))
        frames = [np.zeros((48, 64, 3), np.uint8)]*n
        with tempfile.TemporaryDirectory() as tmp:
            episode = path_b.write_episode(Path(tmp), "ep", frames, 30.,
                dict(left=points, right=np.full_like(points, np.nan)),
                dict(left=np.ones(n), right=np.zeros(n)),
                c2w=np.tile(np.eye(4), (n, 1, 1)), camera_valid=camera_valid,
                camera_observed=camera_valid)
            group = zarr.open_group(str(episode), mode="r")
            self.assertTrue(np.isnan(group["obs_head_pose"][-3:]).all())
            self.assertTrue(np.isnan(group["left.obs_keypoints"][-3:]).all())
            np.testing.assert_array_equal(group["path_b_left_confidence"][-3:], 0)
            with contextlib.redirect_stdout(io.StringIO()):
                loaded = loader.process_episode(str(episode))
            self.assertEqual(loaded["total_frames"], n)
            np.testing.assert_array_equal(loaded["frame_valid"], camera_valid)
            np.testing.assert_array_equal(loaded["path_b_camera_pose_valid"], camera_valid)

    def test_removed_backend_and_flags_are_rejected(self):
        parser = path_b.build_arg_parser()
        for flags in [["--camera_estimator", "da3"], ["--da3_model", "/tmp/model"],
                      ["--da3_motion", "hand_scale"]]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--video", "rgb.mp4", "--output_dir", "out"]+flags)

    def test_da3_metric_backend_has_video_only_flags(self):
        args = path_b.build_arg_parser().parse_args([
            "--video", "rgb.mp4", "--output_dir", "out",
            "--metric_depth_backend", "da3_metric",
            "--da3_repo", "/code/depth-anything-3",
            "--da3_metric_model", "/models/DA3Metric-Large",
        ])
        self.assertEqual(args.metric_depth_backend, "da3_metric")
        self.assertEqual(args.da3_metric_process_res, 504)

    def test_motion_pipeline_cannot_read_reference_zarr(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 12
            frames = [np.zeros((48, 64, 3), np.uint8)]*n
            camera = hand_camera.assumed_camera(64, 48)
            camera.update(source="video:GeoCalib", estimated=True)
            shape = hand_shape()+[0, 0, .7]
            uv = hand_camera.project(shape, hand_camera.camera_matrix(camera))
            predictions = [dict(frame=i, right=1, score=.9, keypoints=shape.copy(),
                                keypoints_2d=uv.copy()) for i in range(n)]
            poses = np.tile(np.eye(4), (n, 1, 1))
            poses[:, 0, 3] = np.linspace(0, .1, n)
            stats = dict(source="video:COLMAP", registered_keyframes=4,
                         translation_scale={"meters_per_native_unit": .3})
            args = path_b.build_arg_parser().parse_args([
                "--video", "only_rgb.mp4", "--output_dir", tmp,
                "--unidepth_model", "mock_model"])
            original = zarr.open_group
            def write_only(*args, **kwargs):
                self.assertEqual(kwargs.get("mode"), "w")
                return original(*args, **kwargs)
            with patch.object(path_b, "read_video", return_value=(frames, 30.)), \
                 patch.object(path_b, "estimate_camera", return_value=camera), \
                 patch.object(camera_motion, "estimate_motion", return_value=(poses, {"c2w": poses}, stats)), \
                 patch.object(path_b, "run_wilor", return_value=predictions), \
                 patch.object(zarr, "open_group", side_effect=write_only), \
                 contextlib.redirect_stdout(io.StringIO()):
                path_b.process_video(args)
            self.assertTrue((Path(tmp)/"video_000_camera.npz").is_file())


if __name__ == "__main__":
    unittest.main()
