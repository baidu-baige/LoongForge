# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic independent cameras, without loading the video benchmark or Zarr."""

import hashlib
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pycolmap

from steps.camera_selfcal import apply_camera, essential_error, propose_camera
from steps.hand_camera import camera_matrix, project, stabilize_camera_estimate
from steps.path_b import build_arg_parser
from steps import camera_motion


def make_database(path, focal=300., scale=1., low_parallax=False,
                  planar=False, changing_focal=False, validation_focal=None):
    rng = np.random.default_rng(20261111)
    width, height = int(640 * scale), int(360 * scale)
    camera = dict(fl_x=focal * scale * 1.5, fl_y=focal * scale * 1.5,
                  cx=284 * scale, cy=167 * scale, w=width, h=height,
                  source="synthetic:initial", estimated=True)
    points = rng.uniform([-1.3, -.8, 4], [1.3, .8, 7], (180, 3))
    with pycolmap.Database.open(path) as db:
        cam = pycolmap.Camera(model="PINHOLE", width=width, height=height,
                             params=[camera[k] for k in ("fl_x", "fl_y", "cx", "cy")],
                             has_prior_focal_length=True)
        camera_id = db.write_camera(cam)
        views = []
        for i in range(18):
            block, local = divmod(i, 6)
            r = cv2.Rodrigues(rng.normal(0, .15, 3))[0]
            center = rng.normal(0, .25, 3) * (.0001 if low_parallax else 1.)
            actual_focal = focal * scale * (1.8 if changing_focal and block == 1 else 1.)
            if block == 2 and validation_focal is not None:
                actual_focal = validation_focal * scale
            k = camera_matrix(dict(camera, fl_x=actual_focal, fl_y=actual_focal))
            uv = project((points - center) @ r.T, k)
            image_id = db.write_image(pycolmap.Image(name=f"{i:06d}.jpg", camera_id=camera_id))
            db.write_keypoints(image_id, uv.astype(np.float32))
            views.append((image_id, r, center, k))
        for i, j in itertools.combinations(range(18), 2):
            if i // 6 != j // 6:
                continue
            a, r1, c1, k1 = views[i]
            b, r2, c2, k2 = views[j]
            r = r2 @ r1.T
            t = r2 @ (c1 - c2)
            skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
            f = np.linalg.inv(k2).T @ skew @ r @ np.linalg.inv(k1)
            f /= np.linalg.norm(f)
            geometry = pycolmap.TwoViewGeometry(
                config=4 if planar else 3, F=f,
                inlier_matches=np.c_[np.arange(len(points)), np.arange(len(points))].astype(np.uint32))
            db.write_two_view_geometry(a, b, geometry)
    return camera


class VideoSelfCalibration(unittest.TestCase):
    def test_default_never_uses_resolution_focal_prior(self):
        args = build_arg_parser().parse_args(["--video", "rgb.mp4", "--output_dir", "out"])
        self.assertLess(args.camera_focal_prior_max_deviation, 0)
        self.assertEqual(args.pose_intrinsics_policy, "fixed")
        self.assertFalse(args.camera_self_calibration)
        camera = dict(fl_x=900., fl_y=900., cx=320., cy=180., w=640, h=360)
        self.assertEqual(stabilize_camera_estimate(camera), camera)

    def test_recovers_varied_focals_crop_centers_and_resolution(self):
        for focal, scale in [(150., 1.), (300., 1.), (650., 1.), (1100., 1.), (300., 2.)]:
            with self.subTest(focal=focal, scale=scale), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "database.db"
                initial = make_database(path, focal, scale)
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                result, report = propose_camera(path, initial)
                self.assertTrue(report["accepted"], report)
                self.assertAlmostEqual(result["fl_x"] / (focal * scale), 1., delta=.001)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
                self.assertEqual(result["cx"], initial["cx"])
                blocks = [set(block) for block in report["block_image_ids"]]
                self.assertFalse(blocks[0] & blocks[1] or blocks[0] & blocks[2] or blocks[1] & blocks[2])

    def test_planar_and_low_parallax_rejected(self):
        for options in [dict(planar=True), dict(low_parallax=True)]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "database.db"
                initial = make_database(path, **options)
                result, report = propose_camera(path, initial)
                self.assertFalse(report["accepted"], report)
                self.assertEqual(result, initial)

    def test_changing_focal_and_unseen_validation_focal_rejected(self):
        for options in [dict(changing_focal=True), dict(validation_focal=600.)]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "database.db"
                initial = make_database(path, **options)
                result, report = propose_camera(path, initial)
                self.assertFalse(report["accepted"], report)
                self.assertEqual(result, initial)

    def test_reverification_receives_selected_camera_and_original_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "database.db"
            camera = make_database(path)
            camera.update(fl_x=300., fl_y=300.)
            with patch.object(pycolmap, "verify_matches") as verify:
                apply_camera(path, camera, root / "pairs.txt")
            self.assertEqual(len((root / "pairs.txt").read_text().splitlines()), 45)
            verify.assert_called_once()
            with pycolmap.Database.open(path) as db:
                np.testing.assert_allclose(db.read_all_cameras()[0].params, [300., 300., 284., 167.])
                self.assertEqual(len(db.read_two_view_geometries()[0]), 0)

    def test_essential_error_invariant_to_matrix_scale(self):
        k = np.diag([400., 400., 1.])
        f = np.random.default_rng(0).normal(size=(3, 3))
        self.assertAlmostEqual(essential_error(f, k), essential_error(f * 123.4, k))

    def test_selected_camera_shared_by_hand_fit_depth_and_export(self):
        camera = dict(fl_x=450., fl_y=450., cx=320., cy=180., w=640, h=360,
                      source="synthetic:initial", estimated=True)
        selected = dict(camera, fl_x=300., fl_y=300., source="synthetic:view_graph")
        k = camera_matrix(selected)
        shape = np.random.default_rng(0).normal(size=(21, 3)) * [.03, .05, .01]
        expected = shape - shape[:1] + [.02, .03, .6]
        prediction = dict(frame=0, score=1., right=1., keypoints=shape + [1, 1, 2],
                          keypoints_2d=project(expected, k))
        poses = np.tile(np.eye(4), (4, 1, 1))
        valid = np.ones(4, bool)
        def reconstruct(frames, candidate, *args, **kwargs):
            self.assertTrue(kwargs["self_calibrate"])
            candidate.update(selected)
            return (object(), dict.fromkeys(range(4)), poses, valid, valid,
                    dict(registered_keyframes=4, sampled_frame_indices=list(range(4)),
                         sparse_points=300, mean_reprojection_error_px=.5,
                         camera_reference_frame_index=0, self_calibration={"accepted": True},
                         selected_intrinsics=selected))
        def depth(model, frame, received_k, **kwargs):
            np.testing.assert_allclose(received_k, k)
            return np.ones((360, 640)), np.ones((360, 640))
        observations = {(float(i), float(i)): 1. for i in range(80)}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(camera_motion, "reconstruct", side_effect=reconstruct), \
             patch.object(camera_motion, "load_metric_depth", return_value=object()), \
             patch.object(camera_motion, "infer_metric_depth", side_effect=depth), \
             patch.object(camera_motion, "depth_scale_observations", return_value=observations):
            (Path(tmp) / "config.json").write_text("{}")
            _, arrays, stats = camera_motion.estimate_motion(
                [np.zeros((360, 640, 3), np.uint8)] * 4, camera, [prediction],
                tmp, None, tmp, metric_depth_backend="da3_metric", self_calibrate=True)
        np.testing.assert_allclose(prediction["keypoints"], expected, atol=1e-6)
        np.testing.assert_allclose(arrays["intrinsics"], k)
        self.assertEqual(stats["motion_intrinsics"]["fl_x"], 300.)
        self.assertEqual(stats["hand_camera_fit"]["fitted"], 1)
        self.assertEqual(camera["fl_x"], 300.)


if __name__ == "__main__":
    unittest.main()
