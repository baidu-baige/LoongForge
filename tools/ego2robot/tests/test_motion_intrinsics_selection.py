# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Video-only focal selection must resist missing data and easy-point bias."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from steps import camera_motion
from steps.camera_motion import compare_intrinsics_candidates


class IntrinsicsSelection(unittest.TestCase):
    def setUp(self):
        self.quality = {
            name: dict(valid_fraction=.98, registered_fraction=.95,
                       sparse_points=1000, reprojection_px=.8)
            for name in ("original", "protected")
        }
        # Fixed pixels occupy the background, independently of candidate K.
        self.pixels = [(x * 1000, y * 1000)
                       for x in range(40, 601, 70) for y in range(30, 331, 50)]
        self.obs = {name: {} for name in self.quality}
        noise = np.linspace(-1, 1, len(self.pixels))
        for t in (0, 10, 20, 30):
            for name, spread, scale in (("original", .4, 2.), ("protected", .08, .3)):
                self.obs[name][t] = dict(zip(self.pixels, scale * np.exp(spread * noise)))

    def choose(self, quality=None, observations=None):
        return compare_intrinsics_candidates(
            quality or self.quality, observations or self.obs, 640, 360)

    def test_depth_shape_selects_improvement_with_unknown_metric_scale(self):
        chosen, info = self.choose()
        self.assertEqual(chosen, "protected")
        self.assertFalse(info["reference_data_used"])
        self.assertEqual(info["candidates"]["protected"]["comparison_frames"], 4)

    def test_reprojection_alone_cannot_select_alternative(self):
        self.obs["protected"] = copy.deepcopy(self.obs["original"])
        self.quality["protected"]["reprojection_px"] = .01
        self.assertEqual(self.choose()[0], "original")

    def test_coverage_or_reprojection_loss_rejects_depth_improvement(self):
        for field, bad_value in (("valid_fraction", .8),
                                 ("registered_fraction", .8),
                                 ("reprojection_px", 4.)):
            quality = copy.deepcopy(self.quality)
            quality["protected"][field] = bad_value
            self.assertEqual(self.choose(quality=quality)[0], "original", field)

    def test_fewer_map_points_with_reliable_shared_support_can_win(self):
        self.quality["protected"]["sparse_points"] = 250
        chosen, info = self.choose()
        self.assertEqual(chosen, "protected")
        self.assertEqual(info["policy"], "video_consistency_v2")
        self.assertEqual(info["candidates"]["protected"]["background_support"]["sparse_points_ratio"], .25)

    def test_too_few_reliable_observations_rejects_candidate(self):
        for name in self.obs:
            self.obs[name] = {
                t: dict(list(points.items())[::2]) for t, points in self.obs[name].items()}
        chosen, info = self.choose()
        self.assertEqual(chosen, "original")
        self.assertFalse(info["candidates"]["protected"]["gates"]["reliable_background_support"])

    def test_temporally_concentrated_support_rejects_candidate(self):
        for name in self.obs:
            for t in (40, 50, 60, 70):
                self.obs[name][t] = {(t, 1 if name == "original" else 2): 1.}
        chosen, info = self.choose()
        self.assertEqual(chosen, "original")
        self.assertFalse(info["candidates"]["protected"]["gates"]["temporal_support"])

    def test_one_good_view_cannot_hide_regressions_in_other_views(self):
        noise = np.linspace(-1, 1, len(self.pixels))
        for t, spread in zip(self.obs["protected"], (.001, .41, .41, .41)):
            self.obs["protected"][t] = dict(zip(self.pixels, .3 * np.exp(spread * noise)))
        chosen, info = self.choose()
        self.assertEqual(chosen, "original")
        gates = info["candidates"]["protected"]["gates"]
        self.assertTrue(gates["depth_improved"])
        self.assertFalse(gates["depth_improved_across_views"])

    def test_extra_easy_points_do_not_change_comparison(self):
        before = self.choose()[1]["candidates"]["protected"]["log_depth_error"]
        for points in self.obs["protected"].values():
            points.update({(i, i): .3 for i in range(500)})
        after = self.choose()[1]["candidates"]["protected"]["log_depth_error"]
        self.assertEqual(before, after)

    def test_insufficient_common_frames_or_pixels_keeps_original(self):
        for frame in (0, 10):
            del self.obs["protected"][frame]
        self.assertEqual(self.choose()[0], "original")

    def test_spatially_concentrated_matches_cannot_select_focal(self):
        for name in self.obs:
            self.obs[name] = {t: {(i * 1000, i * 1000): r
                                 for i, r in enumerate(points.values())}
                              for t, points in self.obs[name].items()}
        self.assertEqual(self.choose()[0], "original")

    def test_temporally_inconsistent_candidate_is_ineligible(self):
        for t, factor in zip(self.obs["protected"], (.01, .1, 10., 100.)):
            self.obs["protected"][t] = {
                p: r * factor for p, r in self.obs["protected"][t].items()}
        chosen, info = self.choose()
        self.assertEqual(chosen, "original")
        self.assertFalse(info["candidates"]["protected"]["eligible"])

    def test_selected_k_controls_depth_scale_export_and_metadata(self):
        frames = [np.zeros((360, 640, 3), np.uint8)] * 4
        camera = dict(fl_x=400., fl_y=400., cx=320., cy=180., w=640, h=360,
                      source="video:GeoCalib")
        alternative = dict(camera, fl_x=256., fl_y=256.,
                           source="video:GeoCalib:resolution_prior")
        poses = np.tile(np.eye(4), (4, 1, 1))
        poses[:, 0, 3] = np.linspace(0, 1., 4)
        valid = np.ones(4, bool)

        def reconstruct(*args, **kwargs):
            return (object(), {t: t for t in range(4)}, poses, valid, valid,
                    dict(registered_keyframes=4, sampled_frame_indices=list(range(4)),
                         sparse_points=1000, mean_reprojection_error_px=.8,
                         camera_reference_frame_index=0))

        samples = iter([self.obs[name][t] for t in self.obs["original"]
                        for name in ("original", "protected")])
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(camera_motion, "reconstruct", side_effect=reconstruct), \
             patch.object(camera_motion, "load_metric_depth", return_value=object()), \
             patch.object(camera_motion, "infer_metric_depth",
                          return_value=(np.ones((360, 640)), np.ones((360, 640)))) as infer, \
             patch.object(camera_motion, "depth_scale_observations",
                          side_effect=lambda *args: next(samples)):
            (Path(tmp) / "config.json").write_text("{}")
            result, arrays, stats = camera_motion.estimate_motion(
                frames, camera, [], tmp, None, tmp,
                metric_depth_backend="da3_metric",
                intrinsics_policy="video_consistency", alternative_camera=alternative)
            self.assertEqual(infer.call_count, 4)  # Same DA3 network outputs for both K.
            np.testing.assert_allclose(arrays["intrinsics"][0, 0], 256.)
            np.testing.assert_allclose(arrays["metric_depth"], 256. / 400.)
            self.assertAlmostEqual(result[-1, 0, 3], .3, places=3)
            self.assertEqual(stats["intrinsics_selection"]["selected"], "protected")
            saved = json.loads((Path(tmp) / "camera_metadata.json").read_text())
            self.assertEqual(saved["motion_intrinsics"]["fl_x"], 256.)

    def test_low_parallax_recovery_stops_when_registration_is_sufficient(self):
        models = [SimpleNamespace(num_reg_images=lambda n=n: n) for n in (3, 80, 90)]
        mapper = SimpleNamespace(init_min_tri_angle=8.)
        angles = []

        def mapping(*args, **kwargs):
            angles.append(kwargs["options"].mapper.init_min_tri_angle)
            return {0: models[len(angles) - 1]}

        module = SimpleNamespace(incremental_mapping=mapping)
        selected, info = camera_motion.map_with_parallax_recovery(
            module, "database", "images", "/tmp/test", SimpleNamespace(mapper=mapper),
            70, enabled=True)
        self.assertIs(selected, models[1])
        self.assertEqual(angles, [8., 4.])
        self.assertEqual(info["selected_init_min_tri_angle_deg"], 4.)

    def test_initialization_failure_does_not_fabricate_a_model(self):
        module = SimpleNamespace(incremental_mapping=lambda *a, **k: {})
        selected, info = camera_motion.map_with_parallax_recovery(
            module, "database", "images", "/tmp/test",
            SimpleNamespace(mapper=SimpleNamespace()), 70, enabled=True)
        self.assertIsNone(selected)
        self.assertEqual(len(info["attempts"]), 3)

    def test_sparse_replay_cannot_use_a_different_camera(self):
        camera = dict(fl_x=400., fl_y=400., cx=320., cy=180.)
        model = SimpleNamespace(cameras={1: SimpleNamespace(params=[256., 256., 320., 180.])})
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, "intrinsics"):
            camera_motion.localize_reconstruction(
                [np.zeros((360, 640, 3), np.uint8)] * 7,
                camera, [], tmp, model, np.arange(1, 6), 1)

    def test_sparse_replay_preserves_invalid_endpoints(self):
        camera = dict(fl_x=400., fl_y=400., cx=320., cy=180.)
        images = {}
        for t in range(1, 6):
            pose = np.eye(4)
            pose[0, 3] = t * .01
            rigid = SimpleNamespace(matrix=lambda pose=pose: pose[:3])
            images[t] = SimpleNamespace(name=f"{t:06d}.jpg",
                                        cam_from_world=lambda rigid=rigid: rigid)
        model = SimpleNamespace(
            cameras={1: SimpleNamespace(params=[400., 400., 320., 180.])},
            images=images, num_points3D=lambda: 0,
            compute_mean_reprojection_error=lambda: .5)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(camera_motion, "image_map_points",
                          return_value=(np.empty((0, 2)), np.empty((0, 3)))):
            result = camera_motion.localize_reconstruction(
                [np.zeros((360, 640, 3), np.uint8)] * 7,
                camera, [], tmp, model, np.arange(1, 6), 1)
            self.assertTrue(np.isfinite(result[2]).all())
            np.testing.assert_array_equal(result[4], [False, True, True, True, True, True, False])
            with np.load(Path(tmp) / "native_trajectory.npz") as data:
                self.assertTrue(np.isnan(data["c2w_unscaled"][[0, -1]]).all())
                self.assertEqual(int(data["pose_observed"].sum()), 5)

    def test_rejected_localization_is_not_used_as_a_tracking_anchor(self):
        camera = dict(fl_x=400., fl_y=400., cx=320., cy=180.)
        images = {}
        for t in (0, 3, 6, 9, 12):
            rigid = SimpleNamespace(matrix=lambda: np.eye(4)[:3])
            images[t] = SimpleNamespace(
                name=f"{t:06d}.jpg", cam_from_world=lambda rigid=rigid: rigid)
        model = SimpleNamespace(
            cameras={1: SimpleNamespace(params=[400., 400., 320., 180.])},
            images=images, num_points3D=lambda: 30,
            compute_mean_reprojection_error=lambda: .5)
        uv, xyz = np.zeros((30, 2)), np.ones((30, 3))
        frames = [np.full((8, 8, 3), t, np.uint8) for t in range(13)]

        def localize(source, target, *args, **kwargs):
            self.assertNotEqual(int(source[0, 0, 0]), 1)
            return np.eye(4), 30, .5, uv, xyz

        def select(candidates, scene_scale, target_frame=None):
            if target_frame == 1:
                return None, dict(rejected=True, candidate_count=len(candidates))
            return candidates[0], dict(rejected=False)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(camera_motion, "image_map_points", return_value=(uv, xyz)), \
             patch.object(camera_motion, "localize_frame", side_effect=localize), \
             patch.object(camera_motion, "select_localization_candidate", side_effect=select):
            result = camera_motion.localize_reconstruction(
                frames, camera, [], tmp, model, np.array([0, 3, 6, 9, 12]), 3)
        self.assertFalse(result[3][1])  # Still unobserved; only the short gap is interpolated.
        self.assertTrue(result[3][2])
        self.assertTrue(result[4].all())
        self.assertTrue(result[-1]["flow_pnp"]["1"]["rejected"])


if __name__ == "__main__":
    unittest.main()
