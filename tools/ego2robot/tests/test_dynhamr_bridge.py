# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Contract regressions for the official Dyn-HaMR / Path B bridge."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from steps import dynhamr_official as bridge
from steps import path_b


def prediction_fields():
    return dict(pred_keypoints_3d=np.full((2, 21, 3), .8, np.float32),
                keypoints_2d=np.full((2, 21, 2), 20, np.float32),
                hand_pose=np.tile(np.eye(3, dtype=np.float32), (2, 15, 1, 1)),
                global_orient=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1, 1)),
                betas=np.zeros((2, 10), np.float32),
                cam_trans=np.array([[-.2, .1, .7], [.3, .1, .8]], np.float32),
                right=np.array([0, 1]), frame=np.array([0, 2]),
                score=np.array([.8, .9], np.float32))


class DynhamrBridgeContracts(unittest.TestCase):
    def test_rich_offline_predictions_keep_mano_and_2d(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'predictions.npz'
            raw = prediction_fields()
            np.savez(file, **raw)
            detections = path_b._load_prediction_file(file)
            self.assertEqual(len(detections), 2)
            for i, detection in enumerate(detections):
                np.testing.assert_array_equal(detection['keypoints_2d'], raw['keypoints_2d'][i])
                for key in ['hand_pose', 'global_orient', 'betas', 'cam_trans']:
                    np.testing.assert_array_equal(detection['_wilor_extra'][key], raw[key][i])

    def test_both_hands_full_video_and_missing_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'video.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(6):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            raw = prediction_fields()
            predictions = root / 'predictions.npz'
            np.savez(predictions, **raw)
            track_ids = bridge._write_dynhamr_inputs(root / 'inputs', video, 'seq', predictions)
            self.assertEqual(track_ids, '000-001')
            for side in ['000', '001']:
                track = root / 'inputs/dynhamr/track_preds/seq' / side
                self.assertEqual(len(list(track.glob('*_mano.json'))), 6)
                missed = json.loads((track / '000005_keypoints.json').read_text())
                confidence = np.array(missed['people'][0]['pose_keypoints_2d']).reshape(21, 3)[:, 2]
                np.testing.assert_array_equal(confidence, 0)
            left = json.loads((root / 'inputs/dynhamr/track_preds/seq/000/000000_mano.json').read_text())
            self.assertAlmostEqual(left['cam_trans'][0], .2)
            with np.load(root / 'inputs/dynhamr/cameras/seq/shot-0/cameras.npz') as camera:
                self.assertEqual(camera['w2c'].shape[0], 6)
                self.assertEqual(float(camera['focal']), 64)

    def test_result_selection_requires_latest_smooth_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smooth = root / 'seq/smooth_fit'
            smooth.mkdir(parents=True)
            final = smooth / 'seq_000300_world_results.npz'
            np.savez(final, pose_body=[1])
            np.savez(smooth / 'seq_000020_world_results.npz', pose_body=[2])
            (root / 'seq/root_fit').mkdir()
            np.savez(root / 'seq/root_fit/seq_000050_world_results.npz', pose_body=[3])
            self.assertEqual(bridge._find_result(root), final)

    def test_export_restores_left_camera_handedness(self):
        joints = np.ones((2, 3, 21, 3), np.float32)
        restored = bridge._restore_camera_handedness(joints, np.array([[0], [1]]))
        np.testing.assert_array_equal(restored[0, ..., 0], -1)
        np.testing.assert_array_equal(restored[1, ..., 0], 1)
        np.testing.assert_array_equal(restored[..., 1:], 1)

    def test_partial_root_run_is_not_a_completed_smooth_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'root_fit').mkdir()
            np.savez(root / 'root_fit/seq_000050_world_results.npz', pose_body=[1])
            with self.assertRaisesRegex(RuntimeError, 'smooth'):
                bridge._find_result(root)

    def test_integration_keeps_both_optimized_translations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / 'source.npz'
            np.savez(file, **prediction_fields())
            predictions = path_b._load_prediction_file(file)
            exported = np.full((2, 21, 3), .2, np.float32)
            def run(cmd, **kwargs):
                output = Path(cmd[cmd.index('--output') + 1])
                output.mkdir()
                np.savez(output / 'dynhamr_predictions.npz', pred_keypoints_3d=exported,
                         right=[0, 1], frame=[0, 2], score=[.8, .9])
                return type('Completed', (), {'returncode': 0})()
            with patch.object(path_b.subprocess, 'run', side_effect=run):
                refined = path_b.apply_official_dynhamr(predictions, 'video.mp4', 'repo',
                    'mano', root, '0', True, image_size=(64, 48), n_frames=3)
            self.assertEqual(len(refined), 2)
            for detection in refined:
                np.testing.assert_array_equal(detection['keypoints'], exported[0])
                self.assertIn('keypoints_2d', detection)


if __name__ == '__main__':
    unittest.main()
