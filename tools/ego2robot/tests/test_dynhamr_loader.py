# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the patched fa9cd74 dataset loader without model weights or GPU.

Set EGO2ROBOT_DYNHAMR_REPO to an upstream checkout containing fa9cd74. Sources
are read from Git and patched in a temporary directory; the checkout is untouched.
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from steps import dynhamr_official as bridge
from tests.test_dynhamr_bridge import prediction_fields


@unittest.skipUnless(os.environ.get("EGO2ROBOT_DYNHAMR_REPO"),
                     "set EGO2ROBOT_DYNHAMR_REPO to test the actual upstream loader")
class DynhamrLoaderIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="dynhamr-loader-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        repo = os.environ["EGO2ROBOT_DYNHAMR_REPO"]
        compatibility = Path(bridge.__file__).with_name("dynhamr_compat.patch")
        paths = [line.split(" b/", 1)[1]
                 for line in compatibility.read_text().splitlines()
                 if line.startswith("diff --git ")]
        paths.append("dyn-hamr/data/tools.py")
        for path in paths:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(
                ["git", "-C", repo, "show", f"fa9cd74:{path}"]))
        subprocess.run(["git", "apply", "--check", str(compatibility)], cwd=root, check=True)
        subprocess.run(["git", "apply", str(compatibility)], cwd=root, check=True)

        # The real loader uses only model joint counts and logging here.
        # Model construction and video preprocessing are outside this test.
        body_model = types.ModuleType("body_model")
        body_model.MANO_JOINTS = list(range(16))
        body_model.OP_NUM_JOINTS = 21
        logger = types.ModuleType("util.logger")
        logger.Logger = types.SimpleNamespace(log=lambda *args: None)
        camera = types.ModuleType("geometry.camera")
        camera.invert_camera = None
        package_name = "_ego2robot_dynhamr_loader_test"
        package = types.ModuleType(package_name)
        package.__path__ = [str(root / "dyn-hamr/data")]
        vidproc = types.ModuleType(f"{package_name}.vidproc")
        for name in ("preprocess_cameras", "preprocess_frames", "preprocess_tracks"):
            setattr(vidproc, name, None)
        modules = {
            "body_model": body_model, "util.logger": logger,
            "geometry.camera": camera, package_name: package,
            f"{package_name}.vidproc": vidproc,
        }
        previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            spec = importlib.util.spec_from_file_location(
                f"{package_name}.dataset", root / "dyn-hamr/data/dataset.py")
            cls.dataset_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.dataset_module)
        finally:
            # Restore only our shims. Removing newly imported Torch submodules
            # would leave its C extension in a partially initialized state.
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            sys.modules.pop(f"{package_name}.tools", None)

    def make_inputs(self, root):
        video = root / "video.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        self.assertTrue(writer.isOpened())
        for _ in range(7):
            writer.write(np.zeros((48, 64, 3), np.uint8))
        writer.release()
        raw = prediction_fields()
        # Both hands have two observations, with gaps and unobserved boundaries.
        raw = {key: np.repeat(value, 2, axis=0) for key, value in raw.items()}
        raw["frame"] = np.array([1, 5, 2, 4])
        raw["keypoints_2d"][1] += 10
        np.savez(root / "predictions.npz", **raw)
        output = root / "inputs"
        bridge._write_dynhamr_inputs(
            output, video, "seq", root / "predictions.npz")
        return output

    def load_dataset(self, output, track_ids="000-001", interp=True):
        sources = {
            "images": str(output / "images/seq"),
            "tracks": str(output / "dynhamr/track_preds/seq"),
            "shots": str(output / "dynhamr/shot_idcs/seq.json"),
            "cameras": str(output / "dynhamr/cameras/seq/shot-0"),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = self.dataset_module.MultiPeopleDataset(
                sources, "seq", tid_spec=track_ids, is_static=True)
            dataset.load_data(interp_input=interp)
        return dataset

    def test_missing_frames_remain_invisible_after_actual_loader(self):
        for interp in (False, True):
            with self.subTest(interp=interp), tempfile.TemporaryDirectory() as tmp:
                output = self.make_inputs(Path(tmp))
                dataset = self.load_dataset(output, interp=interp)
                self.assertEqual((dataset.start_idx, dataset.end_idx), (0, 7))
                np.testing.assert_array_equal(
                    dataset.track_vis_masks[0], [False, True, False, False, False, True, False])
                np.testing.assert_array_equal(
                    dataset.track_vis_masks[1], [False, False, True, False, True, False, False])
                expected_visibility = (
                    [-1, 1, 0, 0, 0, 1, -1], [-1, -1, 1, 0, 1, -1, -1])
                for side, expected in enumerate(expected_visibility):
                    observed = np.asarray(expected) > 0
                    joints = dataset.data_dict["joints2d"][side]
                    np.testing.assert_array_equal(
                        dataset.data_dict["vis_mask"][side], expected)
                    np.testing.assert_array_equal(joints[~observed], 0)
                    np.testing.assert_allclose(joints[observed, :, 2], .8 if side == 0 else .9)
                    self.assertTrue(np.isfinite(dataset.data_dict["init_trans"][side]).all())
                self.assertEqual(len(dataset.get_camera_data()["cam_R"]), 7)

    def test_visibility_rejects_low_confidence_and_nonfinite_keypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "keypoints.json"
            for confidence, coordinate, visible in (
                    (0., 20., False), (.2, 20., False), (.8, float("nan"), False),
                    (.8, 20., True)):
                with self.subTest(confidence=confidence, coordinate=coordinate):
                    body = np.tile([coordinate, 20., confidence], (21, 1))
                    file.write_text(json.dumps({
                        "people": [{"pose_keypoints_2d": body.reshape(-1).tolist()}]}))
                    self.assertEqual(self.dataset_module.keypoints_visible(file), visible)

    def test_track_selection_counts_observations_instead_of_placeholder_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.make_inputs(Path(tmp))
            empty = output / "dynhamr/track_preds/seq/002"
            empty.mkdir()
            body = np.tile([20., 20., 0.], (21, 1))
            for frame in range(7):
                (empty / f"{frame:06d}_keypoints.json").write_text(json.dumps({
                    "people": [{"pose_keypoints_2d": body.reshape(-1).tolist()}]}))
            with patch.object(self.dataset_module, "MIN_TRACK_LEN", 0):
                dataset = self.load_dataset(output, track_ids="all")
            self.assertEqual(dataset.track_ids, ["000", "001"])


if __name__ == "__main__":
    unittest.main()
