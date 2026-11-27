# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Export a tiny non-default-rate episode and check what the dataset records."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from steps import lerobot_writer, validate

FRAMES, RATE = 4, 15.0
EP = "ep0"


def make_video(path, frames, rate):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=640x368:rate={rate}",
            "-frames:v",
            str(frames),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
class DatasetExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ik, self.state, self.out = self.tmp / "ik", self.tmp / "state", self.tmp / "lerobot"
        self.ik.mkdir()
        self.state.mkdir()
        make_video(self.ik / f"{EP}_robot_on_bg.mp4", FRAMES, RATE)
        np.savez(self.state / f"{EP}.npz", state=np.zeros((FRAMES, 3)), annotations=json.dumps([{"text": "pick"}]))
        self.fields = dict(
            qpos=np.zeros((FRAMES, 3)),
            state_names=np.array(["a", "b", "c"]),
            ik_ok=np.ones(FRAMES, dtype=bool),
            ik_err_pos=np.zeros(FRAMES),
            ik_err_rot=np.zeros(FRAMES),
        )

    def export(self, fps=None, **extra):
        np.savez(self.ik / f"{EP}_ik.npz", **self.fields, **extra)
        args = SimpleNamespace(
            ik_dir=str(self.ik),
            state_dir=str(self.state),
            bg_video_dir=str(self.ik),
            output_dir=str(self.out),
            tmp_dir=str(self.tmp / "work"),
            episodes=None,
            fps=fps,
        )
        old = lerobot_writer.FPS
        self.addCleanup(setattr, lerobot_writer, "FPS", old)
        lerobot_writer.run(args)

    def test_export_keeps_the_input_rate(self):
        self.export(fps=RATE)
        info = json.loads((self.out / "meta/info.json").read_text())
        self.assertEqual(info["fps"], RATE)
        self.assertEqual(info["total_frames"], FRAMES)
        data = pq.read_table(self.out / "data/chunk-000/file-000.parquet")
        np.testing.assert_allclose(data.column("timestamp").to_pylist(), [i / RATE for i in range(FRAMES)], rtol=1e-6)
        args = SimpleNamespace(
            dataset_dir=str(self.out), expect_episodes=None, expect_frames=None, skip_loader_check=True, depth_dir=None
        )
        try:
            validate.run(args)
        except ImportError:
            self.skipTest("PyAV is required for the video check")

    def test_acceptance_results_are_exported(self):
        self.export(
            fps=RATE,
            acceptance_target_profile=np.array("stable_reference"),
            acceptance_original_passed=np.array(False),
            acceptance_stable_passed=np.array(True),
            acceptance_repaired_frames=np.array(13, dtype=np.int64),
            acceptance_original_tcp_rotation_max_deg=np.array(26.58),
        )
        row = pq.read_table(self.out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
        self.assertEqual(row["acceptance/target_profile"], "stable_reference")
        self.assertIs(row["acceptance/original_passed"], False)
        self.assertIs(row["acceptance/stable_passed"], True)
        self.assertEqual(row["acceptance/repaired_frames"], 13)
        self.assertAlmostEqual(row["acceptance/original_tcp_rotation_max_deg"], 26.58)


class ResampleTest(unittest.TestCase):
    def test_frame_count_follows_the_target_rate(self):
        try:
            from steps.loader import resample_to_target_fps
        except ImportError:
            self.skipTest("zarr is required by the loader")
        n = 355
        keys = ["left_obs_ee_pose", "right_obs_ee_pose", "left_obs_wrist_pose", "right_obs_wrist_pose", "obs_head_pose"]
        data = {key: np.tile([0, 0, 0, 1, 0, 0, 0], (n, 1)).astype(float) for key in keys}
        data.update(left_obs_keypoints=np.zeros((n, 63)), right_obs_keypoints=np.zeros((n, 63)))
        data.update(timestamps_ns=(np.arange(n) * 1e9 / 30).astype(np.int64), total_frames=n)
        out = resample_to_target_fps(data, 15.0)
        self.assertEqual(out["total_frames"], 178)
        self.assertEqual(len(out["obs_head_pose"]), 178)


if __name__ == "__main__":
    unittest.main()
