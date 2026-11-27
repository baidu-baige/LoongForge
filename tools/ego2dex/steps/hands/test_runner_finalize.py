# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Finalization of registered-hand outputs when an earlier run left other episodes behind."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


def write_result(output, ep, passed):
    result = output / "_render" / ep
    result.mkdir(parents=True)
    report = dict(
        passed=passed,
        target_profile="stable_reference",
        source_rotation_changed_frames=[3, 4],
        task_tcp_position_max_m=0.002,
        task_tcp_rotation_max_rad=0.03,
    )
    original = dict(report, passed=False, task_tcp_rotation_max_rad=0.5, source_rotation_changed_frames=[])
    (result / "stable_reference_validation.json").write_text(json.dumps(report))
    (result / "original_reference_validation.json").write_text(json.dumps(original))
    np.savez(
        result / "motor_actions.npz",
        motor_state=np.zeros((2, 4)),
        motor_names=np.array(list("abcd")),
        physical_state_names=np.array(list("abcd")),
        physical_from_motor_matrix=np.eye(4),
        physical_from_motor_offset=np.zeros(4),
    )
    np.savez(result / "ik_quality.npz", ik_ok=np.ones(2, bool), ik_err_pos=np.zeros(2), ik_err_rot=np.zeros(2))
    np.savez(output / f"{ep}_ik.npz", qpos=np.zeros((2, 4)))


class FinalizeTest(unittest.TestCase):
    def test_stale_failed_episode_is_ignored(self):
        try:
            from steps.hands import runner
        except ImportError:
            self.skipTest("the hand pipeline dependencies are not installed")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        write_result(tmp, "A", passed=True)
        write_result(tmp, "B", passed=False)
        args = SimpleNamespace(episodes=["A"], end_effector="allegro")
        with mock.patch("steps.hands._runner_base.replace_pipeline_outputs", return_value=tmp):
            runner.replace_pipeline_outputs(args, tmp, tmp)
        done = np.load(tmp / "A_ik.npz")
        self.assertFalse(bool(done["acceptance_original_passed"]))
        self.assertTrue(bool(done["acceptance_stable_passed"]))
        self.assertEqual(int(done["acceptance_repaired_frames"]), 2)
        self.assertAlmostEqual(float(done["acceptance_original_tcp_rotation_max_deg"]), np.degrees(0.5))
        self.assertNotIn("acceptance_stable_passed", np.load(tmp / "B_ik.npz").files)


if __name__ == "__main__":
    unittest.main()
