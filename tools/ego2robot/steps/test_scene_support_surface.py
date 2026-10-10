# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import numpy as np

from steps.retarget import base_search as retarget_base_search
from steps.retarget.targets import _quat_to_mat
from steps.robot_registry import ROBOT_SPECS, get_robot_spec
from steps.robot_retarget import (
    _is_known_baseline_self_contact,
    _base_visual_min_z,
    _snap_base_to_support,
    base_frame,
    build_single_arm_model,
    estimate_scene_support_surface,
    interp_failed_frames,
    repair_branch_jumps,
    refine_arm_ik_pose,
    search_base_pose,
    solve_arm_ik_position_first,
    _fill_invalid_keypoints,
    smooth_retarget_targets,
)


def _check_estimate_horizontal_support_surface_from_metric_depth():
    height, width = 48, 64
    depth = np.ones((6, height, width), dtype=np.float32)
    head = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        (len(depth), 1),
    )
    K = np.array([
        [80.0, 0.0, width / 2.0],
        [0.0, 80.0, height / 2.0],
        [0.0, 0.0, 1.0],
    ])
    targets = np.array([
        [-0.08, -0.05, 0.80],
        [0.08, 0.05, 0.85],
    ])

    surface = estimate_scene_support_surface(
        depth, head, K, targets, reach=0.42)

    assert surface is not None
    np.testing.assert_allclose(surface["coefficients"], [0.0, 0.0, 1.0],
                               atol=1e-6)
    assert surface["rmse"] < 1e-6
    assert surface["inlier_count"] >= 200


def _check_so_arm101_visual_sole_snaps_to_support_plane():
    spec = get_robot_spec("so_arm101")
    model, spec = build_single_arm_model(spec, return_spec=True)
    base_min_z = _base_visual_min_z(model, spec.base_body)
    surface = {
        "coefficients": np.array([0.0, 0.0, 0.75]),
        "points_xy": np.array([[0.1, -0.2]]),
        "base_min_z": base_min_z,
        "base_up_z": -1.0,
    }

    base_pos, gap = _snap_base_to_support(
        np.array([0.1, -0.2, 2.0]), surface)

    assert abs((base_pos[2] - base_min_z) - 0.75) < 1e-6
    assert gap == 0.0


def _check_base_frame_accepts_dataset_gravity_direction():
    _, rotation = base_frame(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]))

    np.testing.assert_allclose(rotation[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, -1.0])
    assert np.linalg.det(rotation) > 0.999


def _check_scene_support_is_enabled_only_for_so_arm101():
    enabled = [name for name, spec in ROBOT_SPECS.items()
               if spec.scene_support_surface]
    assert enabled == ["so_arm101"]


def _check_base_search_snaps_every_candidate_to_support_surface():
    grid = np.linspace(-2.0, 2.0, 41)
    xx, yy = np.meshgrid(grid, grid)
    support = {
        "coefficients": np.array([0.1, 0.0, 0.2]),
        "points_xy": np.column_stack([xx.ravel(), yy.ravel()]),
        "base_min_z": -0.01,
        "max_local_gap": 0.16,
    }
    target_p = np.array([[0.0, 0.0, 0.5], [0.02, 0.01, 0.5]])
    target_R = np.repeat(np.eye(3)[None], len(target_p), axis=0)

    def position_solver(*args, **kwargs):
        return True, 0.0

    def pose_solver(*args, **kwargs):
        return np.zeros(1), True, 0.0, 0.0

    with patch.object(retarget_base_search, "solve_arm_ik_position_only_dls",
                      position_solver), patch.object(
                          retarget_base_search, "solve_arm_ik_dls", pose_solver):
        candidates, _ = search_base_pose(
            object(), None, None, None, None,
            target_p, target_R, np.eye(3), np.array([1.0, 0.0, 0.0]),
            sign=1.0, reach=0.42, camera_pos=np.array([0.0, 0.0, 2.0]),
            search_mode="fast", ik_solver="dls",
            base_orientation_mode="upright", max_target_distance=2.0,
            support_surface=support)

    assert candidates
    for candidate in candidates:
        x, _, z = candidate["base_pos"]
        assert abs(z - (0.1 * x + 0.21)) < 1e-9


def _check_balanced_base_orientation_search_modes():
    target_p = np.array([[0.0, 0.0, 0.5], [0.02, 0.01, 0.5]])
    target_R = np.repeat(np.eye(3)[None], len(target_p), axis=0)

    def position_solver(*args, **kwargs):
        return True, 0.0

    def pose_solver(*args, **kwargs):
        return np.zeros(1), True, 0.0, 0.0

    def run(trajectory_enabled, exhaustive=False):
        with patch.object(retarget_base_search, "solve_arm_ik_position_only_dls",
                          position_solver), patch.object(
                              retarget_base_search, "solve_arm_ik_dls", pose_solver):
            return search_base_pose(
                object(), None, None, None, None, target_p, target_R,
                np.eye(3), np.array([1.0, 0.0, 0.0]), sign=1.0, reach=1.0,
                camera_pos=np.array([0.0, 0.0, 2.0]),
                search_mode="balanced", ik_solver="dls",
                max_target_distance=3.0,
                trajectory_orientation_search=trajectory_enabled,
                enable_base_orientation_search=exhaustive)[0]

    fixed = run(False)
    searched = run(True)
    exhaustive = run(True, exhaustive=True)
    assert fixed
    assert all(candidate["orientation"] == (0.0, 0.0, 0.0)
               for candidate in fixed)
    assert any(candidate["orientation"] != (0.0, 0.0, 0.0)
               for candidate in searched)
    assert len({candidate["orientation"] for candidate in exhaustive}) >= len(
        {candidate["orientation"] for candidate in searched})


def _check_interpolation_uses_position_success_not_pose_success():
    qpos = np.array([
        [0.0, 10.0],
        [99.0, 99.0],
        [2.0, 12.0],
    ])
    position_ok = np.array([
        [True, True],
        [True, False],
        [True, True],
    ])

    repaired = interp_failed_frames(qpos, position_ok, [0], [1])

    assert repaired == 1
    # An orientation-only failure is position-valid and therefore untouched.
    assert qpos[1, 0] == 99.0
    # The genuinely position-invalid right arm is interpolated.
    assert qpos[1, 1] == 11.0


def _check_branch_repair_accepts_position_only_solution():
    class FakeModel:
        nq = 1
        qpos0 = np.zeros(1)
        njnt = 0

    def position_only_solver(*args, **kwargs):
        q = np.asarray(kwargs["q_init"], dtype=float).copy()
        q[0] += 0.1
        return q, False, 0.001, 0.5

    joints = np.array([[0.0], [2.0]])
    pose_ok = np.array([True, False])
    position_ok = np.array([True, True])
    fixed, repaired_pose_ok, repaired_position_ok = repair_branch_jumps(
        FakeModel(), [0], [0], [0], None, joints,
        np.zeros((2, 3)), np.repeat(np.eye(3)[None], 2, axis=0),
        np.zeros(3), np.eye(3), pose_ok,
        position_ok_flags=position_ok, jump_thresh=0.45, n_sweeps=1,
        ik_solver=position_only_solver)

    assert fixed == 1
    np.testing.assert_allclose(joints[:, 0], [0.0, 0.1])
    assert repaired_position_ok.tolist() == [True, True]
    assert repaired_pose_ok.tolist() == [True, False]


def _check_position_first_reports_position_and_pose_separately():
    class FakeContext:
        def solve(self, *args, **kwargs):
            assert kwargs["position_first"] is True
            assert kwargs["max_iterations"] == 500
            assert kwargs["tol_pos"] == 0.005
            return np.array([0.2]), True, 0.004, 0.5

    q, pose_ok, err_pos, err_rot = solve_arm_ik_position_first(
        object(), None, None, None, None, np.zeros(3), np.eye(3),
        mink_context=FakeContext())

    np.testing.assert_allclose(q, [0.2])
    assert pose_ok is False
    assert err_pos == 0.004
    assert err_rot == 0.5


def _check_pose_refinement_only_accepts_safe_improvement():
    class FakeContext:
        def __init__(self, result):
            self.result = result

        def solve(self, *args, **kwargs):
            assert kwargs["position_first"] is False
            return self.result

    baseline = (np.array([0.1]), False, 0.004, 0.5)
    improved = refine_arm_ik_pose(
        object(), [0], None, None, None, np.zeros(3), np.eye(3),
        baseline, previous_q=np.array([0.0]),
        mink_context=FakeContext((np.array([0.2]), True, 0.003, 0.2)))
    assert improved[-1] is True
    np.testing.assert_allclose(improved[0], [0.2])

    position_regression = refine_arm_ik_pose(
        object(), [0], None, None, None, np.zeros(3), np.eye(3),
        baseline, previous_q=np.array([0.0]),
        mink_context=FakeContext((np.array([0.2]), False, 0.006, 0.1)))
    assert position_regression[-1] is False
    np.testing.assert_allclose(position_regression[0], baseline[0])

    branch_jump = refine_arm_ik_pose(
        object(), [0], None, None, None, np.zeros(3), np.eye(3),
        baseline, previous_q=np.array([0.0]),
        mink_context=FakeContext((np.array([0.5]), True, 0.003, 0.1)),
        continuity_max_step=0.35)
    assert branch_jump[-1] is False
    np.testing.assert_allclose(branch_jump[0], baseline[0])


class SceneSupportSurfaceTest(unittest.TestCase):
    def test_retarget_target_smoothing_handles_invalid_frames_and_keeps_so3(self):
        positions = np.stack([np.array([0.01 * i, 0.0, 0.4]) for i in range(9)])
        positions[4] = np.nan
        rotations = np.repeat(np.eye(3)[None], len(positions), axis=0)
        rotations[4] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        widths = np.linspace(0.02, 0.06, len(positions))
        p, r, w = smooth_retarget_targets(positions, rotations, widths)
        assert np.isfinite(p).all()
        assert np.isfinite(w).all()
        np.testing.assert_allclose(np.linalg.det(r), np.ones(len(r)), atol=1e-6)
        np.testing.assert_allclose(np.einsum("nij,nkj->nik", r, r),
                                   np.broadcast_to(np.eye(3), r.shape), atol=1e-6)

    def test_invalid_keypoints_are_interpolated_without_sentinel_leak(self):
        values = np.array([[0.0, 1.0], [np.nan, 1e9], [2.0, 3.0]])
        filled = _fill_invalid_keypoints(values)
        np.testing.assert_allclose(filled[1], [1.0, 2.0])

    def test_estimate_horizontal_support_surface_from_metric_depth(self):
        _check_estimate_horizontal_support_surface_from_metric_depth()

    def test_so_arm101_visual_sole_snaps_to_support_plane(self):
        _check_so_arm101_visual_sole_snaps_to_support_plane()

    def test_base_frame_accepts_dataset_gravity_direction(self):
        _check_base_frame_accepts_dataset_gravity_direction()

    def test_scene_support_is_enabled_only_for_so_arm101(self):
        _check_scene_support_is_enabled_only_for_so_arm101()

    def test_base_search_snaps_every_candidate_to_support_surface(self):
        _check_base_search_snaps_every_candidate_to_support_surface()

    def test_balanced_base_orientation_search_modes(self):
        _check_balanced_base_orientation_search_modes()

    def test_interpolation_uses_position_success_not_pose_success(self):
        _check_interpolation_uses_position_success_not_pose_success()

    def test_branch_repair_accepts_position_only_solution(self):
        _check_branch_repair_accepts_position_only_solution()

    def test_position_first_reports_position_and_pose_separately(self):
        _check_position_first_reports_position_and_pose_separately()

    def test_pose_refinement_only_accepts_safe_improvement(self):
        _check_pose_refinement_only_accepts_safe_improvement()

    def test_aloha_duplicated_arms_use_identical_tcp_axes(self):
        spec = get_robot_spec("aloha_agilex")
        self.assertIsNone(spec.tcp_rot_site_left)
        self.assertIsNone(spec.tcp_rot_site_right)
        R = _quat_to_mat(spec.tcp_rot_site)
        # Common TCP z is approach (site +x); signed TCP y selects site -y.
        np.testing.assert_allclose(R[:, 2], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(R[:, 1], [0.0, -1.0, 0.0], atol=1e-6)

    def test_aloha_nested_base_shoulder_contact_is_baseline(self):
        assert _is_known_baseline_self_contact(
            "left_left/base_link", "left_left/shoulder_link")
        assert _is_known_baseline_self_contact(
            "right_left/shoulder_link", "right_left/base_link")
        assert not _is_known_baseline_self_contact(
            "left_left/gripper_base", "left_left/upper_forearm_link")


if __name__ == "__main__":
    unittest.main()
