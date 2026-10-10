# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for the :mod:`steps.retarget` package.

The implementation is split by responsibility under ``steps/retarget``. This
module keeps the historical CLI module and import paths stable for downstream
callers.
"""

if __package__:
    from .robot_registry import (  # noqa: F401
        ROBOT_SPECS,
        get_model_path,
        get_robot_spec,
        resolve_ee_ref,
        resolve_prefixed_ee_ref,
        resolve_robot_spec,
    )
    from .retarget.base_search import *  # noqa: F401,F403
    from .retarget.base_search import (  # noqa: F401
        _base_orientation_candidates,
        _base_visual_min_z,
        _candidate_score,
        _collision_branch_repair,
        _cross_arm_collision_metrics,
        _evaluate_base_pair_collisions,
        _is_cross_arm_pair,
        _is_known_baseline_self_contact,
        _print_progress,
        _rot_x,
        _rot_y,
        _rot_z,
        _rotation_angle,
        _screen_orientation_candidates,
        _sort_base_records,
        _quality_collision_metrics,
        _snap_base_to_support,
    )
    from .retarget.cli import build_arg_parser, main, run  # noqa: F401
    from .retarget.episode import *  # noqa: F401,F403
    from .retarget.ik import *  # noqa: F401,F403
    from .retarget.ik import (  # noqa: F401
        _JointContinuityLimit,
        _ee_pos,
        _ee_pose_and_jacobian,
        _prewarm,
        _prewarm_dls,
        _project_qpos_to_limits,
    )
    from .retarget.rendering import *  # noqa: F401,F403
    from .retarget.targets import *  # noqa: F401,F403
    from .retarget.targets import (  # noqa: F401
        _fill_invalid_keypoints,
        _quat_to_mat,
        _savgol_window,
        _smooth_rotations,
    )
else:
    from robot_registry import (  # noqa: F401
        ROBOT_SPECS,
        get_model_path,
        get_robot_spec,
        resolve_ee_ref,
        resolve_prefixed_ee_ref,
        resolve_robot_spec,
    )
    from retarget.base_search import *  # noqa: F401,F403
    from retarget.base_search import (  # noqa: F401
        _base_orientation_candidates,
        _base_visual_min_z,
        _candidate_score,
        _collision_branch_repair,
        _cross_arm_collision_metrics,
        _evaluate_base_pair_collisions,
        _is_cross_arm_pair,
        _is_known_baseline_self_contact,
        _print_progress,
        _rot_x,
        _rot_y,
        _rot_z,
        _rotation_angle,
        _screen_orientation_candidates,
        _sort_base_records,
        _quality_collision_metrics,
        _snap_base_to_support,
    )
    from retarget.cli import build_arg_parser, main, run  # noqa: F401
    from retarget.episode import *  # noqa: F401,F403
    from retarget.ik import *  # noqa: F401,F403
    from retarget.ik import (  # noqa: F401
        _JointContinuityLimit,
        _ee_pos,
        _ee_pose_and_jacobian,
        _prewarm,
        _prewarm_dls,
        _project_qpos_to_limits,
    )
    from retarget.rendering import *  # noqa: F401,F403
    from retarget.targets import *  # noqa: F401,F403
    from retarget.targets import (  # noqa: F401
        _fill_invalid_keypoints,
        _quat_to_mat,
        _savgol_window,
        _smooth_rotations,
    )


if __name__ == "__main__":
    main()
