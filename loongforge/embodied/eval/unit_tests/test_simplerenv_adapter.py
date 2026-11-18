# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SimplerEnvAdapter Google-robot sticky-gripper switch."""

from loongforge.embodied.eval.adapters.simplerenv import SimplerEnvAdapter


def _google(**kwargs):
    return SimplerEnvAdapter(
        task_name="google_robot_pick_coke_can",
        robot_setup="google_robot",
        rotation_mode="axis_angle",
        **kwargs,
    )


def _gripper(adapter, value):
    return float(adapter.action_from_canonical({"actions": [0, 0, 0, 0, 0, 0, value]})[-1])


def test_sticky_gripper_is_off_by_default_and_passes_through():
    adapter = _google()
    assert [_gripper(adapter, v) for v in (1.0, 0.25, 0.0)] == [1.0, 0.25, 0.0]


def test_sticky_gripper_enabled_holds_action_for_15_steps():
    adapter = _google(google_sticky_gripper=True)
    outputs = [_gripper(adapter, 1.0)] + [_gripper(adapter, 0.5) for _ in range(15)]
    assert outputs == [-1.0] * 15 + [0.0]


def test_switch_is_ignored_for_widowx():
    adapter = SimplerEnvAdapter(rotation_mode="axis_angle", google_sticky_gripper=True)
    assert _gripper(adapter, 1.0) == 1.0
    assert _gripper(adapter, 0.0) == -1.0
