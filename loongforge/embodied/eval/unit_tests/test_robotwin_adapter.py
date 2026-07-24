# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pytest

from loongforge.embodied.eval.adapters.robotwin import ROBOTWIN_ACTION_REORDER, RoboTwinAdapter
from loongforge.embodied.eval.bridges.robotwin_policy import (
    ModelClient,
    _adapt_to_pi_decode_state,
    _adapt_to_pi_encode_actions,
    _delta_to_absolute_actions,
    _ee6d_action_to_env,
    _PI05_DELTA_JOINT_MASK,
    _PI05_JOINT_FLIP_MASK,
)
from loongforge.embodied.eval.orchestrator.runners.robotwin_runner import (
    _build_robotwin_records,
    _override_eval_policy_for_smoke,
    _override_step_limit,
    _parse_official_episode_outcomes,
    _robotwin_rate_from_result,
    build_command,
    _write_deploy_policy,
)


def fake_robotwin_obs() -> dict:
    """Run fake_robotwin_obs."""
    return {
        "observation": {
            "head_camera": {"rgb": np.full((2, 3, 3), 10, dtype=np.uint8)},
            "left_camera": {"rgb": np.full((2, 3, 3), 20, dtype=np.uint8)},
            "right_camera": {"rgb": np.full((2, 3, 3), 30, dtype=np.uint8)},
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }


def test_robotwin_obs_to_canonical_extracts_three_cameras_and_joint_state() -> None:
    """Run test_robotwin_obs_to_canonical_extracts_three_cameras_and_joint_state."""
    adapter = RoboTwinAdapter(task_name="pick_object")

    canonical = adapter.obs_to_canonical(
        fake_robotwin_obs(),
        {"instruction": "pick the object", "episode_id": "robotwin/pick_object/0", "episode_step": 3},
    )

    assert canonical["instruction"] == "pick the object"
    np.testing.assert_array_equal(canonical["images"]["primary"], np.full((2, 3, 3), 10, dtype=np.uint8))
    np.testing.assert_array_equal(canonical["images"]["head"], np.full((2, 3, 3), 10, dtype=np.uint8))
    np.testing.assert_array_equal(canonical["images"]["left"], np.full((2, 3, 3), 20, dtype=np.uint8))
    np.testing.assert_array_equal(canonical["images"]["right"], np.full((2, 3, 3), 30, dtype=np.uint8))
    assert canonical["state"]["joint"] == list(np.arange(14, dtype=np.float32))
    assert canonical["meta"]["benchmark"] == "robotwin"
    assert canonical["meta"]["bimanual"] is True


def test_robotwin_action_from_flat_array_reorders_for_env_step() -> None:
    """Run test_robotwin_action_from_flat_array_reorders_for_env_step."""
    adapter = RoboTwinAdapter()
    raw_action = np.arange(14, dtype=np.float32)

    env_action = adapter.action_from_canonical({"actions": raw_action})

    np.testing.assert_array_equal(env_action, raw_action[ROBOTWIN_ACTION_REORDER])
    assert env_action.dtype == np.float32


def test_robotwin_delta_action_adds_current_joint_before_reorder() -> None:
    """Run test_robotwin_delta_action_adds_current_joint_before_reorder."""
    adapter = RoboTwinAdapter(action_mode="delta")
    current_joint = np.arange(14, dtype=np.float32)
    delta = np.ones(14, dtype=np.float32)

    env_action = adapter.action_from_canonical({"actions": delta}, {"current_joint": current_joint})

    np.testing.assert_array_equal(env_action, (current_joint + delta)[ROBOTWIN_ACTION_REORDER])


def test_robotwin_action_from_bimanual_fields() -> None:
    """Run test_robotwin_action_from_bimanual_fields."""
    adapter = RoboTwinAdapter(reorder_action=False)
    canonical_action = {
        "left": {"world_vector": [0, 1, 2], "rotation_delta": [3, 4, 5], "gripper": 6},
        "right": {"world_vector": [7, 8, 9], "rotation_delta": [10, 11, 12], "gripper": 13},
    }

    env_action = adapter.action_from_canonical(canonical_action)

    np.testing.assert_array_equal(env_action, np.arange(14, dtype=np.float32))


def test_robotwin_rejects_invalid_action_shape() -> None:
    """Run test_robotwin_rejects_invalid_action_shape."""
    adapter = RoboTwinAdapter()

    with pytest.raises(ValueError, match="14D"):
        adapter.action_from_canonical({"actions": [0.0] * 13})


def test_robotwin_eval_context_marks_script_success_oracle() -> None:
    """Run test_robotwin_eval_context_marks_script_success_oracle."""
    context = RoboTwinAdapter(task_name="pick_object", action_mode="abs").get_eval_context()

    assert context["benchmark"] == "robotwin"
    assert context["bimanual"] is True
    assert context["success_oracle_type"] == "script"
    assert context["has_state_fields"] == ["joint"]
    assert context["action_dim"] == 14


def test_robotwin_policy_duplicate_7d_bridge_is_explicit() -> None:
    """Run test_robotwin_policy_duplicate_7d_bridge_is_explicit."""
    client = ModelClient.__new__(ModelClient)
    client.action_bridge = "duplicate_7d"

    action = client._extract_robotwin_action(np.arange(7, dtype=np.float32))

    np.testing.assert_array_equal(
        action, np.concatenate([np.arange(7, dtype=np.float32), np.arange(7, dtype=np.float32)])
    )


def test_robotwin_policy_rejects_7d_without_bridge() -> None:
    """Run test_robotwin_policy_rejects_7d_without_bridge."""
    client = ModelClient.__new__(ModelClient)
    client.action_bridge = "strict_14d"

    with pytest.raises(ValueError, match="14D"):
        client._extract_robotwin_action(np.arange(7, dtype=np.float32))


def test_pi05_aloha_14d_delta_to_abs_and_roundtrip_masks() -> None:
    """openpi AbsoluteActions: joints relative, grippers absolute; flip mask is ±1."""
    state = np.arange(14, dtype=np.float32) * 0.1
    delta = np.ones(14, dtype=np.float32) * 0.05
    absolute = _delta_to_absolute_actions(delta, state)
    expected = np.where(_PI05_DELTA_JOINT_MASK, delta + state, delta)
    np.testing.assert_allclose(absolute, expected, rtol=1e-5)
    assert not _PI05_DELTA_JOINT_MASK[6] and not _PI05_DELTA_JOINT_MASK[13]
    assert _PI05_JOINT_FLIP_MASK.shape == (14,)
    assert set(np.unique(_PI05_JOINT_FLIP_MASK).tolist()) == {-1.0, 1.0}

    # encode(decode(x)) is not identity (gripper nonlinear), but joint flips cancel.
    env_state = np.linspace(0.02, 0.05, 14).astype(np.float32)
    pi_state = _adapt_to_pi_decode_state(env_state)
    assert pi_state.shape == (14,)
    env_action = _adapt_to_pi_encode_actions(pi_state)
    assert env_action.shape == (14,)


def test_ee6d_action_to_env_is_16d() -> None:
    """X-VLA 20D ee6d → RoboTwin 16D ee (pos+quat+grip × 2)."""
    # Interleaved rot6d of I: mat[:, :2].reshape(6) == [1, 0, 0, 1, 0, 0]
    identity_rot6d = np.asarray([1, 0, 0, 1, 0, 0], dtype=np.float32)
    raw = np.zeros(20, dtype=np.float32)
    raw[0:3] = [0.1, 0.2, 0.3]
    raw[3:9] = identity_rot6d
    raw[9] = 0.0  # open gripper (<= 0.7 → grip cmd +1)
    raw[10:13] = [0.4, 0.5, 0.6]
    raw[13:19] = identity_rot6d
    raw[19] = 0.9  # closed gripper (> 0.7 → grip cmd -1)

    env_action = _ee6d_action_to_env(raw)
    assert env_action.shape == (16,)
    np.testing.assert_allclose(env_action[0:3], [0.1, 0.2, 0.3], rtol=1e-5)
    np.testing.assert_allclose(env_action[8:11], [0.4, 0.5, 0.6], rtol=1e-5)
    assert env_action[7] == pytest.approx(1.0)  # open
    assert env_action[15] == pytest.approx(-1.0)  # closed


def test_robotwin_runner_writes_vla_eval_policy_config(tmp_path: pathlib.Path) -> None:
    """Run test_robotwin_runner_writes_vla_eval_policy_config."""
    args = argparse.Namespace(
        policy_ckpt_path="/ckpts/model.pt",
        task_name="adjust_bottle",
        task_config="demo_clean",
        ckpt_setting="loongforge_demo",
        seed=3,
        instruction_type="unseen",
        host="127.0.0.1",
        port=10093,
        unnorm_key="new_embodiment",
        action_mode="abs",
        no_action_reorder=False,
        max_steps=400,
        control_hz=10,
        disable_action_cache=True,
        return_action_chunk=False,
        disable_eval_video_log=True,
        output_dir=str(tmp_path / "run"),
    )
    config_path = tmp_path / "deploy_policy.yml"

    _write_deploy_policy(args, config_path)

    text = config_path.read_text(encoding="utf-8")
    assert "policy_name: loongforge.embodied.eval.bridges.robotwin_policy" in text
    assert "policy_ckpt_path: /ckpts/model.pt" in text
    assert "task_name: adjust_bottle" in text
    assert "host: 127.0.0.1" in text
    assert "port: 10093" in text
    assert "unnorm_key: new_embodiment" in text
    assert "disable_action_cache: true" in text
    assert "eval_video_log: false" in text
    assert "trace_path:" in text
    assert "artifacts/robotwin/adjust_bottle/demo_clean/trace.json" in text


def test_robotwin_policy_writes_trace_file(tmp_path: pathlib.Path) -> None:
    """Run test_robotwin_policy_writes_trace_file."""
    client = ModelClient.__new__(ModelClient)
    client.trace_path = tmp_path / "trace.json"
    client.trace_records = []
    client.episode_id = "robotwin/adjust_bottle/default"
    client.adapter = RoboTwinAdapter(task_name="adjust_bottle", action_mode="abs")

    client._record_trace(
        step=2,
        instruction="adjust bottle",
        joint=np.arange(14, dtype=np.float32),
        raw_action=np.ones(14, dtype=np.float32),
        output_action=np.ones(14, dtype=np.float32) * 2,
        env_action=np.ones(14, dtype=np.float32) * 3,
        response={"data": {"inference_latency_ms": 12.5}},
    )

    payload = json.loads(client.trace_path.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "robotwin"
    assert payload["task_name"] == "adjust_bottle"
    assert payload["steps"][0]["step"] == 2
    assert payload["steps"][0]["raw_action"] == [1.0] * 14
    assert payload["steps"][0]["output_action"] == [2.0] * 14
    assert payload["steps"][0]["env_action"] == [3.0] * 14


def test_robotwin_runner_builds_eval_policy_command() -> None:
    """Run test_robotwin_runner_builds_eval_policy_command."""
    args = argparse.Namespace(
        robotwin_python="/envs/robotwin/bin/python",
        robotwin_path="/workspace/RoboTwin",
        policy_ckpt_path="/ckpts/model.pt",
        task_name="adjust_bottle",
        task_config="demo_clean",
        ckpt_setting="loongforge_demo",
        seed=3,
    )

    command = build_command(args, pathlib.Path("/tmp/deploy_policy.yml"))

    assert command[:3] == ["/envs/robotwin/bin/python", "/workspace/RoboTwin/script/eval_policy.py", "--config"]
    assert "--policy_ckpt_path" not in command
    assert "loongforge.embodied.eval.bridges.robotwin_policy" in command
    assert "adjust_bottle" in command


def test_robotwin_step_limit_override_round_trip(tmp_path: pathlib.Path) -> None:
    """Run test_robotwin_step_limit_override_round_trip."""
    robotwin_path = tmp_path / "RoboTwin"
    config_dir = robotwin_path / "task_config"
    config_dir.mkdir(parents=True)
    step_limit_path = config_dir / "_eval_step_limit.yml"
    step_limit_path.write_text("adjust_bottle: 400\n", encoding="utf-8")

    original_text = _override_step_limit(robotwin_path, "adjust_bottle", 5)

    assert original_text == "adjust_bottle: 400\n"
    assert "adjust_bottle: 5" in step_limit_path.read_text(encoding="utf-8")
    step_limit_path.write_text(original_text, encoding="utf-8")
    assert step_limit_path.read_text(encoding="utf-8") == "adjust_bottle: 400\n"


def test_robotwin_eval_policy_override_round_trip(tmp_path: pathlib.Path) -> None:
    """Run test_robotwin_eval_policy_override_round_trip."""
    robotwin_path = tmp_path / "RoboTwin"
    script_dir = robotwin_path / "script"
    script_dir.mkdir(parents=True)
    eval_policy_path = script_dir / "eval_policy.py"
    eval_policy_path.write_text(
        "    st_seed = 100000 * (1 + seed)\n    test_num = 100\n    expert_check = True\n",
        encoding="utf-8",
    )

    original_text = _override_eval_policy_for_smoke(robotwin_path, 1, True, 100001)

    text = eval_policy_path.read_text(encoding="utf-8")
    assert original_text == "    st_seed = 100000 * (1 + seed)\n    test_num = 100\n    expert_check = True\n"
    assert "    st_seed = 100001\n" in text
    assert "    test_num = 1\n" in text
    assert "    expert_check = False\n" in text
    eval_policy_path.write_text(original_text, encoding="utf-8")
    assert (
        eval_policy_path.read_text(encoding="utf-8")
        == "    st_seed = 100000 * (1 + seed)\n    test_num = 100\n    expert_check = True\n"
    )


def test_robotwin_rate_from_result_reads_last_float() -> None:
    """Official `_result.txt` ends with suc_num/test_num as a float rate."""
    text = "Timestamp: 2026-07-21 21:45:26\n\nInstruction Type: unseen\n\n0.8\n"
    assert _robotwin_rate_from_result(text) == pytest.approx(0.8)
    assert _robotwin_rate_from_result(None) is None
    assert _robotwin_rate_from_result("") is None


def test_parse_official_episode_outcomes_from_ansi_log() -> None:
    """Per-episode success is inferred from cumulative suc counter + seed."""
    log = (
        "Success!\n"
        "adjust_bottle | policy | demo_clean | ckpt\n"
        "Success rate: \033[96m1\033[0m/\033[96m1\033[0m => \033[95m100.0%\033[0m, "
        "current seed: \033[90m100001\033[0m\n"
        "\n"
        "Success!\n"
        "Success rate: 2/2 => 100.0%, current seed: 100002\n"
        "Fail!\n"
        "Success rate: 2/3 => 66.7%, current seed: 100006\n"
        "Success!\n"
        "Success rate: 3/4 => 75.0%, current seed: 100008\n"
        "Success!\n"
        "Success rate: 4/5 => 80.0%, current seed: 100009\n"
    )
    eps = _parse_official_episode_outcomes(log)
    assert len(eps) == 5
    assert [e["seed"] for e in eps] == [100001, 100002, 100006, 100008, 100009]
    assert [e["success"] for e in eps] == [1, 1, 0, 1, 1]
    assert [e["episode_idx"] for e in eps] == [0, 1, 2, 3, 4]


def test_build_robotwin_records_expands_per_episode() -> None:
    """results.jsonl should be 1 row per official episode with 0/1 success."""
    args = argparse.Namespace(
        task_name="adjust_bottle",
        task_config="demo_clean",
        ckpt_setting="loongforge_xvla_robotwin2",
        max_steps=300,
        seed=0,
        start_seed_override=100001,
    )
    log = (
        "Success rate: 1/1 => 100.0%, current seed: 100001\n"
        "Success rate: 2/2 => 100.0%, current seed: 100002\n"
        "Success rate: 3/3 => 100.0%, current seed: 100005\n"
        "Success rate: 3/4 => 75.0%, current seed: 100006\n"
        "Success rate: 4/5 => 80.0%, current seed: 100008\n"
    )
    result_txt = "Timestamp: x\n\nInstruction Type: unseen\n\n0.8\n"
    records = _build_robotwin_records(
        args,
        returncode=0,
        artifacts={"result_txt": "/tmp/_result.txt", "official_eval_log": "/tmp/log", "trace_path": None, "videos": []},
        episode_time_sec=500.0,
        result_txt=result_txt,
        log_text=log,
    )
    assert len(records) == 5
    assert sum(int(r["success"]) for r in records) == 4
    assert all(r["success_rate"] == pytest.approx(0.8) for r in records)
    assert all(r["n_episodes"] == 5 for r in records)
    assert records[3]["success"] == 0
    assert records[3]["failure_reason"] == "not_successful"
    assert records[3]["seed"] == 100006
    assert records[3]["episode_id"] == "robotwin/adjust_bottle/demo_clean/seed=100006"
    assert records[0]["episode_time_sec"] == pytest.approx(100.0)


def test_build_robotwin_records_fallback_without_log() -> None:
    """When log has no episode lines, keep a single row from `_result.txt` rate."""
    args = argparse.Namespace(
        task_name="adjust_bottle",
        task_config="demo_clean",
        ckpt_setting="loongforge_xvla_robotwin2",
        max_steps=300,
        seed=0,
        start_seed_override=100001,
    )
    records = _build_robotwin_records(
        args,
        returncode=0,
        artifacts={},
        episode_time_sec=10.0,
        result_txt="0.0\n",
        log_text="no success rate lines",
    )
    assert len(records) == 1
    assert records[0]["success"] == 0
    assert records[0]["success_rate"] == pytest.approx(0.0)
    assert records[0]["seed"] == 100001
