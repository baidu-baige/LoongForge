# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pytest

from loongforge.embodied.eval.action_decoders.ee6d import (
    _robotwin_ee6d_row_to_ee as _ee6d_action_to_env,
)
from loongforge.embodied.eval.action_decoders.joint import (
    _PI05_DELTA_JOINT_MASK,
    _PI05_JOINT_FLIP_MASK,
    adapt_to_pi_encode_actions as _adapt_to_pi_encode_actions,
    delta_to_absolute_actions as _delta_to_absolute_actions,
)
from loongforge.embodied.eval.adapters.robotwin import (
    ROBOTWIN_ACTION_REORDER,
    RoboTwinAdapter,
)
from loongforge.embodied.eval.payload_builders.pi05 import (
    adapt_to_pi_decode_state as _adapt_to_pi_decode_state,
)
from loongforge.embodied.eval.bridges.robotwin_policy import ModelClient
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
    client.action_bridge = "ee6d_dual"
    client.adapter = RoboTwinAdapter(task_name="adjust_bottle", action_mode="abs")

    client._record_trace(
        step=2,
        instruction="adjust bottle",
        joint=np.arange(14, dtype=np.float32),
        raw_action=np.ones(20, dtype=np.float32),
        env_action=np.ones(16, dtype=np.float32) * 3,
        response={"data": {"inference_latency_ms": 12.5}},
    )

    payload = json.loads(client.trace_path.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "robotwin"
    assert payload["task_name"] == "adjust_bottle"
    assert payload["steps"][0]["step"] == 2
    assert payload["steps"][0]["raw_action"] == [1.0] * 20
    assert payload["steps"][0]["env_action"] == [3.0] * 16
    assert payload["steps"][0]["action_bridge"] == "ee6d_dual"


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


def test_worker_script_skips_unstable_seeds(tmp_path: pathlib.Path) -> None:
    """The worker copy handles UnStableError in the eval loop, not just expert_check."""
    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    robotwin = tmp_path / "RoboTwin"
    (robotwin / "script").mkdir(parents=True)
    source = (
        "import sys\n"
        'sys.path.append("./description/utils")\n'
        "def eval_policy():\n"
        "    while True:\n"
        "        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
        "        TASK_ENV.set_instruction(instruction=instruction)\n"
    )
    (robotwin / "script" / "eval_policy.py").write_text(source, encoding="utf-8")

    args = argparse.Namespace(
        worker_local_files=True,
        robotwin_path=str(robotwin),
        output_dir=str(tmp_path / "out"),
        ckpt_setting="test",
        step_limit_override=None,
        start_seed_override=None,
        test_num_override=None,
        disable_expert_check=False,
        disable_eval_video_log=False,
        seed_stride=8,
    )
    script = robotwin_runner._worker_eval_script(args)
    text = script.read_text(encoding="utf-8")
    assert "except UnStableError" in text
    assert "succ_seed -= 1" in text  # undo the pre-setup increment
    assert "now_seed += 8" in text  # jump a full round, not into the next shard
    assert "now_seed += 1" not in text
    assert "continue" in text


def test_crashed_run_without_result_is_not_a_fake_episode() -> None:
    """An official crash before any episode must not become a 100-step failure row."""
    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    args = argparse.Namespace(
        task_name="adjust_bottle",
        task_config="demo_clean",
        ckpt_setting="test",
        max_steps=100,
        seed=0,
        start_seed_override=100003,
    )
    records = robotwin_runner._build_robotwin_records(
        args,
        returncode=1,
        artifacts={},
        episode_time_sec=54.0,
        result_txt=None,
        log_text="UnStableError: Objects is unstable in seed(100003)",
    )
    assert records == []


def test_shard_test_num_wins_over_episodes_per_task() -> None:
    """A coordinator shard sets test_num; the run-wide episodes_per_task must not override it."""
    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    args = robotwin_runner.build_argparser().parse_args([])
    config = {
        "benchmark": {"task_name": "adjust_bottle", "episodes_per_task": 8, "test_num": 1},
        "server": {},
    }
    args = robotwin_runner._apply_config(args, config)
    assert args.test_num_override == 1


def test_apply_config_maps_chunk_flags() -> None:
    """benchmark.return_action_chunk must reach the bridge; it was previously dropped."""
    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    args = robotwin_runner.build_argparser().parse_args([])
    config = {
        "benchmark": {
            "return_action_chunk": True,
            "disable_action_cache": True,
            "chunk_execute_steps": 24,
        },
        "server": {},
    }
    args = robotwin_runner._apply_config(args, config)
    assert args.return_action_chunk is True
    assert args.disable_action_cache is True
    assert args.chunk_execute_steps == 24


def test_chunk_rows_are_consumed_in_order() -> None:
    """The bridge holds the client-owned chunk and advances one row per env step."""
    from loongforge.embodied.eval.bridges.robotwin_policy import ModelClient

    model = object.__new__(ModelClient)
    model._chunk = np.arange(6, dtype=np.float32).reshape(3, 2)
    model._chunk_cursor = 0
    model._chunk_len = 3
    assert model._take_chunk_row().tolist() == [[0.0, 1.0]]
    assert model._take_chunk_row().tolist() == [[2.0, 3.0]]
    assert model._take_chunk_row().tolist() == [[4.0, 5.0]]
    assert model._chunk_cursor == 3


def test_deploy_policy_records_chunk_depth(tmp_path: pathlib.Path) -> None:
    """The bridge can only execute 24 steps if the deploy file carries the depth."""
    import yaml

    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    args = robotwin_runner.build_argparser().parse_args([])
    args.robotwin_path = "/workspace/RoboTwin"
    args.output_dir = str(tmp_path)
    args.task_name = "adjust_bottle"
    args.task_config = "demo_clean"
    args.policy_ckpt_path = "/tmp/ckpt"
    args.return_action_chunk = True
    args.chunk_execute_steps = 24

    path = tmp_path / "deploy_policy.yml"
    robotwin_runner._write_deploy_policy(args, path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["return_action_chunk"] is True
    assert data["chunk_execute_steps"] == 24


def test_needs_obs_follows_the_buffered_chunk() -> None:
    """Only steps that will hit the server need a rendered frame."""
    # ModelClient.__init__ dials the policy server, so exercise the predicate
    # on a bare instance: it only reads the chunk cursor and the cache mode.
    chunked = ModelClient.__new__(ModelClient)
    chunked.return_action_chunk = True
    chunked._chunk = None
    chunked._chunk_cursor = 0
    chunked._chunk_len = 0
    assert chunked.needs_obs() is True  # nothing buffered yet

    chunked._chunk = np.zeros((4, 14), dtype=np.float32)
    chunked._chunk_cursor = 1
    chunked._chunk_len = 4
    assert chunked.needs_obs() is False  # replaying a buffered row

    chunked._chunk_cursor = 4
    assert chunked.needs_obs() is True  # chunk exhausted -> request

    server_cache = ModelClient.__new__(ModelClient)
    server_cache.return_action_chunk = False
    server_cache._chunk = np.zeros((4, 14), dtype=np.float32)
    server_cache._chunk_cursor = 1
    server_cache._chunk_len = 4
    assert server_cache.needs_obs() is True  # one request per step


def test_worker_script_skips_intra_chunk_render(tmp_path: pathlib.Path) -> None:
    """The opt-in switch replaces get_obs() with a proprio-only observation."""
    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    robotwin = tmp_path / "RoboTwin"
    (robotwin / "script").mkdir(parents=True)
    source = (
        "import numpy as np\n"
        "def eval_policy(task_name, TASK_ENV, args, model):\n"
        "        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:\n"
        "            observation = TASK_ENV.get_obs()\n"
        "            eval_func(TASK_ENV, model, observation)\n"
    )
    (robotwin / "script" / "eval_policy.py").write_text(source, encoding="utf-8")

    def run(skip: bool) -> str:
        args = argparse.Namespace(
            worker_local_files=True,
            robotwin_path=str(robotwin),
            output_dir=str(tmp_path / f"out_{skip}"),
            ckpt_setting="test",
            step_limit_override=None,
            start_seed_override=None,
            test_num_override=None,
            disable_expert_check=False,
            disable_eval_video_log=False,
            seed_stride=1,
            skip_intra_chunk_obs=skip,
        )
        return robotwin_runner._worker_eval_script(args).read_text(encoding="utf-8")

    enabled = run(True)
    assert "model.needs_obs()" in enabled
    assert "_vla_proprio_observation(TASK_ENV)" in enabled
    assert "def _vla_proprio_observation(TASK_ENV):" in enabled

    disabled = run(False)
    assert "needs_obs" not in disabled
    assert "observation = TASK_ENV.get_obs()" in disabled


def test_worker_script_early_stop_patch(tmp_path: pathlib.Path) -> None:
    """The opt-in switch injects the check_stable early-stop + instrumentation."""
    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    robotwin = tmp_path / "RoboTwin"
    (robotwin / "script").mkdir(parents=True)
    source = (
        "import numpy as np\n"
        "def eval_policy(task_name, TASK_ENV, args, model):\n"
        "        pass\n"
    )
    (robotwin / "script" / "eval_policy.py").write_text(source, encoding="utf-8")

    def run(enabled: bool) -> str:
        args = argparse.Namespace(
            worker_local_files=True,
            robotwin_path=str(robotwin),
            output_dir=str(tmp_path / f"out_{enabled}"),
            ckpt_setting="test",
            step_limit_override=None,
            start_seed_override=None,
            test_num_override=None,
            disable_expert_check=False,
            disable_eval_video_log=False,
            seed_stride=1,
            skip_intra_chunk_obs=False,
            settle_early_stop=enabled,
        )
        return robotwin_runner._worker_eval_script(args).read_text(encoding="utf-8")

    enabled = run(True)
    assert "check_stable early-stop" in enabled
    assert "settled_at_step" in enabled  # instrumentation
    assert "_WINDOW, _CAP, _EPS" in enabled
    assert "_BT._init_task_env_, _BT.check_stable = _patched_init, _patched_check" in enabled
    # Regression: _poses/_settled_at are instance helpers; every call must pass
    # self or check_stable raises on the first window (seen as repeated
    # "error occurs !" with no instrumentation / stuck workers).
    import re
    assert "for _p, _q in zip(_poses(self), ref):" in enabled
    assert "_ref = _poses(self)" in enabled
    assert not re.search(r"(?<!self\.)_poses\(\)", enabled)
    assert "_drift = _settled_at(self, _ref)" in enabled

    disabled = run(False)
    assert "check_stable early-stop" not in disabled
    assert "settled_at_step" not in disabled


def test_worker_script_offline_expert_check_modes(tmp_path: pathlib.Path) -> None:
    """Seed-cache scan/consume rewrite only the worker script, never RoboTwin."""
    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    robotwin = tmp_path / "RoboTwin"
    (robotwin / "script").mkdir(parents=True)
    source = (
        "import numpy as np\n"
        "def main(x):\n"
        "    model = get_model(usr_args)\n"
        "def eval_policy(task_name, TASK_ENV, args, model, st_seed, test_num=1):\n"
        "    expert_check = True\n"
        "    while succ_seed < test_num:\n"
        '        render_freq = args["render_freq"]\n'
        "        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
        '        episode_info_list = [episode_info["info"]]\n'
    )
    (robotwin / "script" / "eval_policy.py").write_text(source, encoding="utf-8")

    def gen(**kw) -> str:
        fields = dict(
            worker_local_files=True,
            robotwin_path=str(robotwin),
            output_dir=str(tmp_path / f"out_{abs(hash(str(kw)))}"),
            ckpt_setting="t",
            step_limit_override=None,
            start_seed_override=None,
            test_num_override=None,
            disable_expert_check=False,
            disable_eval_video_log=True,
            seed_stride=1,
            skip_intra_chunk_obs=False,
            settle_early_stop=False,
            task_name="adjust_bottle",
        )
        fields.update(kw)
        args = argparse.Namespace(**fields)
        return robotwin_runner._worker_eval_script(args).read_text(encoding="utf-8")

    scan = gen(offline_expert_check_scan=str(tmp_path / "cache"))
    assert "_S_SCAN_DIR" in scan
    assert "offline_expert_check scan: expert pass only" in scan  # get_model skipped
    assert "_s_dump_scan(" in scan
    assert "expert_check = False" not in scan

    consume = gen(offline_expert_check=str(tmp_path / "cache"))
    assert "_S_CACHE_PATH" in consume
    assert "expert_check = False" in consume
    assert "now_seed = _S_SEEDS[_idx]" in consume
    assert "_idx = _S_OFFSET + succ_seed" in consume
    assert "cache exhausted" in consume
    assert "CACHE STALE" in consume

    # Consume mode uses the explicit cache index, not start_seed (which stays a
    # seed value); a leftover start_seed=100000 must not shift the index.
    consume_offset = gen(
        offline_expert_check=str(tmp_path / "cache"),
        start_seed_override=100000,
        offline_expert_check_offset=4,
    )
    assert "    _S_OFFSET = 4\n" in consume_offset
    assert "_S_OFFSET = 100000" not in consume_offset

    plain = gen()
    assert "_S_SCAN_DIR" not in plain and "_S_CACHE_PATH" not in plain

    # Every generated worker script must at least be syntactically valid.
    for label, src in (("scan", scan), ("consume", consume), ("plain", plain)):
        compile(src, f"<{label}>", "exec")


def test_official_step_limit_lookup(tmp_path: pathlib.Path) -> None:
    """use_official_step_limits keeps per-task horizons from RoboTwin's YAML."""
    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    config_dir = tmp_path / "task_config"
    config_dir.mkdir()
    (config_dir / "_eval_step_limit.yml").write_text(
        "adjust_bottle: 400\nopen_microwave: 1700\n", encoding="utf-8"
    )
    args = argparse.Namespace(task_name="open_microwave", robotwin_path=str(tmp_path))
    assert robotwin_runner._official_step_limit(args) == 1700

    args.task_name = "not_a_task"
    assert robotwin_runner._official_step_limit(args) is None

    args.robotwin_path = str(tmp_path / "missing")
    args.task_name = "open_microwave"
    assert robotwin_runner._official_step_limit(args) is None


def test_records_use_trace_length_for_early_success(tmp_path: pathlib.Path) -> None:
    """`steps` must be the executed count, not the horizon cap."""
    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"steps": [{"step": i} for i in range(37)]}), encoding="utf-8")
    args = argparse.Namespace(
        task_name="adjust_bottle",
        task_config="demo_clean",
        ckpt_setting="test",
        max_steps=400,
        seed=0,
        start_seed_override=None,
    )
    records = robotwin_runner._build_robotwin_records(
        args,
        returncode=0,
        artifacts={"trace_path": str(trace)},
        episode_time_sec=10.0,
        result_txt=None,
        log_text="",
    )
    assert len(records) == 1
    assert records[0]["steps"] == 37

def test_merge_offline_expert_check_parts(tmp_path: pathlib.Path) -> None:
    """Parts from residue-class scans merge, dedupe, sort and truncate."""
    import pickle

    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    cache = tmp_path / "cache"
    cache.mkdir()
    parts = {
        0: [1000, 1004],
        1: [1001, 1005],
        2: [1002, 1006],
        3: [1003],
    }
    for i, seeds in parts.items():
        data = {
            "schema": 1, "task": "adjust_bottle", "task_config": "demo_clean",
            "n": len(seeds), "seeds": seeds,
            "episode_info": {str(s): {"{A}": f"obj{s}"} for s in seeds},
        }
        (cache / f"adjust_bottle.part{i}.pkl").write_bytes(pickle.dumps(data))

    merged = robotwin_runner.merge_offline_expert_check_scan(
        {"benchmark": {"offline_expert_check_scan": str(cache),
                       "offline_expert_check_scan_shards": 4,
                       "episodes_per_task": 6}}
    )
    assert merged == {"adjust_bottle": 6}
    out = pickle.loads((cache / "adjust_bottle.pkl").read_bytes())
    assert out["seeds"] == [1000, 1001, 1002, 1003, 1004, 1005]
    assert out["episode_info"]["1005"] == {"{A}": "obj1005"}
