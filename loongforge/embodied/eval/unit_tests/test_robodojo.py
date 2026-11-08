# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the RoboDojo adapter, XPolicyLab bridge, plugin deploy loop and runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib

import numpy as np
import pytest
import yaml

from loongforge.embodied.eval.adapters.robodojo import (
    ROBODOJO_CAMERAS,
    RoboDojoAdapter,
    joint14_to_action_dict,
    pack_joint_state,
)
from loongforge.embodied.eval.bridges import robodojo_policy
from loongforge.embodied.eval.bridges.robodojo_policy import (
    LINGBOT_VA_JOINT_CHANNELS,
    ChunkSchedule,
    RoboDojoPolicyModel,
    chunk_lengths_from_model,
)
from loongforge.embodied.eval.orchestrator import run as orchestrator_run
from loongforge.embodied.eval.orchestrator.runners import robodojo_runner

_EVAL_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _EVAL_ROOT.parents[2]
_PLUGIN_DIR = _EVAL_ROOT / "bridges" / "robodojo_xpolicylab"
_CONFIG_DIR = _REPO_ROOT / "examples" / "embodied" / "lingbot_va" / "eval" / "configs" / "robodojo"
# Optional: a RoboDojo checkout enables the parity tests against the official code.
_ROBODOJO_ROOT = pathlib.Path(os.environ.get("ROBODOJO_ROOT", "/nonexistent"))


def _obs(step: int = 0) -> dict:
    rng = np.random.default_rng(step)
    vision = {cam: {"color": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)} for cam in ROBODOJO_CAMERAS}
    state = {
        "left_arm_joint_state": np.arange(6, dtype=np.float64) * 0.1,
        "left_ee_joint_state": np.array([0.25]),
        "right_arm_joint_state": -np.arange(6, dtype=np.float64) * 0.1,
        "right_ee_joint_state": np.array([0.75]),
        "left_ee_pose": np.zeros(7),
        "right_ee_pose": np.zeros(7),
    }
    return {"vision": vision, "state": state, "instruction": "stack the bowls"}


def test_adapter_camera_order_and_joint_packing():
    obs = _obs()
    canonical = RoboDojoAdapter(task_name="stack_bowls").obs_to_canonical(
        obs, {"instruction": "stack the bowls", "episode_id": "e", "episode_step": 3}
    )
    images = canonical["images"]
    assert images["head"] is images["primary"]
    np.testing.assert_array_equal(images["head"], obs["vision"]["cam_head"]["color"])
    np.testing.assert_array_equal(images["left"], obs["vision"]["cam_left_wrist"]["color"])
    np.testing.assert_array_equal(images["right"], obs["vision"]["cam_right_wrist"]["color"])
    joint = canonical["state_raw"]["joint"]
    assert joint.dtype == np.float64 and joint.shape == (14,)
    np.testing.assert_array_equal(joint[:6], obs["state"]["left_arm_joint_state"])
    assert joint[6] == 0.25 and joint[13] == 0.75
    np.testing.assert_array_equal(joint[7:13], obs["state"]["right_arm_joint_state"])
    assert canonical["meta"]["episode_step"] == 3


def test_pack_joint_state_rejects_wrong_dims():
    state = _obs()["state"]
    state["left_ee_joint_state"] = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        pack_joint_state(state)


def test_action_from_canonical_selects_lingbot_va_joint_channels():
    raw = np.arange(30, dtype=np.float32) / 100.0
    action = RoboDojoAdapter(action_channel_ids=LINGBOT_VA_JOINT_CHANNELS).action_from_canonical(raw)
    assert action["left_arm_joint_state"] == pytest.approx(raw[14:20].tolist())
    assert action["left_ee_joint_state"] == pytest.approx([raw[28]])
    assert action["right_arm_joint_state"] == pytest.approx(raw[21:27].tolist())
    assert action["right_ee_joint_state"] == pytest.approx([raw[29]])


def test_lingbot_va_channels_match_official_client():
    model_py = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "LingBot_VA" / "model.py"
    if not model_py.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the parity check")
    text = model_py.read_text(encoding="utf-8")
    start = text.index("JOINT_CONTROL_INDICES")
    block = text[start:text.index("]", start) + 1]
    official = [int(x) for x in block.split("[", 1)[1].rstrip("]").replace("\n", " ").split(",") if x.strip()]
    assert official == LINGBOT_VA_JOINT_CHANNELS


def test_joint14_dict_matches_xpolicylab_unpack():
    process_data = _ROBODOJO_ROOT / "XPolicyLab" / "utils" / "process_data.py"
    if not process_data.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the parity check")
    import sys

    sys.path.insert(0, str(_ROBODOJO_ROOT))
    sys.path.insert(0, str(_ROBODOJO_ROOT / "XPolicyLab"))
    from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state

    vector = np.linspace(-1.0, 1.0, 14)
    official = unpack_robot_state(vector, "joint", get_robot_action_dim_info("arx_x5"))
    ours = joint14_to_action_dict(vector)
    assert set(official) == set(ours)
    for key, value in official.items():
        np.testing.assert_allclose(np.asarray(value), np.asarray(ours[key]))


def test_chunk_schedule_matches_lingbot_va_rollout():
    first, rest = chunk_lengths_from_model({"frame_chunk_size": 2, "action_per_frame": 12})
    assert (first, rest) == (12, 24)
    schedule = ChunkSchedule(first, rest)
    ends = [step for step in range(84) if schedule.is_chunk_end(step)]
    assert ends == [11, 35, 59, 83]


class _FakeEnv:
    """Records the env call order of a deploy loop; the episode ends after ``n_actions``."""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self.taken = 0
        self.calls = []

    def is_episode_end(self):
        self.calls.append("end?")
        return self.taken >= self.n_actions

    def get_obs(self):
        self.calls.append("obs")
        return {"t": self.taken}

    def take_action(self, action):
        self.calls.append(f"act{action['i']}")
        self.taken += 1


class _ChunkClient:
    """Official adapter style: get_action returns a whole chunk (``first``, then ``rest``)."""

    def __init__(self, first=12, rest=24):
        self.next_i = 0
        self.first = True
        self.lengths = (first, rest)

    def call(self, func_name, obs=None):
        if func_name != "get_action":
            return None
        n = self.lengths[0] if self.first else self.lengths[1]
        self.first = False
        chunk = [{"i": self.next_i + k} for k in range(n)]
        self.next_i += n
        return chunk


class _StepClient:
    """LoongForge bridge style: one action per call plus the chunk_end flag."""

    def __init__(self, first=12, rest=24):
        self.schedule = ChunkSchedule(first, rest)
        self.i = 0

    def call(self, func_name, obs=None):
        if func_name != "get_action":
            return None
        out = {"actions": [{"i": self.i}], "chunk_end": self.schedule.is_chunk_end(self.i)}
        self.i += 1
        return out


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("n_actions", [5, 12, 13, 36, 50])
def test_plugin_deploy_replays_official_env_call_order(n_actions):
    ours = _load_module(_PLUGIN_DIR / "deploy.py", "lf_robodojo_deploy")
    env_ours = _FakeEnv(n_actions)
    ours.eval_one_episode(env_ours, _StepClient())

    official_path = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "LingBot_VA" / "deploy.py"
    if not official_path.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the parity check")
    official = _load_module(official_path, "official_lingbot_va_deploy")
    env_official = _FakeEnv(n_actions)
    official.eval_one_episode(env_official, _ChunkClient())
    assert env_ours.calls == env_official.calls
    assert env_ours.taken == n_actions


class _FakePolicyClient:
    def __init__(self, host="127.0.0.1", port=0, timeout=0, ping_interval=20, ping_timeout=20):
        self.ping_interval = ping_interval
        self.resets = []
        self.calls = []

    def reset(self, episode_id):
        self.resets.append(episode_id)
        return {"ok": True}

    def predict_action(self, **payload):
        self.calls.append(payload)
        return {"ok": True, "data": {"actions": [np.arange(30, dtype=np.float32).tolist()]}}

    def close(self):
        pass


def _bridge(monkeypatch, tmp_path=None):
    import loongforge.embodied.eval.transport as transport

    monkeypatch.setattr(transport, "PolicyClient", _FakePolicyClient)
    cfg = {"task_name": "stack_bowls", "seed": 1, "first_chunk_len": 12, "chunk_len": 24, "lf_port": 1}
    if tmp_path is not None:
        cfg["trace_dir"] = str(tmp_path / "trace")
    return RoboDojoPolicyModel(cfg)


def test_bridge_episode_boundaries_and_payload(monkeypatch, tmp_path):
    model = _bridge(monkeypatch, tmp_path)
    assert model.disable_action_cache  # lingbot_va must see every step
    # RoboDojo resets twice before each episode.
    model.reset()
    model.reset()
    out0 = model.get_action(_obs(0))
    out1 = model.get_action(_obs(1))
    assert model.client.resets == ["robodojo/stack_bowls/seed1/ep0000"]
    assert model.client.ping_interval is None  # long chunk-boundary calls must not trip keepalive
    steps = [call["episode_step"] for call in model.client.calls]
    assert steps == [0, 1]
    first = model.client.calls[0]
    assert len(first["images"]) == 3 and first["instructions"] == ["stack the bowls"]
    assert first["state"] is None
    assert out0["chunk_end"] is False and len(out0["actions"]) == 1
    assert out1["actions"][0]["right_ee_joint_state"] == pytest.approx([29.0])
    model.reset()
    model.get_action(_obs(2))
    assert model.client.resets[-1] == "robodojo/stack_bowls/seed1/ep0001"
    assert model.client.calls[-1]["episode_step"] == 0
    lines = (tmp_path / "trace" / "ep0000.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [0, 1]


def test_bridge_update_obs_then_get_action(monkeypatch):
    model = _bridge(monkeypatch)
    model.update_obs(_obs(0))
    assert model.get_action()["actions"][0]["left_ee_joint_state"] == pytest.approx([28.0])
    with pytest.raises(RuntimeError):
        model.get_action()


def _runner_args(tmp_path: pathlib.Path) -> argparse.Namespace:
    config = yaml.safe_load((_CONFIG_DIR / "stack_bowls_smoke.yaml").read_text(encoding="utf-8"))
    config["run"]["output_dir"] = str(tmp_path / "out")
    args = robodojo_runner.build_argparser().parse_args([])
    return robodojo_runner._apply_config(args, config)


def test_runner_config_mapping(tmp_path):
    args = _runner_args(tmp_path)
    assert args.task_name == "stack_bowls" and args.seed == 0 and args.eval_num == 2
    assert (args.first_chunk_len, args.chunk_len) == (12, 24)
    assert args.device_id == 0 and args.port == 62101
    assert args.action_channel_ids == LINGBOT_VA_JOINT_CHANNELS
    args.policy_server_port = 62300
    cfg = robodojo_runner.write_policy_deploy_config(args, tmp_path / "deploy.yml")
    loaded = yaml.safe_load((tmp_path / "deploy.yml").read_text(encoding="utf-8"))
    assert loaded == cfg and loaded["policy_name"] == "LoongForge" and loaded["lf_port"] == 62101
    args.robodojo_root = str(tmp_path / "rd")
    args.additional_info = "ckpt_name=x,action_type=joint"
    args.run_id = "rid"
    command = robodojo_runner.build_sim_command(args)
    assert command[:2] == ["bash", str((tmp_path / "rd").resolve() / "scripts" / "eval_policy.sh")]
    assert command[command.index("--policy_name") + 1] == "LoongForge"
    assert command[command.index("--seed") + 1] == "0"
    assert robodojo_runner.result_dir(args).parts[-7:] == (
        "eval_result", "RoboDojo", "stack_bowls", "LoongForge", "arx_x5", "0_ckpt_name=x,action_type=joint", "rid"
    )


def test_runner_records_from_official_result(tmp_path):
    args = _runner_args(tmp_path)
    args.run_id = "rid"
    result = {
        "success_rate": 0.5,
        "score": 70.0,
        "eval_time": 2,
        "details": {"1": {"layout_id": 4, "success": False, "score": 0.4}, "0": {"layout_id": 3, "success": True, "score": 1.0}},
    }
    records = robodojo_runner.build_records(args, result, 0, 10.0)
    assert [r["episode_idx"] for r in records] == [0, 1]
    assert [r["success"] for r in records] == [1, 0]
    assert [r["score"] for r in records] == [1.0, 0.4]
    assert records[1]["layout_id"] == 4 and records[1]["failure_reason"] == "not_successful"
    assert np.mean([r["score"] for r in records]) * 100 == pytest.approx(result["score"])


def test_plugin_link_refuses_foreign_entry(tmp_path):
    policy_dir = tmp_path / "XPolicyLab" / "policy"
    policy_dir.mkdir(parents=True)
    link = robodojo_runner.ensure_plugin_link(tmp_path)
    assert link.resolve() == _PLUGIN_DIR.resolve()
    assert robodojo_runner.ensure_plugin_link(tmp_path) == link  # idempotent
    link.unlink()
    (policy_dir / "LoongForge").mkdir()
    with pytest.raises(FileExistsError):
        robodojo_runner.ensure_plugin_link(tmp_path)


def test_plugin_files_are_import_free():
    # The sim client imports the package and deploy.py; they must not pull LoongForge.
    for name in ("__init__.py", "deploy.py"):
        text = (_PLUGIN_DIR / name).read_text(encoding="utf-8")
        assert "import loongforge" not in text and "from loongforge" not in text
    meta = yaml.safe_load((_PLUGIN_DIR / "deploy.yml").read_text(encoding="utf-8"))
    assert meta["eval_batch"] is False and meta["protocol"] == "ws"


def test_orchestrator_dispatches_robodojo():
    assert orchestrator_run._benchmark_family({"benchmark": {"name": "robodojo"}}) == "robodojo"


def test_example_configs_match_official_lingbot_va_settings():
    smoke = yaml.safe_load((_CONFIG_DIR / "stack_bowls_smoke.yaml").read_text(encoding="utf-8"))
    model = smoke["model"]
    assert model["used_action_channel_ids"] == list(range(30))
    assert len(model["norm_q01"]) == len(model["norm_q99"]) == 30
    assert (model["height"], model["width"], model["camera_layout"]) == (256, 256, "width_concat")
    assert (model["frame_chunk_size"], model["action_per_frame"]) == (2, 12)
    full = yaml.safe_load((_CONFIG_DIR / "robodojo_all_tasks.yaml").read_text(encoding="utf-8"))
    tasks = full["sweep"]["benchmark.task_name"]
    assert len(tasks) == len(set(tasks)) == 54
    assert full["sweep"]["benchmark.seed"] == [0, 1, 2]
    assert full["model"] == model

    official_cfg = (
        _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "LingBot_VA" / "lingbot_va" / "wan_va" / "configs"
        / "va_robotwin30_train_cfg.py"
    )
    if not official_cfg.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the quantile parity check")
    import ast

    for node in ast.parse(official_cfg.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "attr", "") == "norm_stat":
            stats = ast.literal_eval(node.value)
            assert stats["q01"] == model["norm_q01"] and stats["q99"] == model["norm_q99"]
            break
    else:
        raise AssertionError("norm_stat not found in the official config")


def test_bridge_wiring_is_known():
    assert set(robodojo_policy._BRIDGE_WIRING) == {"lingbot_va_joint30", "pi05_joint14", "xvla_ee20"}
    assert json.loads(json.dumps(LINGBOT_VA_JOINT_CHANNELS)) == LINGBOT_VA_JOINT_CHANNELS


# ── X-VLA (official XPolicyLab/policy/X_VLA) ─────────────────────────────
def _official_xvla_funcs():
    """Pure helpers of the official X_VLA adapter, taken from source (no torch / xvla imports)."""
    import ast

    model_py = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "X_VLA" / "model.py"
    if not model_py.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the X-VLA parity check")
    from scipy.spatial.transform import Rotation

    wanted = {"quat_to_rotate6d", "rotate6d_to_quat", "build_xvla_proprio", "action_chunk_to_ee_dict_list"}
    tree = ast.parse(model_py.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    ns = {"np": np, "R": Rotation, "Any": object}
    exec(compile(ast.Module(nodes, []), str(model_py), "exec"), ns)
    return ns


def _ee_obs(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = _obs(seed)
    for side in ("left", "right"):
        quat = rng.normal(size=4)
        obs["state"][f"{side}_ee_pose"] = np.concatenate([rng.normal(size=3), quat / np.linalg.norm(quat)])
        obs["state"][f"{side}_ee_joint_state"] = np.array([rng.uniform()])
    return obs


@pytest.mark.parametrize("seed", range(8))
def test_xvla_robodojo_proprio_matches_official(seed):
    from loongforge.embodied.eval.payload_builders.xvla import build_robodojo_proprio

    official = _official_xvla_funcs()
    obs = _ee_obs(seed)
    canonical = RoboDojoAdapter(action_space="ee20").obs_to_canonical(obs, {"instruction": "x"})
    ours = build_robodojo_proprio(canonical["state_raw"])
    ref = official["build_xvla_proprio"](obs)
    assert ours.dtype == np.float32 and ours.shape == (20,)
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.parametrize("seed", range(8))
def test_xvla_ee20_action_dict_matches_official(seed):
    official = _official_xvla_funcs()
    chunk = np.random.default_rng(100 + seed).normal(size=(30, 20)).astype(np.float32)
    ref = official["action_chunk_to_ee_dict_list"](chunk)
    adapter = RoboDojoAdapter(action_space="ee20")
    for row, expected in zip(chunk, ref):
        ours = adapter.action_from_canonical(row)
        assert set(ours) == set(expected)
        for key in expected:
            np.testing.assert_array_equal(np.asarray(ours[key], dtype=np.float32), expected[key])


@pytest.mark.parametrize("n_actions", [5, 30, 31, 61, 100])
def test_plugin_deploy_replays_official_xvla_env_call_order(n_actions):
    ours = _load_module(_PLUGIN_DIR / "deploy.py", "lf_robodojo_deploy")
    env_ours = _FakeEnv(n_actions)
    ours.eval_one_episode(env_ours, _StepClient(30, 30))

    official_path = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "X_VLA" / "deploy.py"
    if not official_path.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the parity check")
    official = _load_module(official_path, "official_xvla_deploy")
    env_official = _FakeEnv(n_actions)
    official.eval_one_episode(env_official, _ChunkClient(30, 30))
    assert env_ours.calls == env_official.calls
    assert env_ours.taken == n_actions


class _FakeXVLAClient(_FakePolicyClient):
    def predict_action(self, **payload):
        self.calls.append(payload)
        return {"ok": True, "data": {"actions": [np.linspace(-1, 1, 20, dtype=np.float32).tolist()]}}


def test_bridge_xvla_payload_and_ee_action(monkeypatch):
    import loongforge.embodied.eval.transport as transport

    monkeypatch.setattr(transport, "PolicyClient", _FakeXVLAClient)
    model = RoboDojoPolicyModel(
        {"task_name": "build_tower", "action_bridge": "xvla_ee20", "first_chunk_len": 30, "chunk_len": 30, "lf_port": 1}
    )
    assert not model.disable_action_cache  # one inference per chunk, the server cache serves the rest
    model.reset()
    outs = [model.get_action(_ee_obs(i)) for i in range(31)]
    call = model.client.calls[0]
    assert len(call["images"]) == 1 and call["images"][0] is not None  # head camera only
    assert call["domain_id"] == 0 and np.asarray(call["state"]).shape == (20,)
    assert [i for i, out in enumerate(outs) if out["chunk_end"]] == [29]
    action = outs[0]["actions"][0]
    assert set(action) == {"left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"}
    assert len(action["left_ee_pose"]) == 7 and np.linalg.norm(action["left_ee_pose"][3:]) == pytest.approx(1.0, abs=1e-5)


def test_example_xvla_config_matches_official_adapter():
    cfg = yaml.safe_load(
        (_REPO_ROOT / "examples/embodied/xvla/eval/configs/robodojo/put_bottles_into_dustbin.yaml").read_text()
    )
    assert cfg["benchmark"]["action_bridge"] == "xvla_ee20"
    assert cfg["benchmark"]["first_chunk_len"] == cfg["benchmark"]["chunk_len"] == cfg["model"]["num_actions"] == 30
    assert cfg["server"]["use_bf16"] is False and cfg["server"]["chunk_execute_steps"] < 0
    deploy = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "X_VLA" / "deploy.yml"
    if not deploy.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the deploy.yml check")
    official = yaml.safe_load(deploy.read_text())
    assert official["action_type"] == "ee" and int(official["domain_id"]) == 0 and int(official["steps"]) == 10


# ── pi0.5 (official XPolicyLab/policy/Pi_05, openpi aloha protocol, adapt_to_pi=False) ──
def test_pi05_robodojo_decoder_delta_to_absolute_with_chunk_anchor():
    from loongforge.embodied.eval.action_decoders import build_action_decoder

    dec = build_action_decoder("pi05_aloha_robodojo")
    state0 = np.arange(14, dtype=np.float32)
    delta = np.full((1, 32), 0.5, dtype=np.float32)
    out0 = dec(delta, {"pi_state": state0, "is_fresh_chunk": True})[0]
    expect = state0 + 0.5
    expect[[6, 13]] = 0.5  # grippers absolute (make_bool_mask(6, -1, 6, -1))
    np.testing.assert_allclose(out0, expect)
    # a cached step keeps the chunk's anchor even though the env state moved
    out1 = dec(delta, {"pi_state": state0 + 100, "is_fresh_chunk": False})[0]
    np.testing.assert_allclose(out1, expect)
    out2 = dec(delta, {"pi_state": state0 + 100, "is_fresh_chunk": True})[0]
    assert out2[0] == pytest.approx(100.5)


def test_pi05_robodojo_matches_official_openpi_output_transforms():
    """AbsoluteActions + AlohaOutputs(adapt_to_pi=False) of the vendored openpi, from source."""
    import ast

    tf_py = _ROBODOJO_ROOT / "XPolicyLab/policy/Pi_05/openpi/src/openpi/transforms.py"
    if not tf_py.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the openpi parity check")
    tree = ast.parse(tf_py.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_bool_mask"]
    ns = {}
    exec(compile(ast.Module(nodes, []), str(tf_py), "exec"), ns)
    mask = np.asarray(ns["make_bool_mask"](6, -1, 6, -1))
    from loongforge.embodied.eval.action_decoders.joint import _PI05_DELTA_JOINT_MASK

    np.testing.assert_array_equal(mask, _PI05_DELTA_JOINT_MASK)
    cfg = (_ROBODOJO_ROOT / "XPolicyLab/policy/Pi_05/openpi/src/openpi/training/config.py").read_text()
    start = cfg.index('name="pi05_base_aloha_full_sim_arx-x5_seed_0"')
    block = cfg[start:cfg.index("TrainConfig(", start)]
    assert "LeRobotAlohaDataConfig(" in block and "adapt_to_pi" not in block  # default False


class _FakePi05Client(_FakePolicyClient):
    def predict_action(self, **payload):
        self.calls.append(payload)
        latency = 5.0 if len(self.calls) % 50 == 1 else None  # server cache: fresh every 50 steps
        return {"ok": True, "data": {"actions": [np.zeros(32, dtype=np.float32).tolist()],
                                     "inference_latency_ms": latency}}


def test_bridge_pi05_payload_and_joint_action(monkeypatch):
    import loongforge.embodied.eval.transport as transport

    monkeypatch.setattr(transport, "PolicyClient", _FakePi05Client)
    model = RoboDojoPolicyModel(
        {"task_name": "stack_bowls", "action_bridge": "pi05_joint14", "first_chunk_len": 50, "chunk_len": 50, "lf_port": 1}
    )
    assert not model.disable_action_cache
    model.reset()
    outs = [model.get_action(_obs(i)) for i in range(51)]
    call = model.client.calls[0]
    assert len(call["images"]) == 3
    np.testing.assert_allclose(call["state"], RoboDojoAdapter().obs_to_canonical(_obs(0), {"instruction": "x"})["state_raw"]["joint"])
    assert [i for i, o in enumerate(outs) if o["chunk_end"]] == [49]
    # zero delta -> the chunk anchor's joints (state of step 0), grippers 0
    act = outs[10]["actions"][0]
    assert act["left_arm_joint_state"] == pytest.approx(list(np.arange(6) * 0.1))
    assert act["left_ee_joint_state"] == pytest.approx([0.0])


def test_example_pi05_config_matches_official_adapter():
    cfg = yaml.safe_load((_REPO_ROOT / "examples/embodied/pi05/eval/configs/robodojo/stack_bowls.yaml").read_text())
    assert cfg["benchmark"]["action_bridge"] == "pi05_joint14"
    assert cfg["benchmark"]["first_chunk_len"] == cfg["benchmark"]["chunk_len"] == cfg["model"]["action_horizon"] == 50
    assert cfg["model"]["action_dim"] == cfg["model"]["state_dim"] == 14
    assert cfg["server"]["chunk_execute_steps"] == 0 and cfg["server"]["eval_normalization_mode"] == "q99"
