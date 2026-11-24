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
    assert set(robodojo_policy._BRIDGE_WIRING) == {"lingbot_va_joint30", "pi05_joint14", "xvla_ee20", "fastwam_joint14", "gr00t_n17_joint14", "gr00t_n16_joint14", "dreamzero_joint14", "pi05_sf_joint14"}
    assert set(robodojo_policy._EPISODE_END_CHECK) <= set(robodojo_policy._BRIDGE_WIRING)
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


def test_pi05_openpi_resize_with_pad_matches_openpi_layout():
    from loongforge.embodied.eval.payload_builders.pi05 import openpi_resize_with_pad

    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    out = openpi_resize_with_pad(img, 224)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert (out[:28] == 0).all() and (out[196:] == 0).all() and out[28:196].any()  # 168 rows, centered


# ── FastWAM (official XPolicyLab/policy/FastWAM, RoboTwin deploy_policy reused) ──
class _StepClientChunkCheck(_StepClient):
    """Bridge style with ``end_check: chunk`` (FastWAM)."""

    def call(self, func_name, obs=None):
        out = super().call(func_name, obs)
        if out is not None:
            out["end_check"] = "chunk"
        return out


@pytest.mark.parametrize("n_actions", [5, 23, 24, 25, 48, 60])
def test_plugin_deploy_replays_official_fastwam_env_call_order(n_actions):
    ours = _load_module(_PLUGIN_DIR / "deploy.py", "lf_robodojo_deploy")
    env_ours = _FakeEnv(n_actions)
    ours.eval_one_episode(env_ours, _StepClientChunkCheck(24, 24))

    official_path = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "FastWAM" / "deploy.py"
    if not official_path.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the parity check")
    official = _load_module(official_path, "official_fastwam_deploy")
    env_official = _FakeEnv(n_actions)
    official.eval_one_episode(env_official, _ChunkClient(24, 24))
    # is_episode_end only at chunk ends; the rest of a chunk is still sent (no-ops in the real env)
    assert env_ours.calls == env_official.calls
    assert "end?" not in env_ours.calls[2:2 + 2 * 23]


def _official_fastwam_standardize():
    import ast

    model_py = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "FastWAM" / "model.py"
    if not model_py.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the FastWAM parity check")
    import cv2

    tree = ast.parse(model_py.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_standardize_rgb"]
    ns = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(nodes, []), str(model_py), "exec"), ns)
    return ns["_standardize_rgb"]


@pytest.mark.parametrize("shape", [(480, 640, 3), (240, 320, 3), (720, 1280, 3)])
def test_fastwam_standardize_rgb_matches_official(shape):
    from loongforge.embodied.eval.payload_builders.fastwam import standardize_rgb

    official = _official_fastwam_standardize()
    img = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    np.testing.assert_array_equal(standardize_rgb(img, 320, 240), official(img))


class _FakeFastWAMClient(_FakePolicyClient):
    def predict_action(self, **payload):
        self.calls.append(payload)
        latency = 5.0 if len(self.calls) % 24 == 1 else None  # server cache: fresh every 24 steps
        return {"ok": True, "data": {"actions": [np.linspace(-1, 1, 14, dtype=np.float32).tolist()],
                                     "inference_latency_ms": latency}}


def test_bridge_fastwam_payload_and_joint_action(monkeypatch):
    import loongforge.embodied.eval.transport as transport

    monkeypatch.setattr(transport, "PolicyClient", _FakeFastWAMClient)
    model = RoboDojoPolicyModel(
        {"task_name": "stack_bowls", "action_bridge": "fastwam_joint14", "first_chunk_len": 24, "chunk_len": 24, "lf_port": 1}
    )
    assert not model.disable_action_cache  # one inference per chunk, the server cache serves the rest
    model.reset()
    outs = [model.get_action(_obs(i)) for i in range(25)]
    call = model.client.calls[0]
    assert [np.asarray(v).shape for v in call["images"]] == [(240, 320, 3)] * 3  # head, left, right
    assert call["instructions"] == [
        "A video recorded from a robot's point of view executing the following instruction: stack the bowls"
    ]
    # the official adapter casts the packed state to float32 (``.astype(np.float32)``)
    np.testing.assert_array_equal(call["state"], pack_joint_state(_obs(0)["state"]).astype(np.float32))
    assert [i for i, o in enumerate(outs) if o["chunk_end"]] == [23]
    assert {o["end_check"] for o in outs} == {"chunk"}
    expected = joint14_to_action_dict(np.linspace(-1, 1, 14, dtype=np.float32))  # absolute joints, no decoder
    for key, value in expected.items():
        np.testing.assert_allclose(outs[0]["actions"][0][key], value)
    if (_ROBODOJO_ROOT / "XPolicyLab" / "utils" / "process_data.py").is_file():
        import sys

        sys.path.insert(0, str(_ROBODOJO_ROOT))
        sys.path.insert(0, str(_ROBODOJO_ROOT / "XPolicyLab"))
        from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state

        ref = pack_robot_state(_obs(0), "joint", get_robot_action_dim_info("arx_x5"), source_type="obs", state_type="state")
        np.testing.assert_array_equal(call["state"], np.asarray(ref, dtype=np.float32))


def test_example_fastwam_config_matches_official_adapter():
    cfg = yaml.safe_load(
        (_REPO_ROOT / "examples/embodied/fastwam/eval/configs/robodojo/put_bottles_into_dustbin.yaml").read_text()
    )
    bench, model, server = cfg["benchmark"], cfg["model"], cfg["server"]
    assert bench["action_bridge"] == "fastwam_joint14"
    assert bench["first_chunk_len"] == bench["chunk_len"] == 24 and server["chunk_execute_steps"] == 0
    assert model["camera_layout"] == "robotwin_t" and model["norm_mode"] == "z-score" and model["variant"] == "uncond"
    assert model["noise_seed"] == bench["seed"] and model["redirect_common_files"] is True
    deploy = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "FastWAM" / "deploy.yml"
    if not deploy.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the deploy.yml check")
    official = yaml.safe_load(deploy.read_text())
    assert int(official["replan_steps"]) == bench["chunk_len"]
    assert int(official["num_inference_steps"]) == model["num_inference_steps"]
    assert official["mixed_precision"] == "bf16" and model["dtype"] == "bfloat16" and server["use_bf16"] is True
    assert official["sigma_shift"] is None and model["sigma_shift"] == 5.0  # scheduler infer_shift
    assert float(official["text_cfg_scale"]) == 1.0 and official["rand_device"] == "cpu"
    sim_task = yaml.safe_load(
        (_ROBODOJO_ROOT / "XPolicyLab/policy/FastWAM/FastWAM/configs/model/fastwam.yaml").read_text()
    )
    assert sim_task["action_scheduler"]["infer_shift"] == model["sigma_shift"]
    assert sim_task["model_id"] == model["model_id"] and sim_task["tokenizer_model_id"] == model["tokenizer_model_id"]


# ── GR00T-N1.7 (official XPolicyLab/policy/GR00T_N17) ──
def _official_gr00t_funcs():
    import ast

    model_py = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "GR00T_N17" / "model.py"
    if not model_py.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the GR00T_N17 parity check")
    wanted = {"_ensure_hwc_uint8", "_extract_image", "_to_rgb_hwc", "_as_1d", "_extract_prompt",
              "_pack_arm_state", "_encode_observation", "_gr00t_action_to_env"}
    tree = ast.parse(model_py.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign) and any(
        getattr(t, "id", "") == "VIDEO_KEY_CANDIDATES" for t in n.targets)]
    ns = {"np": np, "Any": object}
    exec(compile(ast.Module(nodes, []), str(model_py), "exec"), ns)
    return ns


@pytest.mark.parametrize("n_actions", [5, 16, 17, 40])
def test_plugin_deploy_replays_official_gr00t_env_call_order(n_actions):
    ours = _load_module(_PLUGIN_DIR / "deploy.py", "lf_robodojo_deploy")
    env_ours = _FakeEnv(n_actions)
    ours.eval_one_episode(env_ours, _StepClient(16, 16))
    official_path = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "GR00T_N17" / "deploy.py"
    if not official_path.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the parity check")
    official = _load_module(official_path, "official_gr00t_deploy")
    env_official = _FakeEnv(n_actions)
    official.eval_one_episode(env_official, _ChunkClient(16, 16))
    assert env_ours.calls == env_official.calls
    assert env_ours.taken == n_actions


class _FakeGr00tClient(_FakePolicyClient):
    def predict_action(self, **payload):
        self.calls.append(payload)
        latency = 5.0 if len(self.calls) % 16 == 1 else None
        return {"ok": True, "data": {"actions": [np.linspace(-1, 1, 14, dtype=np.float32).tolist()],
                                     "inference_latency_ms": latency}}


def test_bridge_gr00t_payload_matches_official_encode(monkeypatch):
    import loongforge.embodied.eval.transport as transport

    monkeypatch.setattr(transport, "PolicyClient", _FakeGr00tClient)
    model = RoboDojoPolicyModel(
        {"task_name": "stack_bowls", "action_bridge": "gr00t_n17_joint14", "first_chunk_len": 16, "chunk_len": 16, "lf_port": 1}
    )
    assert not model.disable_action_cache and model.episode_end_check == "action"
    model.reset()
    outs = [model.get_action(_obs(i)) for i in range(17)]
    call = model.client.calls[0]
    assert [i for i, o in enumerate(outs) if o["chunk_end"]] == [15]
    assert call["instructions"] == ["stack the bowls"]
    expected = joint14_to_action_dict(np.linspace(-1, 1, 14, dtype=np.float32))  # absolute joints, no decoder
    for key, value in expected.items():
        np.testing.assert_allclose(outs[0]["actions"][0][key], value)

    official = _official_gr00t_funcs()
    enc = official["_encode_observation"](_obs(0), "default")
    ref_state = np.concatenate([enc["state"]["left_arm"][0, 0], enc["state"]["right_arm"][0, 0]])
    assert call["state"].dtype == np.float32
    np.testing.assert_array_equal(call["state"], ref_state)
    import albumentations as A

    from loongforge.embodied.data.datasets.gr00t_n1_7.transforms.image_augmentations import LetterBoxPad

    # same order as the official video keys; letterboxed to 640x640 exactly as the official
    # (vendored) eval transform's first step LetterBoxPad does
    letterbox = A.Compose([LetterBoxPad()])
    for ours, key in zip(call["images"], ["front", "left_wrist", "right_wrist"]):
        native = enc["video"][key][0, 0]
        assert np.asarray(ours).shape == (640, 640, 3)
        np.testing.assert_array_equal(np.asarray(ours), letterbox(image=native)["image"])
    assert enc["language"]["annotation.human.task_description"] == [call["instructions"]]
    chunk = np.linspace(-1, 1, 28, dtype=np.float32).reshape(2, 14)
    ref_actions = official["_gr00t_action_to_env"]({"left_arm": chunk[None, :, :7], "right_arm": chunk[None, :, 7:]}, "joint")
    for step, ref in enumerate(ref_actions):
        ours = joint14_to_action_dict(chunk[step])
        assert set(ours) == set(ref)
        for key in ref:
            np.testing.assert_allclose(np.asarray(ours[key]), ref[key])


def test_gr00t_robodojo_modality_matches_official_checkpoint_layout():
    from loongforge.embodied.data.datasets.gr00t_n1_7.transforms.gr00t_transform import (
        EMBODIMENT_STAT_CONFIGS,
        EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
        MODALITY_CONFIGS,
        STATISTICS_TAG_ALIASES,
    )

    cfg = MODALITY_CONFIGS["robodojo_arx_x5"]
    assert EMBODIMENT_TAG_TO_PROJECTOR_INDEX["robodojo_arx_x5"] == EMBODIMENT_TAG_TO_PROJECTOR_INDEX["new_embodiment"]
    assert STATISTICS_TAG_ALIASES["robodojo_arx_x5"] == "new_embodiment"
    meta = EMBODIMENT_STAT_CONFIGS["robodojo_arx_x5"]["modality_meta"]
    assert meta["state"] == meta["action"] == {"left_arm": {"start": 0, "end": 7}, "right_arm": {"start": 7, "end": 14}}
    ckpt = os.environ.get("GR00T_N17_ROBODOJO_CKPT")
    if not ckpt:
        pytest.skip("set GR00T_N17_ROBODOJO_CKPT to the checkpoint dir for the processor_config check")
    official = json.loads((pathlib.Path(ckpt) / "processor_config.json").read_text())
    official = official["processor_kwargs"]["modality_configs"]["new_embodiment"]
    for modality in ("video", "state", "action", "language"):
        assert list(cfg[modality].delta_indices) == official[modality]["delta_indices"]
        assert list(cfg[modality].modality_keys) == official[modality]["modality_keys"]
    reps = [str(getattr(c.rep, "name", c.rep)) for c in cfg["action"].action_configs]
    assert reps == [c["rep"] for c in official["action"]["action_configs"]]
    embodiment_ids = json.loads((pathlib.Path(ckpt) / "embodiment_id.json").read_text())
    assert embodiment_ids["new_embodiment"] == EMBODIMENT_TAG_TO_PROJECTOR_INDEX["robodojo_arx_x5"]


def test_example_gr00t_config_matches_official_adapter():
    cfg = yaml.safe_load(
        (_REPO_ROOT / "examples/embodied/gr00t_n1_7/eval/configs/robodojo/put_bottles_into_dustbin.yaml").read_text()
    )
    assert cfg["benchmark"]["action_bridge"] == "gr00t_n17_joint14"
    assert cfg["benchmark"]["first_chunk_len"] == cfg["benchmark"]["chunk_len"] == 16
    assert cfg["server"]["embodiment_tag"] == "robodojo_arx_x5" and cfg["server"]["chunk_execute_steps"] == 0
    assert cfg["server"]["use_bf16"] is True and "action_horizon" not in cfg["model"]


def test_gr00t_official_vendored_eval_transform_always_letterboxes():
    """The vendored GR00T eval transform starts with LetterBoxPad unconditionally (no flag)."""
    src = _ROBODOJO_ROOT / "XPolicyLab/policy/GR00T_N17/gr00t_n17/gr00t/model/gr00t_n1d7/image_augmentations.py"
    if not src.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the GR00T_N17 source check")
    text = src.read_text()
    body = text[text.index("def build_image_transformations_albumentations"):]
    assert "letter_box_transform" not in body.split("):", 1)[0]  # no switch in the signature
    eval_block = body[body.index("eval_transform = A.Compose("):]
    assert eval_block.index("LetterBoxPad()") < eval_block.index("SmallestMaxSize")


# ── Spatial_Forcing (official XPolicyLab/policy/Spatial_Forcing, open_sf = openpi + SF) ──
def test_spatial_forcing_official_config_is_absolute_pi05():
    cfg_py = _ROBODOJO_ROOT / "XPolicyLab/policy/Spatial_Forcing/open_sf/src/openpi/external_config/pi05sf_jax_robodojo_v21_offcache.py"
    if not cfg_py.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the Spatial_Forcing config check")
    text = cfg_py.read_text()
    assert "Pi0Config(pi05=True)" in text and "LeRobotAlohaDataConfig(" in text
    assert "adapt_to_pi=False" in text and "use_delta_joint_actions=False" in text
    deploy = yaml.safe_load((_ROBODOJO_ROOT / "XPolicyLab/policy/Spatial_Forcing/deploy.yml").read_text())
    assert deploy["train_config_name"] == "pi05sf_jax_robodojo_v21_offcache" and deploy["action_type"] == "joint"
    official = _ROBODOJO_ROOT / "XPolicyLab/policy/Spatial_Forcing/deploy.py"
    assert official.read_text() == (_ROBODOJO_ROOT / "XPolicyLab/policy/Pi_05/deploy.py").read_text()


def test_bridge_pi05_sf_payload_and_absolute_joint_action(monkeypatch):
    import loongforge.embodied.eval.transport as transport

    monkeypatch.setattr(transport, "PolicyClient", _FakePi05Client)
    sf = RoboDojoPolicyModel(
        {"task_name": "stack_bowls", "action_bridge": "pi05_sf_joint14", "first_chunk_len": 50, "chunk_len": 50, "lf_port": 1}
    )
    sf.reset()
    out = sf.get_action(_obs(0))
    call = sf.client.calls[0]
    assert len(call["images"]) == 3 and np.asarray(call["images"][0]).shape == (224, 224, 3)
    # zero raw action -> zero joints (absolute, no delta anchoring), unlike pi05_joint14
    act = out["actions"][0]
    assert act["left_arm_joint_state"] == pytest.approx([0.0] * 6)
    assert act["right_ee_joint_state"] == pytest.approx([0.0])
    assert sf.episode_end_check == "action"


def test_example_pi05_sf_config_matches_official_adapter():
    cfg = yaml.safe_load(
        (_REPO_ROOT / "examples/embodied/pi05/eval/configs/robodojo/stack_bowls_spatial_forcing.yaml").read_text()
    )
    assert cfg["benchmark"]["action_bridge"] == "pi05_sf_joint14"
    assert cfg["benchmark"]["first_chunk_len"] == cfg["benchmark"]["chunk_len"] == cfg["model"]["action_horizon"] == 50
    assert cfg["server"]["chunk_execute_steps"] == 0 and cfg["server"]["eval_normalization_mode"] == "q99"


def test_bridge_gr00t_n16_payload_matches_official_n17_encode(monkeypatch):
    """No official N1.6 plugin: N1.6 follows the official GR00T_N17 adapter's state / camera order;
    the N1.6 model letterboxes in its own eval image transform, so the views go out env-native."""
    import loongforge.embodied.eval.transport as transport

    monkeypatch.setattr(transport, "PolicyClient", _FakeGr00tClient)
    model = RoboDojoPolicyModel(
        {"task_name": "stack_bowls", "action_bridge": "gr00t_n16_joint14", "first_chunk_len": 16, "chunk_len": 16, "lf_port": 1}
    )
    assert not model.disable_action_cache and model.episode_end_check == "action"
    model.reset()
    outs = [model.get_action(_obs(i)) for i in range(17)]
    call = model.client.calls[0]
    assert [i for i, o in enumerate(outs) if o["chunk_end"]] == [15]
    expected = joint14_to_action_dict(np.linspace(-1, 1, 14, dtype=np.float32))
    for key, value in expected.items():
        np.testing.assert_allclose(outs[0]["actions"][0][key], value)
    enc = _official_gr00t_funcs()["_encode_observation"](_obs(0), "default")
    ref_state = np.concatenate([enc["state"]["left_arm"][0, 0], enc["state"]["right_arm"][0, 0]])
    assert call["state"].dtype == np.float32
    np.testing.assert_array_equal(call["state"], ref_state)
    for ours, key in zip(call["images"], ["front", "left_wrist", "right_wrist"]):
        np.testing.assert_array_equal(np.asarray(ours), enc["video"][key][0, 0])
    assert enc["language"]["annotation.human.task_description"] == [call["instructions"]]


def test_gr00t_n16_robodojo_modality_matches_n17():
    from loongforge.embodied.data.datasets.gr00t_n1_6.transforms import utils as n16
    from loongforge.embodied.data.datasets.gr00t_n1_7.transforms import gr00t_transform as n17

    assert n16.EMBODIMENT_TAG_TO_PROJECTOR_INDEX["robodojo_arx_x5"] == n16.EMBODIMENT_TAG_TO_PROJECTOR_INDEX["new_embodiment"]
    assert n16.STATISTICS_TAG_ALIASES == n17.STATISTICS_TAG_ALIASES
    assert n16.EMBODIMENT_STAT_CONFIGS["robodojo_arx_x5"]["modality_meta"] == n17.ROBODOJO_ARX_X5_MODALITY_META
    ours, ref = n16.MODALITY_CONFIGS["robodojo_arx_x5"], n17.MODALITY_CONFIGS["robodojo_arx_x5"]
    for modality in ("video", "state", "action", "language"):
        assert list(ours[modality].delta_indices) == list(ref[modality].delta_indices)
        assert list(ours[modality].modality_keys) == list(ref[modality].modality_keys)
    reps = lambda cfg: [str(getattr(c.rep, "name", c.rep)) for c in cfg["action"].action_configs]
    assert reps(ours) == reps(ref) == ["RELATIVE", "RELATIVE"]


def test_gr00t_n16_eval_transform_starts_with_letterbox():
    """The transform GrootN1d6Policy.predict_action applies (modeling_gr00t_n1_6.py)."""
    from loongforge.embodied.data.datasets.gr00t_n1_6.transforms.eagle3_model.image_augmentations import (
        build_image_transformations_albumentations,
    )

    _, eval_tf = build_image_transformations_albumentations(None, None, None, None, 256, 0.95)
    assert type(eval_tf.transforms[0]).__name__ == "LetterBoxPad"


def test_example_gr00t_n16_config_matches_n17_adapter():
    cfg = yaml.safe_load(
        (_REPO_ROOT / "examples/embodied/gr00t_n1_6/eval/configs/robodojo/put_bottles_into_dustbin.yaml").read_text()
    )
    assert cfg["benchmark"]["action_bridge"] == "gr00t_n16_joint14"
    assert cfg["benchmark"]["first_chunk_len"] == cfg["benchmark"]["chunk_len"] == 16
    assert cfg["server"]["embodiment_tag"] == "robodojo_arx_x5" and cfg["server"]["chunk_execute_steps"] == 0
    assert "action_horizon" not in cfg["model"] and cfg["model"]["action_dim"] == 14


# ── DreamZero-AgiBot (official XPolicyLab/policy/DreamZero) ──
_DZ_DIR = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "DreamZero"


def _official_dreamzero_ns():
    """Pure helpers of the official adapter and its vendored groot transforms, taken from source."""
    import ast

    import torch
    from einops import rearrange

    model_py = _DZ_DIR / "model.py"
    if not model_py.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the DreamZero parity check")
    ns = {"np": np, "torch": torch, "Any": object, "rearrange": rearrange}
    tree = ast.parse(model_py.read_text())
    wanted = {"_pad_or_trim", "_stack_recent_frames", "_agibot_arm_slot_targets",
              "_xpolicylab_packed_to_agibot_state", "_agibot_to_xpolicylab_packed"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign) and any(
        getattr(t, "id", "") in {"AGIBOT_STATE_DIM", "DEFAULT_IMAGE_SIZE", "_AGIBOT_J3_LOCKED_PER_ARM_INDEX",
                                 "_ARX_X5_ARM_TO_AGIBOT_ARM_SLOTS"} for t in n.targets)]
    exec(compile(ast.Module(nodes, []), str(model_py), "exec"), ns)
    sa_py = _DZ_DIR / "dreamzero/groot/vla/data/transform/state_action.py"
    sa = ast.parse(sa_py.read_text())
    exec(compile(ast.Module([n for n in sa.body if isinstance(n, ast.ClassDef) and n.name == "Normalizer"], []),
                 str(sa_py), "exec"), ns)
    dt_py = _DZ_DIR / "dreamzero/groot/vla/model/dreamzero/transform/dreamzero_cotrain.py"
    dt = ast.parse(dt_py.read_text())
    cls = next(n for n in dt.body if isinstance(n, ast.ClassDef) and n.name == "DreamTransform")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_video")
    ns["EmbodimentTag"] = argparse.Namespace(OXE_DROID="oxe_droid")
    exec(compile(ast.Module([fn], []), str(dt_py), "exec"), ns)
    ns["dreamzero_cotrain_src"] = dt_py.read_text()
    return ns


_DZ_ROBOT_INFO = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def _dz_stats(rng):
    from loongforge.embodied.eval.factories.dreamzero_robodojo import AGIBOT_ACTION_KEYS, AGIBOT_STATE_KEYS

    def block(keys):
        out = {}
        for key, dim in keys:
            q01 = rng.normal(size=dim) - 1.0
            q99 = q01 + rng.uniform(0.5, 2.0, size=dim)
            if key == "waist_lift":
                q99 = q01.copy()  # degenerate dim: the official normalizer passes it through
            out[key] = {"q01": q01.tolist(), "q99": q99.tolist()}
        return out

    return {"state": block(AGIBOT_STATE_KEYS), "action": block(AGIBOT_ACTION_KEYS)}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dreamzero_agibot_state_and_action_mapping_matches_official(seed):
    import torch

    from loongforge.embodied.eval.factories import dreamzero_robodojo as dz

    official = _official_dreamzero_ns()
    rng = np.random.default_rng(seed)
    joint14 = rng.normal(size=14).astype(np.float32)
    ref20 = official["_xpolicylab_packed_to_agibot_state"](joint14, _DZ_ROBOT_INFO)
    ours20 = dz.joint14_to_agibot_state(joint14)
    np.testing.assert_array_equal(ours20, ref20)

    raw = _dz_stats(rng)
    stats = dz.AgibotQ99(raw)
    ref_parts = []
    for key, x in dz.split_keys(ref20, dz.AGIBOT_STATE_KEYS).items():
        norm = official["Normalizer"]("q99", {k: list(v) for k, v in raw["state"][key].items()})
        ref_parts.append(norm.forward(torch.from_numpy(x[None].astype(np.float32))).numpy()[0])
    np.testing.assert_array_equal(stats.normalize_state(ours20), np.concatenate(ref_parts))

    chunk = rng.uniform(-1, 1, size=(48, 32)).astype(np.float32)
    ours_actions = stats.unnormalize_action(chunk[:, :22])
    joints = dz.agibot_chunk_to_joint14(ours_actions, ours20, 24)
    assert joints.shape == (24, 14)
    raw_state = dz.split_keys(ref20, dz.AGIBOT_STATE_KEYS)
    for step in (0, 7, 23):
        ref = {}
        for key, x in dz.split_keys(chunk[step:step + 1, :22], dz.AGIBOT_ACTION_KEYS).items():
            norm = official["Normalizer"]("q99", {k: list(v) for k, v in raw["action"][key].items()})
            value = norm.inverse(torch.from_numpy(x)).numpy()[0]
            ref[key] = value + raw_state[key] if key in dz.AGIBOT_RELATIVE_ACTION_KEYS else value
        packed = official["_agibot_to_xpolicylab_packed"](
            ref["left_arm_joint_position"], ref["right_arm_joint_position"],
            ref["left_effector_position"], ref["right_effector_position"], "joint", _DZ_ROBOT_INFO)
        np.testing.assert_allclose(joints[step], packed, rtol=0, atol=1e-6)


def test_dreamzero_frame_history_and_grid_match_official():
    import torch
    import torchvision.transforms.v2 as T

    from loongforge.embodied.eval.factories import dreamzero_robodojo as dz

    official = _official_dreamzero_ns()
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(6)]
    for n in (1, 3, 6):
        np.testing.assert_array_equal(dz.stack_recent_frames(frames[:n], 4),
                                      official["_stack_recent_frames"](frames[:n], 4))
    # VideoToTensor -> VideoCrop(eval CenterCrop, 0.95 of the 640x480 metadata) -> VideoResize(176x320,
    # linear, antialias) -> VideoToNumpy, with the torchvision v2 classes the vendored transform builds.
    views = [dz.stack_recent_frames(frames[i:i + 4], 4) for i in range(3)]
    ref_views = []
    for v in views:
        x = torch.from_numpy(v).to(torch.float32).div(255.0).permute(0, 3, 1, 2)
        x = T.Resize((176, 320), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(T.CenterCrop((456, 608))(x))
        ref_views.append((x.permute(0, 2, 3, 1) * 255).to(torch.uint8).numpy())
    ours = [dz.agibot_view_transform(v) for v in views]
    for a, b in zip(ours, ref_views):
        np.testing.assert_array_equal(a, b)
    # ConcatTransform order [top_head, hand_left, hand_right] -> (T, V, H, W, C) -> DreamTransform grid.
    stub = argparse.Namespace(embodiment_tag="agibot")
    ref_grid = official["_prepare_video"](stub, {"video": np.stack(ref_views, axis=1)})
    ref_grid = np.transpose(ref_grid[0], (0, 2, 3, 1))  # (1, T, C, 2H, 2W) -> (T, 2H, 2W, C)
    np.testing.assert_array_equal(dz.agibot_grid(*ours), ref_grid)
    assert ref_grid.shape == (4, 352, 640, 3)


def test_dreamzero_agibot_prompt_matches_official_template():
    from loongforge.embodied.eval.factories import dreamzero_robodojo as dz

    src = _official_dreamzero_ns()["dreamzero_cotrain_src"]
    head = '"A multi-view video shows that a robot " + str(item).lower() + " '
    start = src.index(head) + len(head)
    middle = src[start:src.index('" + str(item).lower()', start)]
    instruction = "Pick up the  Bottles and throw them into the dustbin."
    lowered = instruction.lower()
    expected = " ".join(("A multi-view video shows that a robot " + lowered + " " + middle + lowered).split())
    assert dz.agibot_prompt(instruction) == expected


class _FakeDreamZeroInfer:
    def __init__(self):
        self.calls, self.resets = [], 0

    def predict(self, **kw):
        self.calls.append(kw)
        return np.zeros((1, 48, 32), dtype=np.float32)

    def reset(self):
        self.resets += 1


def test_dreamzero_robodojo_eval_model_schedule_and_session():
    from loongforge.embodied.eval.factories import dreamzero_robodojo as dz

    rng = np.random.default_rng(5)
    stats = dz.AgibotQ99(_dz_stats(rng))
    infer = _FakeDreamZeroInfer()
    model = dz.DreamZeroRoboDojoEvalModel(infer, stats, negative_prompt="neg")
    state = rng.normal(size=14).astype(np.float32)
    views = lambda i: [np.full((480, 640, 3), 10 * i + k, dtype=np.uint8) for k in range(3)]
    for step in range(30):
        out = model.predict_action([views(step)], ["Stack"], state=state, episode_id="ep0", episode_step=step)
        assert out.shape == (1, 14)
    assert len(infer.calls) == 2  # steps 0 and 24
    first, second = infer.calls
    assert first["apply_eval_image_transform"] is False and first["video"].shape == (4, 352, 640, 3)
    assert (first["video"][:, :176, :320] == 0).all()  # four copies of the first head frame (value 0)
    assert second["video"][-1, 0, 0, 0] == 240 and second["video"][0, 0, 0, 0] == 210  # steps 21..24
    assert first["state"].shape == (64,) and not first["state"][20:].any()
    # zero normalized action -> q99 midpoint; relative keys add the raw state.
    mid = {k: (np.asarray(v[0]) + np.asarray(v[1])) / 2 for k, v in stats.action.items()}
    raw20 = dz.joint14_to_agibot_state(state)
    raw = dz.split_keys(raw20, dz.AGIBOT_STATE_KEYS)
    expected = dz.agibot_to_joint14(mid["left_arm_joint_position"] + raw["left_arm_joint_position"],
                                    mid["right_arm_joint_position"] + raw["right_arm_joint_position"],
                                    mid["left_effector_position"], mid["right_effector_position"])
    np.testing.assert_allclose(out[0], expected, atol=1e-6)
    model.predict_action([views(0)], ["Stack"], state=state, episode_id="ep1")
    assert infer.resets == 0 and len(infer.calls) == 3  # new episode: buffers cleared, session kept
    reset_model = dz.DreamZeroRoboDojoEvalModel(infer, stats, "neg", reset_session_on_new_episode=True)
    reset_model.predict_action([views(0)], ["Stack"], state=state, episode_id="a")
    reset_model.predict_action([views(0)], ["Stack"], state=state, episode_id="b")
    assert infer.resets == 1


class _FakeDreamZeroClient(_FakePolicyClient):
    def predict_action(self, **payload):
        self.calls.append(payload)
        return {"ok": True, "data": {"actions": [np.linspace(-1, 1, 14, dtype=np.float32).tolist()],
                                     "inference_latency_ms": 5.0}}


def test_bridge_dreamzero_payload_every_step_and_chunk_end(monkeypatch):
    import loongforge.embodied.eval.transport as transport

    monkeypatch.setattr(transport, "PolicyClient", _FakeDreamZeroClient)
    model = RoboDojoPolicyModel(
        {"task_name": "stack_bowls", "action_bridge": "dreamzero_joint14", "first_chunk_len": 24, "chunk_len": 24, "lf_port": 1}
    )
    assert model.disable_action_cache and model.episode_end_check == "action"
    model.reset()
    outs = [model.get_action(_obs(i)) for i in range(49)]
    assert [i for i, o in enumerate(outs) if o["chunk_end"]] == [23, 47]
    assert len(model.client.calls) == 49 and all(c["disable_action_cache"] for c in model.client.calls)
    call = model.client.calls[0]
    for ours, cam in zip(call["images"], ["cam_head", "cam_left_wrist", "cam_right_wrist"]):
        np.testing.assert_array_equal(ours, _obs(0)["vision"][cam]["color"])
    ref_state = np.concatenate([np.arange(6) * 0.1, [0.25], -np.arange(6) * 0.1, [0.75]]).astype(np.float32)
    np.testing.assert_array_equal(call["state"], ref_state)
    assert call["state"].dtype == np.float32 and call["instructions"] == ["stack the bowls"]


def test_example_dreamzero_config_matches_official_adapter():
    cfg = yaml.safe_load(
        (_REPO_ROOT / "examples/embodied/dreamzero/eval/configs/robodojo/put_bottles_into_dustbin.yaml").read_text()
    )
    model = cfg["model"]
    assert cfg["benchmark"]["action_bridge"] == "dreamzero_joint14"
    assert cfg["benchmark"]["first_chunk_len"] == cfg["benchmark"]["chunk_len"] == model["eval_action_horizon"] == 24
    assert model["eval_protocol"] == "robodojo_agibot" and model["backbone_variant"] == "wan21_14b"
    assert model["eval_video_history"] == 4 and model["eval_num_inference_timesteps"] == 4
    assert model["reset_session_on_new_episode"] is False
    official = yaml.safe_load((_DZ_DIR / "deploy.yml").read_text()) if (_DZ_DIR / "deploy.yml").is_file() else None
    if official:
        assert official["action_horizon"] == model["eval_action_horizon"] and official["video_history"] == model["eval_video_history"]
        assert official["skip_img_transform"] is False and official["native_dojo_action"] is False


# ── eval_batch: several envs per Isaac Sim process (LoongForgeBatch plugin) ──
_BATCH_PLUGIN_DIR = _EVAL_ROOT / "bridges" / "robodojo_xpolicylab_batch"


class _FakeBatchEnv:
    """Batched env: env i ends after ``lengths[i]`` actions; records the call order."""

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.taken = [0] * len(lengths)
        self.calls = []

    def _running(self):
        return [i for i, n in enumerate(self.lengths) if self.taken[i] < n]

    def is_episode_end(self):
        self.calls.append("end?")
        return not self._running()

    def get_running_env_idx_list(self):
        return self._running()

    def get_obs_batch(self, env_idx_list=None):
        self.calls.append(("obs", tuple(env_idx_list)))
        return [{"env_idx": i, "t": self.taken[i]} for i in env_idx_list]

    def take_action_batch(self, actions, env_idx_list):
        self.calls.append(("act", tuple(env_idx_list), tuple(a["i"] for a in actions)))
        for i in env_idx_list:
            if self.taken[i] < self.lengths[i]:
                self.taken[i] += 1


class _BatchChunkClient:
    """Official-style client: ``get_action_batch`` returns ``chunk`` actions per env."""

    def __init__(self, chunk=24):
        self.chunk = chunk
        self.step = {}

    def call(self, func_name, obs=None):
        if func_name != "get_action_batch":
            return None
        out = []
        for idx in obs:
            start = self.step.get(idx, 0)
            out.append([{"i": start + k} for k in range(self.chunk)])
            self.step[idx] = start + self.chunk
        return out


@pytest.mark.parametrize("lengths", [[5, 30, 48], [24, 24], [60, 1, 25, 49]])
def test_batch_plugin_replays_official_fastwam_batch_call_order(lengths):
    ours = _load_module(_BATCH_PLUGIN_DIR / "deploy.py", "lf_robodojo_batch_deploy")
    env_ours = _FakeBatchEnv(lengths)
    ours.eval_one_episode_batch(env_ours, _BatchChunkClient(24))
    assert env_ours.taken == lengths
    assert yaml.safe_load((_BATCH_PLUGIN_DIR / "deploy.yml").read_text())["eval_batch"] is True
    official_path = _ROBODOJO_ROOT / "XPolicyLab" / "policy" / "FastWAM" / "deploy.py"
    if not official_path.is_file():
        pytest.skip("set ROBODOJO_ROOT to a RoboDojo checkout for the parity check")
    official = _load_module(official_path, "official_fastwam_batch_deploy")
    env_official = _FakeBatchEnv(lengths)
    official.eval_one_episode_batch(env_official, _BatchChunkClient(24))
    assert env_ours.calls == env_official.calls


class _FakeChunkClient(_FakePolicyClient):
    """Returns a whole 24-action chunk, as the server does for ``return_action_chunk``."""

    def predict_action(self, **payload):
        self.calls.append(payload)
        chunk = np.tile(np.linspace(-1, 1, 14, dtype=np.float32), (24, 1))
        return {"ok": True, "data": {"actions": chunk.tolist(), "inference_latency_ms": 5.0}}


def test_bridge_batch_one_request_per_env_per_chunk(monkeypatch, tmp_path):
    import loongforge.embodied.eval.transport as transport

    monkeypatch.setattr(transport, "PolicyClient", _FakeChunkClient)
    model = RoboDojoPolicyModel(
        {"task_name": "stack_bowls", "action_bridge": "fastwam_joint14", "first_chunk_len": 24,
         "chunk_len": 24, "lf_port": 1, "trace_dir": str(tmp_path / "trace")}
    )
    model.reset()
    model.update_obs_batch([{**_obs(0), "env_idx": 0}, {**_obs(1), "env_idx": 2}])
    out = model.get_action_batch([0, 2])
    assert [len(actions) for actions in out] == [24, 24]
    model.update_obs_batch([{**_obs(2), "env_idx": 2}])
    out2 = model.get_action_batch([2])
    assert len(out2[0]) == 24
    calls = model.client.calls
    assert len(calls) == 3 and sorted(call["episode_step"] for call in calls) == [0, 0, 24]
    assert {call["episode_id"] for call in calls} == {
        "robodojo/stack_bowls/seed0/b0000e00",
        "robodojo/stack_bowls/seed0/b0000e02",
    }
    assert all(call["return_action_chunk"] for call in calls)
    model.reset()  # the next batch runs new episodes
    model.update_obs_batch([{**_obs(0), "env_idx": 0}])
    model.get_action_batch([0])
    assert any(
        call["episode_id"].endswith("b0001e00") and call["episode_step"] == 0 for call in model.client.calls
    )
    assert sorted(p.name for p in (tmp_path / "trace").iterdir()) == [
        "b0000e00.jsonl", "b0000e02.jsonl", "b0001e00.jsonl"
    ]


def test_bridge_batch_requires_an_observation(monkeypatch):
    import loongforge.embodied.eval.transport as transport

    monkeypatch.setattr(transport, "PolicyClient", _FakeChunkClient)
    model = RoboDojoPolicyModel(
        {"task_name": "stack_bowls", "action_bridge": "fastwam_joint14", "first_chunk_len": 24,
         "chunk_len": 24, "lf_port": 1}
    )
    model.reset()
    with pytest.raises(RuntimeError):
        model.get_action_batch([0])


def test_runner_eval_batch_plugin_and_deploy_config(tmp_path):
    args = _runner_args(tmp_path)
    args.eval_batch, args.num_envs = True, 6
    args.policy_server_port = 62300
    args.robodojo_root = str(tmp_path / "rd")
    args.additional_info, args.run_id = "x", "rid"
    cfg = robodojo_runner.write_policy_deploy_config(args, tmp_path / "deploy.yml")
    assert cfg["policy_name"] == "LoongForgeBatch" and cfg["eval_batch"] is True
    assert robodojo_runner.result_dir(args).parts[-4] == "LoongForgeBatch"
    assert robodojo_runner.plugin_name(argparse.Namespace(eval_batch=False)) == "LoongForge"


def _write_official_eval_policy(root: pathlib.Path) -> pathlib.Path:
    script = root / "scripts" / "eval_policy.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/bin/bash\n"
        f"{robodojo_runner._EVAL_POLICY_ROOT_LINE}\n"
        'sim_cfg_file="$PROJECT_ROOT/env_cfg/sim/sim_config.yml"\n'
        f"{robodojo_runner._EVAL_POLICY_NUM_ENVS_PREFIX}\n"
        'echo "$num_envs"\n',
        encoding="utf-8",
    )
    return script


def test_runner_eval_policy_copy_changes_only_two_lines(tmp_path):
    args = _runner_args(tmp_path)
    args.robodojo_root = str(tmp_path / "rd")
    official = _write_official_eval_policy(pathlib.Path(args.robodojo_root))
    args.eval_batch, args.num_envs, args.eval_num = True, 6, 0
    copy = robodojo_runner.eval_policy_script(args)
    assert copy.name == "eval_policy_num_envs.sh"
    original_lines = official.read_text().splitlines()
    copy_lines = copy.read_text().splitlines()
    differing = [i for i, (a, b) in enumerate(zip(original_lines, copy_lines)) if a != b]
    assert len(original_lines) == len(copy_lines) and len(differing) == 2
    assert 'num_envs="6"' in copy_lines[differing[1]]
    assert str((tmp_path / "rd").resolve()) in copy_lines[differing[0]]
    # never more envs than the run has episodes
    args.eval_num = 3
    assert 'num_envs="3"' in robodojo_runner.eval_policy_script(args).read_text()
    # one env per process keeps the official script
    args.eval_batch = False
    assert robodojo_runner.eval_policy_script(args) == official
    # a changed official script fails instead of running something unintended
    args.eval_batch = True
    official.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        robodojo_runner.eval_policy_script(args)


def test_runner_zenity_hang_needs_no_step_progress(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(robodojo_runner.time, "time", lambda: clock[0])
    log = tmp_path / "sim.log"
    log.write_text("env0 step: 5 / 100\n", encoding="utf-8")
    since: dict = {}
    assert not robodojo_runner._zenity_hung(log, since)  # no crash dialog yet
    with log.open("a", encoding="utf-8") as handle:
        handle.write("/bin/bash: line 1: zenity: command not found\n")
    assert not robodojo_runner._zenity_hung(log, since) and "t" in since
    # the crash reporter keeps writing, but the sim makes no progress
    with log.open("a", encoding="utf-8") as handle:
        handle.write("[Warning] [carb.crashreporter-breakpad.plugin] Uploading minidump\n")
    clock[0] += robodojo_runner.ZENITY_HANG_SEC + 1
    assert robodojo_runner._zenity_hung(log, since)
    # step progress after the dialog clears the watchdog
    since.clear()
    log.write_text("zenity: command not found\n", encoding="utf-8")
    assert not robodojo_runner._zenity_hung(log, since)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("env0 step: 6 / 100\n")
    clock[0] += robodojo_runner.ZENITY_HANG_SEC + 1
    assert not robodojo_runner._zenity_hung(log, since) and not since


def test_runner_detects_kit_shutdown_hang(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(robodojo_runner.time, "time", lambda: clock[0])
    log = tmp_path / "sim.log"
    log.write_text("env0 step: 5 / 100\n", encoding="utf-8")
    since: dict = {}
    assert not robodojo_runner._shutdown_hung(log, since)
    log.write_text("Simulation App Shutting Down\n", encoding="utf-8")
    assert not robodojo_runner._shutdown_hung(log, since)
    clock[0] += robodojo_runner.SHUTDOWN_HANG_SEC + 1
    assert robodojo_runner._shutdown_hung(log, since)


def test_runner_client_completed_line(tmp_path):
    log = tmp_path / "sim.log"
    log.write_text("env0 step: 5 / 100\n", encoding="utf-8")
    assert not robodojo_runner.client_completed(log)
    log.write_text("all done (eval completed)\n", encoding="utf-8")
    assert robodojo_runner.client_completed(log)
