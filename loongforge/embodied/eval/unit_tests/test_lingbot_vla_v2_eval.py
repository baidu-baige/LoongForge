# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the LingBot-VLA v2 RoboTwin eval wiring (no weights, no GPU)."""

from __future__ import annotations

import numpy as np

from loongforge.embodied.eval.adapters.robotwin import RoboTwinAdapter
from loongforge.embodied.eval.bridges.robotwin_policy import _BRIDGE_WIRING
from loongforge.embodied.eval.payload_builders import build_payload_builder
from loongforge.embodied.eval.servers.loongforge_policy import GenericPredictActionPolicy


def _robotwin_obs() -> dict:
    """RoboTwin-shaped observation with distinct per-camera values and a float64 joint vector."""
    joint = np.linspace(-1.0, 1.0, 14, dtype=np.float64) + 1e-9
    return {
        "observation": {
            "head_camera": {"rgb": np.full((240, 320, 3), 10, dtype=np.uint8)},
            "left_camera": {"rgb": np.full((240, 320, 3), 20, dtype=np.uint8)},
            "right_camera": {"rgb": np.full((240, 320, 3), 30, dtype=np.uint8)},
        },
        "joint_action": {"vector": joint},
    }


def _canonical(obs: dict) -> dict:
    adapter = RoboTwinAdapter(task_name="adjust_bottle")
    return adapter.obs_to_canonical(obs, {"instruction": "pick up the bottle", "episode_id": "e0", "episode_step": 0})


def test_bridge_wiring_is_joint14_identity() -> None:
    assert _BRIDGE_WIRING["lingbot_vla_v2_joint14"] == ("lingbot_vla_v2", "robotwin_joint14", "")


def test_builder_takes_the_float64_joint_from_ctx_without_touching_the_adapter() -> None:
    obs = _robotwin_obs()
    canonical = _canonical(obs)
    # The shared adapter is unchanged: it still exposes only the float32 joint.
    assert canonical["state_raw"]["joint"].dtype == np.float32
    assert "joint_f64" not in canonical["state_raw"]
    builder = build_payload_builder("lingbot_vla_v2", yaml_model={"state_encoding": "robotwin_joint14"})
    joint_f64 = np.asarray(obs["joint_action"]["vector"], dtype=np.float64)
    payload = builder.build(canonical, {"benchmark_name": "robotwin", "joint_f64": joint_f64})
    np.testing.assert_array_equal(payload["state"], joint_f64)
    # Without the ctx key it still works, on the float32 value the adapter provides.
    fallback = builder.build(canonical, {"benchmark_name": "robotwin"})
    np.testing.assert_array_equal(fallback["state"], canonical["state_raw"]["joint"].astype(np.float64))


def test_payload_matches_community_client_fields() -> None:
    obs = _robotwin_obs()
    builder = build_payload_builder("lingbot_vla_v2", yaml_model={"state_encoding": "robotwin_joint14"})
    ctx = {"benchmark_name": "robotwin", "joint_f64": np.asarray(obs["joint_action"]["vector"], dtype=np.float64)}
    payload = builder.build(_canonical(obs), ctx)
    # cam_high, cam_left_wrist, cam_right_wrist: raw HWC uint8, no resize
    assert [int(v[0, 0, 0]) for v in payload["images"]] == [10, 20, 30]
    assert all(v.shape == (240, 320, 3) and v.dtype == np.uint8 for v in payload["images"])
    assert payload["state"].dtype == np.float64
    np.testing.assert_array_equal(payload["state"], obs["joint_action"]["vector"])
    assert payload["instructions"] == ["pick up the bottle"]
    assert builder.disable_action_cache is False


class _FakeChunkModel:
    """Returns a (50, 14) chunk whose rows encode (call index, step index)."""

    def __init__(self) -> None:
        self.calls = 0

    def predict_action(self, images, instructions, state=None, dataset_stats=None, episode_id=None, episode_step=0):
        self.calls += 1
        return np.stack([np.full(14, self.calls * 1000 + i, dtype=np.float32) for i in range(50)])


def test_generic_policy_replays_full_50_step_chunk_then_replans() -> None:
    model = _FakeChunkModel()
    policy = GenericPredictActionPolicy(model=model, metadata={}, action_dim=14)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    served = []
    for step in range(101):
        out = policy.predict_action(images=[image] * 3, instructions=["x"], episode_id="ep", episode_step=step,
                                    state=np.zeros(14))
        served.append(float(np.asarray(out["actions"])[0, 0]))
    assert model.calls == 3
    assert served[:50] == [1000 + i for i in range(50)]
    assert served[50:100] == [2000 + i for i in range(50)]
    assert served[100] == 3000
    policy.reset("ep")
    policy.predict_action(images=[image] * 3, instructions=["x"], episode_id="ep", episode_step=0, state=np.zeros(14))
    assert model.calls == 4


class _FakeClient:
    def __init__(self) -> None:
        self.requests = []

    def reset(self, episode_id):
        return None

    def predict_action(self, **kwargs):
        self.requests.append(kwargs)
        n = len(self.requests)
        return {"ok": True, "data": {"actions": np.stack([np.full(14, n * 1000 + i, dtype=np.float32) for i in range(50)]),
                                     "inference_latency_ms": 1.0}}


class _FakeTaskEnv:
    """take_action no-ops at step_lim or after success, like RoboTwin's _base_task.take_action."""

    def __init__(self, step_lim: int, succeed_at: int) -> None:
        self.step_lim, self.succeed_at = step_lim, succeed_at
        self.take_action_cnt, self.eval_success, self.executed = 0, False, []

    def get_instruction(self) -> str:
        return "pick up the bottle"

    def take_action(self, action, action_type="qpos") -> None:
        assert action_type == "qpos"
        if self.take_action_cnt == self.step_lim or self.eval_success:
            return
        self.take_action_cnt += 1
        self.executed.append(float(action[0]))
        if self.take_action_cnt == self.succeed_at:
            self.eval_success = True


def _proprio_obs(obs: dict) -> dict:
    """What the runner's skip_intra_chunk_obs patch passes when no render is needed."""
    return {"joint_action": {"vector": obs["joint_action"]["vector"]}}


def _chunk_bridge_client():
    """A ModelClient wired like the YAML does: client-owned 50-step chunk, no server cache."""
    from loongforge.embodied.eval.action_decoders import build_action_decoder
    from loongforge.embodied.eval.bridges.robotwin_policy import ModelClient

    client = ModelClient.__new__(ModelClient)
    client.client = _FakeClient()
    client.adapter = RoboTwinAdapter(task_name="adjust_bottle")
    client.payload_builder = build_payload_builder("lingbot_vla_v2", yaml_model={"state_encoding": "robotwin_joint14"})
    client.decoder = build_action_decoder("")
    client.action_bridge = "lingbot_vla_v2_joint14"
    client.disable_action_cache = False
    client.return_action_chunk = True
    client.chunk_execute_steps = 0  # execute the whole returned chunk before replanning
    client.task_description, client.episode_id = None, None
    client._chunk, client._chunk_cursor, client._chunk_len = None, 0, 0
    client._chunk_state, client._chunk_endpose = None, None
    client.trace_path, client.trace_records = None, []
    return client


def _official_loop(env, client) -> list:
    """RoboTwin script/eval_policy.py episode loop, with the skip_intra_chunk_obs patch applied.

    Returns the per-step observation kind so a test can assert the cameras were rendered
    only on the steps that actually hit the policy server.
    """
    from loongforge.embodied.eval.bridges import robotwin_policy

    robotwin_policy.reset_model(client)
    obs_kinds = []
    while env.take_action_cnt < env.step_lim:
        rendered = client.needs_obs()
        obs_kinds.append("render" if rendered else "proprio")
        observation = _robotwin_obs() if rendered else _proprio_obs(_robotwin_obs())
        robotwin_policy.eval(env, client, observation)
        if env.eval_success:
            break
    return obs_kinds


def test_chunk_bridge_executes_full_chunks_and_stops_on_success() -> None:
    env, client = _FakeTaskEnv(step_lim=400, succeed_at=120), _chunk_bridge_client()
    obs_kinds = _official_loop(env, client)
    # one request per 50-step chunk, uncached, full chunk returned
    assert len(client.client.requests) == 3
    assert all(not r["disable_action_cache"] and r["return_action_chunk"] for r in client.client.requests)
    assert env.executed == [1000 + i for i in range(50)] + [2000 + i for i in range(50)] + [3000 + i for i in range(20)]
    assert env.eval_success
    # cameras rendered only on the three replanning steps
    assert [i for i, kind in enumerate(obs_kinds) if kind == "render"] == [0, 50, 100]


def test_chunk_bridge_respects_step_limit() -> None:
    env, client = _FakeTaskEnv(step_lim=120, succeed_at=10**9), _chunk_bridge_client()
    _official_loop(env, client)
    assert env.take_action_cnt == 120 and len(client.client.requests) == 3 and not env.eval_success


def test_chunk_bridge_sends_the_float64_joint_from_the_env_observation() -> None:
    env, client = _FakeTaskEnv(step_lim=50, succeed_at=10**9), _chunk_bridge_client()
    _official_loop(env, client)
    sent = client.client.requests[0]["state"]
    assert sent.dtype == np.float64
    np.testing.assert_array_equal(sent, _robotwin_obs()["joint_action"]["vector"])
