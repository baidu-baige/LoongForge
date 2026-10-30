# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free protocol and spawned RNG-isolation regression tests."""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from loongforge.embodied.eval.adapters.libero import LiberoAdapter
from loongforge.embodied.eval.orchestrator.config import apply_config
from loongforge.embodied.eval.orchestrator.runners import libero_gr00t_runner as gr00t
from loongforge.embodied.eval.orchestrator.runners import libero_runner as legacy
from loongforge.embodied.eval.payload_builders.gr00t_n1_7 import GrootN1d7PayloadBuilder


def observation():
    return {
        "agentview_image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([1., 2., 3.]),
        "robot0_eef_quat": np.array([0., 0., 0., 1.]),
        "robot0_gripper_qpos": np.array([0.02, -0.02]),
    }


class FakeEnv:
    def __init__(self, task=None, resolution=256):
        self.task = task or SimpleNamespace(done_after=2)
        self.env = SimpleNamespace(robots=[SimpleNamespace(controller=SimpleNamespace(
            ee_pos=np.array([99., 99., 99.]), ee_ori_mat=np.eye(3), use_delta=False))])
        self.seeds = []
        self.resets = 0
        self.actions = []

    def seed(self, seed):
        self.seeds.append(seed)
        np.random.seed(seed)

    def reset(self):
        self.resets += 1
        self.steps = 0
        self.obs = observation()
        self.obs["rng"] = float(np.random.random())
        self.obs["pid"] = os.getpid()
        self.obs["seeds"] = list(self.seeds)
        self.obs["resets"] = self.resets
        return self.obs

    def set_init_state(self, state):
        raise AssertionError("Official path must never call set_init_state")

    def step(self, action):
        self.steps += 1
        self.actions.append(action)
        done = self.steps >= self.task.done_after
        return self.obs, float(done), done, {}

    def check_success(self):
        return self.steps >= self.task.done_after

    def close(self):
        self.closed = True


class FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.calls = []
        self.resets = []
        self.closed = False
        self.__class__.instances.append(self)

    def reset(self, episode_id):
        self.resets.append(episode_id)
        return {"ok": True, "data": {}}

    def predict_action(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True, "data": {"actions": np.full((16, 7), 0.5)}}

    def close(self):
        self.closed = True


def arguments(**kwargs):
    args = legacy.build_argparser().parse_args([])
    args.libero_protocol = "gr00t_official"
    args.episodes_per_task = 11
    args.num_steps_wait = 0
    args.max_steps = 17
    args.seed = 42
    args.raw_config = {"model": {"model_type": "Gr00tN1d7"}}
    args.per_episode_timeout_sec = 30
    args.__dict__.update(kwargs)
    return args


def test_adapter_state_gripper_and_legacy_unchanged():
    obs = observation()
    ctx = {"instruction": "test", "ee_pos": np.array([99., 99., 99.]),
           "ee_ori_mat": np.eye(3)}
    adapter = gr00t.GrootOfficialAdapter()
    canonical = adapter.obs_to_canonical(obs, ctx)
    payload = GrootN1d7PayloadBuilder().build(canonical, ctx)
    np.testing.assert_allclose(payload["state"], [1, 2, 3, 0, 0, 0, .02, -.02])
    np.testing.assert_array_equal(canonical["images"]["primary"], obs["agentview_image"][::-1, ::-1])
    for grip, expected in [(0, 1), (.499, 1), (.5, 0), (.501, -1), (1, -1)]:
        assert adapter.action_from_canonical({"actions": [0] * 6 + [grip]})[-1] == expected
    old = LiberoAdapter()
    assert old.action_from_canonical({"actions": [0] * 6 + [.5]})[-1] == 1
    assert LiberoAdapter(continuous_gripper=True).action_from_canonical(
        {"actions": [0] * 6 + [.5]})[-1] == .5
    np.testing.assert_array_equal(old.obs_to_canonical(obs, ctx)["state_raw"]["eef_pos"], [99] * 3)


def test_session_seed_once_random_reset_reuses_env_and_interrupts():
    env = FakeEnv()
    session = gr00t.GrootOfficialEnvSession(env, 42, 17)
    for _ in range(3):
        _, completed = session.step([[0] * 7] * 8)
        assert completed["steps"] == 2
        assert completed["success"] == 1
    assert env.resets == 4
    assert env.seeds == [42]
    assert len(env.actions) == 6
    assert env.env.robots[0].controller.use_delta is True


def test_macro_eight_actual_steps_and_exact_truncation():
    env = FakeEnv(SimpleNamespace(done_after=100))
    session = gr00t.GrootOfficialEnvSession(env, 42, 17)
    assert session.step([[0] * 7] * 8)[1] is None
    assert session.steps == 8
    assert session.step([[0] * 7] * 8)[1] is None
    _, completed = session.step([[0] * 7] * 8)
    assert completed["steps"] == 17
    assert completed["success"] == 0
    assert len(env.actions) == 17


def test_autoreset_success_is_merged_into_terminal_episode():
    class ResetSuccessEnv(FakeEnv):
        def check_success(self):
            return self.resets >= 2 and self.steps == 0

    env = ResetSuccessEnv(SimpleNamespace(done_after=100))
    session = gr00t.GrootOfficialEnvSession(env, 42, 1)
    _, completed = session.step([[0] * 7] * 8)
    assert completed["steps"] == 1
    assert completed["success"] == 1
    assert session.success is False


def test_initial_reset_success_does_not_mark_first_episode_successful():
    class InitialSuccessEnv(FakeEnv):
        def check_success(self):
            return self.resets == 1 and self.steps == 0

    session = gr00t.GrootOfficialEnvSession(InitialSuccessEnv(SimpleNamespace(done_after=100)), 42, 1)
    assert session.reset_success is True
    _, completed = session.step([[0] * 7] * 8)
    assert completed["success"] == 0


class EnvFactory:
    """Record every env collect_task builds; one env per task is expected."""

    def __init__(self, env_cls=FakeEnv, done_after=2):
        self.env_cls = env_cls
        self.done_after = done_after
        self.envs = []

    def __call__(self, task, resolution):
        assert resolution == 256
        env = self.env_cls(SimpleNamespace(done_after=self.done_after))
        self.envs.append(env)
        return env


@pytest.mark.parametrize("requested", [1, 3, 11])
def test_collection_exact_limit_cache_reset_and_rpc(requested):
    factory = EnvFactory()
    records = gr00t.collect_groot_official_task(
        arguments(episodes_per_task=requested), SimpleNamespace(language="test"), 7, "run",
        env_factory=factory, client_factory=FakeClient)
    assert len(records) == requested
    assert [r["episode_idx"] for r in records] == list(range(requested))
    assert all(r["env_idx"] == 0 and r["env_seed"] == 42 for r in records)
    assert all(r["success"] == 1 and r["steps"] == 2 for r in records)
    assert all("episode_seed" not in r and "seed" not in r for r in records)
    # One env per task, seeded once and reused through random resets.
    assert len(factory.envs) == 1
    env = factory.envs[0]
    assert env.seeds == [42]
    assert env.resets == requested + 1
    assert env.closed
    client = FakeClient.instances[-1]
    assert len(client.calls) == requested
    assert client.resets == ["run/task7/env0"] * (requested + 1)
    assert all(c["disable_action_cache"] and c["return_action_chunk"] for c in client.calls)
    assert all(c["episode_step"] == 0 for c in client.calls)
    # Official sign gripper: model 0.5 -> 0.
    assert all(action[-1] == 0 for action in env.actions)
    assert client.closed


def test_macro_rounds_advance_episode_step_until_completion():
    factory = EnvFactory(done_after=10)
    records = gr00t.collect_groot_official_task(
        arguments(episodes_per_task=1), SimpleNamespace(language="test"), 0, "run",
        env_factory=factory, client_factory=FakeClient)
    assert [r["steps"] for r in records] == [10]
    client = FakeClient.instances[-1]
    assert [c["episode_step"] for c in client.calls] == [0, 8]
    assert len(factory.envs[0].actions) == 10


class BrokenResetEnv(FakeEnv):
    def reset(self):
        if self.resets:
            raise RuntimeError("autoreset failed")
        return super().reset()


def test_failure_after_partial_completion_raises_and_closes():
    factory = EnvFactory(env_cls=BrokenResetEnv)
    with pytest.raises(RuntimeError, match="autoreset failed"):
        gr00t.collect_groot_official_task(arguments(), SimpleNamespace(language="test"), 0, "run",
                              env_factory=factory, client_factory=FakeClient)
    assert FakeClient.instances[-1].closed and factory.envs[-1].closed


def test_policy_error_and_short_chunk_fail_task():
    class ErrorClient(FakeClient):
        def predict_action(self, **kwargs):
            return {"ok": False, "error": "policy unavailable"}

    class ShortClient(FakeClient):
        def predict_action(self, **kwargs):
            return {"ok": True, "data": {"actions": np.zeros((7, 7))}}

    for client, error in [(ErrorClient, RuntimeError), (ShortClient, ValueError)]:
        factory = EnvFactory()
        with pytest.raises(error):
            gr00t.collect_groot_official_task(arguments(), SimpleNamespace(language="test"), 0, "run",
                                  env_factory=factory, client_factory=client)
        assert factory.envs[-1].closed


def test_task_failure_excluded_and_resume_cannot_pollute(tmp_path, monkeypatch):
    suite = SimpleNamespace(n_tasks=3, get_task=lambda i: SimpleNamespace(language=f"task{i}"))
    suite.get_task_init_states = Mock(side_effect=AssertionError("No fixed states"))
    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_goal": lambda: suite})
    monkeypatch.setitem(sys.modules, "libero.libero", SimpleNamespace(benchmark=benchmark))
    (tmp_path / "results.jsonl").write_text('{"success":1,"task_id":99}\n')
    args = arguments(output_dir=str(tmp_path), max_tasks=0, resume=True, restart=True)

    def collect(args, task, task_id, run_id):
        if task_id == 1:
            raise RuntimeError("partial task failed")
        # Different episode counts deliberately verify simple task-rate mean.
        return [{"success": s, "frames": [], "trace": [], "episode_idx": i}
                for i, s in enumerate([1] if task_id == 0 else [0, 0, 1])]

    monkeypatch.setattr(gr00t, "collect_groot_official_task", collect)
    first = legacy.run_batch(args)
    second = legacy.run_batch(args)
    assert first["success_rate"] == pytest.approx(2 / 3)
    assert first["valid_tasks"] == 2 and first["failed_tasks"] == 1
    assert first["total_tasks"] == 3 and first["n_records"] == 4
    assert first["results_path"] != second["results_path"]
    assert (tmp_path / "results.jsonl").read_text() == '{"success":1,"task_id":99}\n'
    entries = [json.loads(line) for line in open(first["tasks_path"])]
    assert entries[1]["status"] == "FAILED" and entries[1]["success_rate"] is None
    assert suite.get_task_init_states.call_count == 0
    assert "valid_tasks" in open(first["suite_summary_path"]).read()


def test_all_failed_or_unparsed_has_no_rate():
    summary = gr00t.summarize_groot_official_tasks([
        {"status": "FAILED", "success_rate": None},
        {"status": "OK", "success_rate": "unparsed"},
        {"status": "OK", "success_rate": float("nan")},
    ])
    assert summary["success_rate"] is None
    assert summary["valid_tasks"] == 0


def test_explicit_config_routing_and_legacy_default(monkeypatch):
    args = legacy.build_argparser().parse_args([])
    assert args.libero_protocol == "legacy"
    apply_config(args, {"benchmark": {"libero_protocol": "gr00t_official"}})
    run = Mock(return_value={"official": True})
    monkeypatch.setattr(legacy, "_run_batch_groot_official", run)
    assert legacy.run_batch(args) == {"official": True}
    run.assert_called_once_with(args)
    args.libero_protocol = "typo"
    with pytest.raises(ValueError, match="Unknown LIBERO protocol"):
        legacy.run_batch(args)


@pytest.mark.parametrize("overrides", [
    {"num_steps_wait": 10}, {"chunk_execute_steps": 4}, {"episodes_per_task": 0},
    {"continuous_gripper": True}, {"control_mode": "absolute"}, {"render_resolution": 128},
])
def test_protocol_rejects_conflicting_configuration(overrides):
    with pytest.raises(ValueError):
        gr00t._run_batch_groot_official(arguments(**overrides))


def test_other_model_rejected_only_in_opt_in_path():
    args = arguments(raw_config={"model": {"model_type": "pi05"}})
    with pytest.raises(ValueError, match="requires GR00T"):
        gr00t.collect_groot_official_task(args, SimpleNamespace(language="test"), 0, "run")


def test_legacy_episode_keeps_init_state_and_episode_seed(monkeypatch):
    args = arguments(max_steps=1, num_steps_wait=0)
    env = FakeEnv()
    env.set_init_state = Mock(return_value=observation())
    env.env.robots[0].controller.update = Mock()
    create = Mock(return_value=(env, "test"))
    monkeypatch.setattr(legacy, "_get_libero_env", create)
    builder = GrootN1d7PayloadBuilder()
    client = FakeClient()
    client.metadata = {}
    decoder = Mock(side_effect=lambda chunk, ctx: chunk)
    record = legacy.run_episode(
        client=client, adapter=LiberoAdapter(), payload_builder=builder,
        action_decoder=decoder, action_decoder_key="", task_suite_name="libero_goal",
        task_id=2, episode_idx=3, task=SimpleNamespace(language="test"),
        initial_state="fixed-state", args=args)
    create.assert_called_once_with(SimpleNamespace(language="test"), 256, 2045)
    env.set_init_state.assert_called_once_with("fixed-state")
    assert record["seed"] == 2045
    assert client.calls[0]["state"][0] == 99
    assert env.actions[0][-1] == 1
