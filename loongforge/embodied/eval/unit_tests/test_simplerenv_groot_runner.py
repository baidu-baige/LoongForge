"""Regression tests for deterministic SimplerEnv vector episode scheduling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from loongforge.embodied.eval.orchestrator.runners import simplerenv_runner as runner


class _FakeAdapter:
    env_name = "FakeSimplerEnv-v0"

    def __init__(self, *args, **kwargs):
        pass

    def obs_to_canonical(self, obs, ctx):
        return {"instruction": ctx["instruction"], "obs": obs}

    def action_from_canonical(self, action):
        return np.asarray(action["actions"], dtype=np.float32)


class _FakeBuilder:
    def build(self, canonical, ctx):
        return {
            "images": [np.zeros((2, 2, 3), dtype=np.uint8)],
            "instructions": [ctx["instruction"]],
            "state": np.zeros(8, dtype=np.float32),
        }

    def reset(self, episode_id):
        self.episode_id = episode_id

    def update_from_response(self, data):
        pass


class _FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.resets = []
        self.calls = []
        self.__class__.instances.append(self)

    def reset(self, episode_id):
        self.resets.append(episode_id)

    def predict_action(self, **kwargs):
        self.calls.append(kwargs)
        batch = len(kwargs["images"]) if kwargs.get("batch_request") else 1
        return {"ok": True, "data": {"actions": np.zeros((batch, 1, 7), dtype=np.float32)}}

    def close(self):
        pass


class _FakePool:
    def __init__(self, args):
        self.args = args
        self.seeds = [args.seed + i for i in range(args.num_envs)]
        self.reset_counts = [0] * args.num_envs
        self.pending = [False] * args.num_envs

    def initial(self):
        return [({"env": i}, f"instruction-{i}", None, seed, False) for i, seed in enumerate(self.seeds)]

    def step(self, actions):
        results = []
        for i, action in enumerate(actions):
            if self.pending[i]:
                self.reset_counts[i] += 1
                self.seeds[i] = 1000 + i * 100 + self.reset_counts[i]
                self.pending[i] = False
                results.append(({"env": i}, f"instruction-{i}", None, self.seeds[i], True))
                continue
            seed = self.seeds[i]
            completed = {"success": int(seed % 2 == 0), "steps": 1, "instruction": f"instruction-{i}", "seed": seed}
            self.pending[i] = True
            results.append(({"env": i}, f"instruction-{i}", completed, self.seeds[i], False))
        return results

    def close(self):
        pass


def test_worker_refreshes_dynamic_instruction(monkeypatch):
    class Env:
        unwrapped = SimpleNamespace(_episode_seed=7)

        def __init__(self):
            self.steps = 0

        def step(self, action):
            self.steps += 1
            return {"step": self.steps}, 0.0, False, False, {"success": False}

        def close(self):
            pass

    class Conn:
        def __init__(self):
            self.sent = []
            self.messages = [np.zeros(7, dtype=np.float32), None]

        def recv(self):
            return self.messages.pop(0)

        def send(self, value):
            self.sent.append(value)

        def close(self):
            pass

    env = Env()
    conn = Conn()
    monkeypatch.setattr(runner, "_make_adapter", lambda args: _FakeAdapter())
    monkeypatch.setattr(runner, "_build_env", lambda args, adapter: (env, env.__class__.__name__))
    monkeypatch.setattr(runner, "_reset_env", lambda env, args: ({"step": 0}, {}))
    monkeypatch.setattr(
        runner,
        "_get_instruction",
        lambda env, fallback: "open drawer" if env.steps == 0 else "place cube into drawer",
    )
    args = _args(SimpleNamespace())

    runner._env_worker(conn, args, 7)

    assert conn.sent[0][1][1] == "open drawer"
    assert conn.sent[1][1][1] == "place cube into drawer"


def _args(tmp_path):
    return SimpleNamespace(
        task_name="google_robot_pick_object",
        robot_setup="google_robot",
        control_freq=5,
        max_steps=300,
        camera_name=None,
        action_scale=1.0,
        rotation_mode="axis_angle",
        simplerenv_root=None,
        num_envs=3,
        episodes_per_task=8,
        seed=10,
        output_dir=str(tmp_path),
        host="127.0.0.1",
        port=1,
        instruction=None,
        disable_action_cache=False,
        batch_inference=True,
        cfg_scale=1.0,
    )


def test_run_batch_uses_official_global_completion_semantics(monkeypatch, tmp_path):
    _FakeClient.instances.clear()
    monkeypatch.setattr(runner._common, "ensure_vulkan_runtime", lambda *args: None)
    monkeypatch.setattr(runner, "_Pool", _FakePool)
    monkeypatch.setattr(runner, "PolicyClient", _FakeClient)
    monkeypatch.setattr(runner, "_make_adapter", lambda args: _FakeAdapter())
    monkeypatch.setattr(
        runner._common,
        "build_policy_stack",
        lambda args, adapter: (_FakeBuilder(), "", lambda actions, ctx: actions),
    )

    summary = runner.run_batch(_args(tmp_path))

    records = [
        __import__("json").loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert summary["n_episodes"] == 8
    assert [record["seed"] for record in records] == [10, 11, 12, 1001, 1101, 1201, 1002, 1102]
    assert len({record["episode_id"] for record in records}) == 8
    assert summary["success"] == 4
    calls = _FakeClient.instances[0].calls
    assert calls
    assert all(call["batch_request"] is True for call in calls)
    assert all(len(call["images"]) == 3 for call in calls)
