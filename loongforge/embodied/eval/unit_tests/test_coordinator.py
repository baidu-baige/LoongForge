# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the coordinator (pure helpers)."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest
from typing import Any

from loongforge.embodied.eval.orchestrator.coordinator import _shard_config


def _shard(sid: str, overrides: dict, expected: int) -> dict:
    return {"id": sid, "overrides": overrides, "expected": expected}


def test_shard_config_applies_overrides_and_strips_coordinator_keys() -> None:
    """The shard patch is applied mechanically; coordinator-only keys are dropped."""
    base = {
        "benchmark": {"suites": ["a", "b"], "suite_overrides": {"a": {"max_steps": 1}}},
        "server": {"ckpt_path": "x"},
        "coordinator": {"n_workers": 4},
        "servers": [{"gpus": [0, 1]}],
        "pools": {"cpu_sim": {"n_sims_per_server": 2}},
    }
    patch = {"benchmark": {"suite": "b", "task_ids": [3], "episode_offset": 10, "episodes_per_task": 20}}
    shard = _shard_config(base, _shard("b/task=3", patch, 20), "ws://h:1", "/tmp/out")
    assert "coordinator" not in shard and "servers" not in shard and "pools" not in shard
    assert "suites" not in shard["benchmark"] and "suite_overrides" not in shard["benchmark"]
    assert shard["benchmark"]["suite"] == "b"
    assert shard["benchmark"]["task_ids"] == [3]
    assert shard["benchmark"]["episode_offset"] == 10
    assert shard["benchmark"]["episodes_per_task"] == 20
    assert shard["server"]["external_url"] == "ws://h:1"
    assert shard["server"]["host"] == "h"
    assert shard["server"]["port"] == 1
    assert shard["run"]["timestamped_output"] is False
    assert "coordinator" in base  # base not mutated


def test_shard_config_defaults_max_retries_but_respects_explicit() -> None:
    """Coordinated units retry infra episode failures by default; an explicit value wins."""
    base = {"benchmark": {"suite": "a"}, "server": {"ckpt_path": "x"}}
    shard = _shard_config(base, _shard("a/ep=0", {}, 2), "ws://h:1", "/tmp/out")
    assert shard["run"]["max_retries"] == 2
    explicit = {**base, "run": {"max_retries": 0}}
    assert _shard_config(explicit, _shard("a/ep=0", {}, 2), "ws://h:1", "/tmp/out")["run"]["max_retries"] == 0


class _FakeProc:
    """Minimal stand-in for ``subprocess.Popen`` in the dispatch loop."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.pid = 12345
        self.terminated = False

    def poll(self) -> int:
        """Run poll."""
        return self.returncode

    def terminate(self) -> None:
        """Run terminate."""
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        """Run wait."""
        return self.returncode

    def kill(self) -> None:
        """Run kill."""
        self.terminated = True


def test_run_retries_a_failed_unit_once(monkeypatch: Any, tmp_path: Any) -> None:
    """A unit that exits non-zero is re-queued once; a later success is not a failure."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    launches: dict[int, int] = {}

    def fake_launch(config: Any, shard: Any, index: int, url: str, output_dir: Any, tmp: Any) -> _FakeProc:
        launches[index] = launches.get(index, 0) + 1
        return _FakeProc(1 if (index == 0 and launches[index] == 1) else 0)

    monkeypatch.setattr(coord, "_launch_unit", fake_launch)
    monkeypatch.setattr(coord, "start_server", lambda args, env=None: _FakeProc(0))
    monkeypatch.setattr(coord, "wait_healthz", lambda *args, **kwargs: None)
    monkeypatch.setattr(coord, "_aggregate", lambda *args, **kwargs: {})
    monkeypatch.setattr(coord, "short_shards", lambda *args, **kwargs: [])

    config = {
        "benchmark": {"name": "libero", "suite": "libero_10", "task_ids": [0], "episodes_per_task": 2},
        "servers": [{"gpus": [0]}],
        "pools": {"cpu_sim": {"n_sims_per_server": 1}},
        "run": {"output_dir": str(tmp_path)},
    }
    result = coord.run(config, "")
    assert result["n_units"] == 2
    assert launches[0] == 2  # failed once, retried
    assert result["failed_units"] == []


def test_watchdog_restarts_a_dead_server_in_place(monkeypatch: Any) -> None:
    """A server whose process exited is restarted on the same port by the watchdog."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    started = []

    def fake_start(args: Any, env: Any = None) -> _FakeProc:
        proc = _FakeProc(None)  # alive
        started.append(proc)
        return proc

    monkeypatch.setattr(coord, "start_server", fake_start)
    monkeypatch.setattr(coord, "wait_healthz", lambda *args, **kwargs: None)

    args = SimpleNamespace(host="127.0.0.1", port=14560, health_port=14561, server_start_timeout_sec=5)
    servers = [(_FakeProc(1), args)]  # the server process has exited
    watchdog = coord._ServerWatchdog(servers, [0], interval_sec=0.05)
    watchdog.start()
    time.sleep(0.4)
    watchdog.stop()
    watchdog.join(timeout=2)

    assert watchdog.restarts == 1
    assert servers[0][0] is started[0]  # replaced in place, so the URL stays valid
    assert servers[0][1] is args


def test_watchdog_leaves_a_live_server_alone(monkeypatch: Any) -> None:
    """Run test_watchdog_leaves_a_live_server_alone."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    monkeypatch.setattr(coord, "start_server", lambda *a, **k: _FakeProc(None))
    monkeypatch.setattr(coord, "wait_healthz", lambda *args, **kwargs: None)

    args = SimpleNamespace(host="127.0.0.1", port=14560, health_port=14561, server_start_timeout_sec=5)
    live = _FakeProc(None)
    servers = [(live, args)]
    watchdog = coord._ServerWatchdog(servers, [0], interval_sec=0.05)
    watchdog.start()
    time.sleep(0.3)
    watchdog.stop()
    watchdog.join(timeout=2)

    assert watchdog.restarts == 0
    assert servers[0][0] is live


def test_unit_timeout_derives_from_the_episode_budget() -> None:
    """The per-unit budget is max episodes x per_episode_sec + connect + slack."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    shards = [_shard("a/t0", {}, 4), _shard("a/t1", {}, 2)]
    config = {"timeouts": {"per_episode_sec": 100}, "server": {"connect_timeout_sec": 10}}
    assert coord._unit_timeout_sec(config, shards) == 4 * 100 + 10 + coord.UNIT_TIMEOUT_SLACK_SEC
    assert coord._unit_timeout_sec({**config, "run": {"unit_timeout_sec": 42}}, shards) == 42


def test_await_slot_returns_a_finished_unit() -> None:
    """Run test_await_slot_returns_a_finished_unit."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    done = _FakeProc(0)
    proc, timed_out = coord._await_slot({done: (0, 3)}, {3: time.time()}, timeout_sec=10)
    assert proc is done
    assert timed_out is False


def test_await_slot_kills_a_unit_that_blows_its_budget() -> None:
    """A unit hung past its budget is killed so the slot frees (it then gets retried)."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    stuck = _FakeProc(None)  # never finishes
    pool = {stuck: (0, 7)}
    started_at = {7: time.time() - 10}  # already 10s into its budget
    proc, timed_out = coord._await_slot(pool, started_at, timeout_sec=0.5)
    assert proc is stuck
    assert timed_out is True


def test_run_exits_nonzero_when_coverage_is_incomplete(monkeypatch: Any, tmp_path: Any) -> None:
    """A coordinated run whose coverage is still incomplete must not exit 0."""
    import sys as _sys

    from loongforge.embodied.eval.orchestrator import coordinator as coord
    from loongforge.embodied.eval.orchestrator import run as run_mod

    cfg = tmp_path / "c.yaml"
    cfg.write_text("coordinator: {}\n", encoding="utf-8")
    monkeypatch.setattr(_sys, "argv", ["run", "--config", str(cfg)])
    monkeypatch.setattr(run_mod, "load_config", lambda path: {"coordinator": {}})
    monkeypatch.setattr(coord, "run", lambda config, path: {"validation": {"ok": False}})
    with pytest.raises(SystemExit) as excinfo:
        run_mod.main()
    assert excinfo.value.code == 2


def test_run_exits_zero_when_coverage_is_complete(monkeypatch: Any, tmp_path: Any) -> None:
    """Run test_run_exits_zero_when_coverage_is_complete."""
    import sys as _sys

    from loongforge.embodied.eval.orchestrator import coordinator as coord
    from loongforge.embodied.eval.orchestrator import run as run_mod

    cfg = tmp_path / "c.yaml"
    cfg.write_text("coordinator: {}\n", encoding="utf-8")
    monkeypatch.setattr(_sys, "argv", ["run", "--config", str(cfg)])
    monkeypatch.setattr(run_mod, "load_config", lambda path: {"coordinator": {}})
    monkeypatch.setattr(coord, "run", lambda config, path: {"validation": {"ok": True}})
    run_mod.main()  # must not raise


def test_watchdog_does_not_leak_a_restart_started_during_shutdown(monkeypatch: Any) -> None:
    """A restart still in flight when the run shuts down must not leak a server process."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    created = []

    def fake_start(args: Any, env: Any = None) -> _FakeProc:
        proc = _FakeProc(None)
        created.append(proc)
        return proc

    monkeypatch.setattr(coord, "start_server", fake_start)
    args = SimpleNamespace(host="127.0.0.1", port=14560, health_port=14561, server_start_timeout_sec=5)
    servers = [(_FakeProc(1), args)]
    watchdog = coord._ServerWatchdog(servers, [0], interval_sec=0.05)
    # shutdown lands while the restart is waiting for the server to get healthy
    monkeypatch.setattr(coord, "wait_healthz", lambda *a, **k: watchdog.stop())

    watchdog.start()
    time.sleep(0.4)
    watchdog.stop()
    watchdog.join(timeout=2)

    assert watchdog.restarts == 0  # never counted as a successful restart
    assert servers[0][0].returncode == 1  # the original entry is untouched
    assert created and created[0].terminated  # the process we started was cleaned up


def test_watchdog_restarts_a_hung_server(monkeypatch: Any) -> None:
    """A live process that stops answering pings is restarted after the failure limit."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    monkeypatch.setattr(coord, "start_server", lambda args, env=None: _FakeProc(None))
    monkeypatch.setattr(coord, "wait_healthz", lambda *args, **kwargs: None)
    args = SimpleNamespace(host="127.0.0.1", port=14560, health_port=14561, server_start_timeout_sec=5)
    servers = [(_FakeProc(None), args)]  # process alive, but wedged
    watchdog = coord._ServerWatchdog(servers, [0], interval_sec=0.05, probe_interval_sec=0.0)
    monkeypatch.setattr(watchdog, "_alive", lambda a: False)

    watchdog.start()
    time.sleep(0.5)
    watchdog.stop()
    watchdog.join(timeout=2)

    assert watchdog.restarts >= 1  # it keeps failing the probe, so it may restart more than once


def test_watchdog_leaves_a_responding_server_alone(monkeypatch: Any) -> None:
    """Run test_watchdog_leaves_a_responding_server_alone."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    args = SimpleNamespace(host="127.0.0.1", port=14560, health_port=14561, server_start_timeout_sec=5)
    live = _FakeProc(None)
    watchdog = coord._ServerWatchdog([(live, args)], [0], interval_sec=0.05, probe_interval_sec=0.0)
    monkeypatch.setattr(watchdog, "_alive", lambda a: True)

    watchdog.start()
    time.sleep(0.3)
    watchdog.stop()
    watchdog.join(timeout=2)

    assert watchdog.restarts == 0


def test_alive_probe_fails_fast_and_restores_proxy_env(monkeypatch: Any) -> None:
    """A dead port fails the probe, and the probe puts the proxy env back."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    monkeypatch.setattr(coord, "PROBE_CONNECT_TIMEOUT_SEC", 0.1)
    monkeypatch.setenv("HTTP_PROXY", "http://example.invalid:1")
    watchdog = coord._ServerWatchdog([], [], interval_sec=99)
    assert watchdog._alive(SimpleNamespace(host="127.0.0.1", port=9, health_port=9)) is False
    assert os.environ.get("HTTP_PROXY") == "http://example.invalid:1"


def _mini_config(output_dir: str, **run_extra: Any) -> dict:
    return {
        "benchmark": {"name": "libero", "suite": "libero_10", "task_ids": [0], "episodes_per_task": 1},
        "servers": [{"gpus": [0]}],
        "pools": {"cpu_sim": {"n_sims_per_server": 1}},
        "run": {"output_dir": output_dir, **run_extra},
    }


def test_run_refuses_to_clobber_a_finished_run(tmp_path: Any) -> None:
    """An existing coordinator_result.json means "already run" -- refuse unless resuming."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    out = tmp_path / "out"
    out.mkdir()
    (out / "coordinator_result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        coord.run(_mini_config(str(out)), "")
    assert "already exists" in str(excinfo.value)


def test_run_allows_a_finished_dir_when_resuming(monkeypatch: Any, tmp_path: Any) -> None:
    """Run test_run_allows_a_finished_dir_when_resuming."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    out = tmp_path / "out"
    out.mkdir()
    (out / "coordinator_result.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(coord, "start_server", lambda args, env=None: _FakeProc(0))
    monkeypatch.setattr(coord, "wait_healthz", lambda *args, **kwargs: None)
    monkeypatch.setattr(coord, "_launch_unit", lambda *args, **kwargs: _FakeProc(0))
    monkeypatch.setattr(coord, "short_shards", lambda *args, **kwargs: [])
    result = coord.run(_mini_config(str(out), resume=True), "")
    assert result["n_units"] == 1


def test_policy_timeout_warning_fires_only_when_below_the_floor(capsys: Any) -> None:
    """The floor is N x service_ms x safety; warn below it, stay quiet above (or unknown)."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    coord._warn_policy_timeout({"timeouts": {"service_ms": 500, "policy_call_ms": 5000}}, 16)
    assert "below the queue-wait floor" in capsys.readouterr().err

    coord._warn_policy_timeout({"timeouts": {"service_ms": 500, "policy_call_ms": 60000}}, 16)
    assert capsys.readouterr().err == ""

    coord._warn_policy_timeout({"timeouts": {"policy_call_ms": 5000}}, 16)  # no service_ms -> skipped
    assert capsys.readouterr().err == ""


def test_watchdog_gives_up_after_repeated_failed_restarts(monkeypatch: Any) -> None:
    """A server that cannot come back is not restarted forever -- it is given up on."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("cannot start")

    monkeypatch.setattr(coord, "start_server", lambda args, env=None: _FakeProc(0))
    monkeypatch.setattr(coord, "wait_healthz", boom)
    args = SimpleNamespace(host="127.0.0.1", port=14560, health_port=14561, server_start_timeout_sec=5)
    watchdog = coord._ServerWatchdog([(_FakeProc(1), args)], [0], interval_sec=0.05, probe_interval_sec=999)

    watchdog.start()
    time.sleep(0.6)
    watchdog.stop()
    watchdog.join(timeout=2)

    assert 0 in watchdog.given_up
    assert watchdog.restarts == 0  # nothing ever came back


def test_run_exits_nonzero_when_a_server_was_given_up(monkeypatch: Any, tmp_path: Any) -> None:
    """Run test_run_exits_nonzero_when_a_server_was_given_up."""
    import sys as _sys

    from loongforge.embodied.eval.orchestrator import coordinator as coord
    from loongforge.embodied.eval.orchestrator import run as run_mod

    cfg = tmp_path / "c.yaml"
    cfg.write_text("coordinator: {}\n", encoding="utf-8")
    monkeypatch.setattr(_sys, "argv", ["run", "--config", str(cfg)])
    monkeypatch.setattr(run_mod, "load_config", lambda path: {"coordinator": {}})
    monkeypatch.setattr(coord, "run", lambda config, path: {"validation": {"ok": True}, "servers_given_up": [0]})
    with pytest.raises(SystemExit) as excinfo:
        run_mod.main()
    assert excinfo.value.code == 2


def test_watchdog_terminates_a_hung_server_before_restarting(monkeypatch: Any) -> None:
    """A hung server still holds its port, so it must be terminated before the rebind."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    monkeypatch.setattr(coord, "start_server", lambda args, env=None: _FakeProc(None))
    monkeypatch.setattr(coord, "wait_healthz", lambda *args, **kwargs: None)
    args = SimpleNamespace(host="127.0.0.1", port=14560, health_port=14561, server_start_timeout_sec=5)
    hung = _FakeProc(None)  # process alive, but wedged
    watchdog = coord._ServerWatchdog([(hung, args)], [0], interval_sec=0.05, probe_interval_sec=0.0)
    monkeypatch.setattr(watchdog, "_alive", lambda a: False)

    watchdog.start()
    time.sleep(0.5)
    watchdog.stop()
    watchdog.join(timeout=2)

    assert hung.terminated  # otherwise the new server cannot bind the port
    assert watchdog.restarts >= 1


def test_run_tears_down_servers_when_startup_fails(monkeypatch: Any, tmp_path: Any) -> None:
    """A failure before the watchdog starts must still terminate the servers.

    Regression: ``watchdog.join()`` on a never-started thread raised, and that
    exception skipped the teardown loop, leaking a policy server per GPU.
    """
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    server = _FakeProc(None)  # alive
    monkeypatch.setattr(coord, "start_server", lambda args, env=None: server)

    def never_healthy(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("never healthy")

    monkeypatch.setattr(coord, "wait_healthz", never_healthy)
    config = {
        "benchmark": {"name": "libero", "suite": "libero_10", "task_ids": [0], "episodes_per_task": 1},
        "servers": [{"gpus": [0]}],
        "pools": {"cpu_sim": {"n_sims_per_server": 1}},
        "run": {"output_dir": str(tmp_path)},
    }
    with pytest.raises(RuntimeError, match="never healthy"):
        coord.run(config, "")
    assert server.terminated  # not leaked


def test_port_preflight_reports_the_conflict() -> None:
    """A taken port must fail fast with the port number, not "server not healthy"."""
    import socket

    from loongforge.embodied.eval.orchestrator import coordinator as coord

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", 0))
        listener.listen(1)
        taken = listener.getsockname()[1]
        with pytest.raises(SystemExit, match=str(taken)):
            coord._assert_ports_free([taken])


def test_port_preflight_accepts_a_released_port() -> None:
    """SO_REUSEADDR semantics: a just-closed (TIME_WAIT) socket is not a conflict."""
    import socket

    from loongforge.embodied.eval.orchestrator import coordinator as coord

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("", 0))
        released = probe.getsockname()[1]
    coord._assert_ports_free([released])  # must not raise


def test_resolve_placement_uses_gpu_pool_for_robotwin() -> None:
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    config = {
        "benchmark": {"name": "robotwin"},
        "servers": [{"gpus": [0, 1]}],
        "pools": {"gpu_sim": {"n_sims_per_server": 3}},
    }
    server_gpus, n_per_server, sim_gpus, placement, is_gpu = coord._resolve_placement(config)
    assert server_gpus == [0, 1]
    assert n_per_server == 3
    assert sim_gpus == [0, 1]  # co_locate by default
    assert placement == "co_locate"
    assert is_gpu is True


def test_resolve_placement_uses_cpu_pool_for_libero() -> None:
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    config = {
        "benchmark": {"name": "libero"},
        "servers": [{"gpus": [2]}],
        "pools": {"cpu_sim": {"n_sims_per_server": 7}},
    }
    server_gpus, n_per_server, sim_gpus, placement, is_gpu = coord._resolve_placement(config)
    assert (server_gpus, n_per_server, sim_gpus, placement, is_gpu) == ([2], 7, [2], "co_locate", False)


def test_cpu_sim_family_warns_when_only_gpu_sim_pool_is_set(capsys) -> None:
    """LIBERO copied from a RoboTwin template: N would silently fall back to 1."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    config = {
        "benchmark": {"name": "libero_10"},
        "servers": [{"gpus": [0, 1]}],
        "pools": {"gpu_sim": {"n_sims_per_server": 16}},
    }
    _server_gpus, n_per_server, _sim_gpus, _placement, is_gpu = coord._resolve_placement(config)
    assert n_per_server == 1  # unchanged behaviour, but no longer silent
    assert is_gpu is False
    out = capsys.readouterr().out
    assert "pools.gpu_sim is set but this benchmark family reads pools.cpu_sim" in out

    # The right key still works and stays quiet.
    ok = {**config, "pools": {"cpu_sim": {"n_sims_per_server": 16}}}
    assert coord._resolve_placement(ok)[1] == 16
    assert capsys.readouterr().out == ""


def test_gpu_sim_family_warns_when_only_cpu_sim_pool_is_set(capsys) -> None:
    """The same typo in the other direction (RoboTwin configured as cpu_sim)."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    config = {
        "benchmark": {"name": "robotwin"},
        "servers": [{"gpus": [0]}],
        "pools": {"cpu_sim": {"n_sims_per_server": 4}},
    }
    # Legacy cross-fallback: the GPU-sim lookup also reads cpu_sim, so N=4 here
    # rather than 1. The point of the test is that it is no longer silent.
    assert coord._resolve_placement(config)[1] == 4
    assert "pools.cpu_sim is set but this benchmark family reads pools.gpu_sim" in capsys.readouterr().out


def test_separate_placement_requires_sim_gpus() -> None:
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    config = {
        "benchmark": {"name": "robotwin"},
        "servers": [{"gpus": [0]}],
        "pools": {"gpu_sim": {"n_sims_per_server": 1, "placement": "separate"}},
    }
    with pytest.raises(ValueError, match="sim_gpus"):
        coord._resolve_placement(config)


def test_separate_placement_rejects_gpu_overlap() -> None:
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    config = {
        "benchmark": {"name": "robotwin"},
        "servers": [{"gpus": [0, 1]}],
        "pools": {"gpu_sim": {"n_sims_per_server": 1, "placement": "separate", "sim_gpus": [1, 2]}},
    }
    with pytest.raises(ValueError, match="both a server and sims"):
        coord._resolve_placement(config)


def test_build_slots_co_locate_pins_worker_to_its_server_gpu() -> None:
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    # Sim-layer major: one slot per GPU per layer (not grouped by GPU).
    slots = coord._build_slots([0, 1], 2, [0, 1], "co_locate", ["ws://a", "ws://b"])
    assert slots == [("ws://a", 0), ("ws://b", 1), ("ws://a", 0), ("ws://b", 1)]


def test_build_slots_separate_spreads_workers_over_sim_gpus() -> None:
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    slots = coord._build_slots([0, 1], 2, [4, 5], "separate", ["ws://a", "ws://b"])
    assert slots == [("ws://a", 4), ("ws://b", 5), ("ws://a", 4), ("ws://b", 5)]


def test_build_slots_dispatch_order_spreads_across_gpus() -> None:
    """Popping free slots from the end must hit distinct GPUs first.

    A job with fewer units than slots should use every card, not pack onto the
    highest-numbered ones (regression for the episodes_per_chunk packing bug).
    """
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    g = 8
    slots = coord._build_slots(list(range(g)), 8, list(range(g)), "co_locate",
                               [f"ws://s{i}" for i in range(g)])
    assert len(slots) == g * 8
    # _run_batch does free_slots.pop() (from the end); inspect the first g pops.
    first_g_gpus = [slots[-1 - i][1] for i in range(g)]
    assert sorted(first_g_gpus) == list(range(g))  # one unit per GPU


def test_gpu_vram_budget_rejects_too_many_sims() -> None:
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    config = {
        "resources": {"gpu_sim": {"vram_total_mb": 80_000, "vram_per_sim_mb": 5_500, "vram_model_mb": 16_000}},
    }
    coord._validate_gpu_resources(config, 11, True)  # 76.5GB: fits
    with pytest.raises(ValueError, match="vram_total_mb"):
        coord._validate_gpu_resources(config, 12, True)  # 82GB: does not


def test_shard_config_pins_worker_gpu_only_when_requested() -> None:
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    base = {"benchmark": {"name": "robotwin"}, "run": {}}
    shard = _shard("robotwin/seed=1", {}, 1)
    pinned = coord._shard_config(base, shard, "ws://h:1", "/tmp/out", gpu=3, pin_gpu=True)
    unpinned = coord._shard_config(base, shard, "ws://h:1", "/tmp/out", gpu=3, pin_gpu=False)
    assert pinned["run"]["cuda_visible_devices"] == "3"
    assert "cuda_visible_devices" not in unpinned["run"]


def test_run_does_not_retry_a_unit_that_wrote_every_episode(monkeypatch: Any, tmp_path: Any) -> None:
    """A non-zero exit code after all episodes were recorded must not re-run the unit."""
    import json
    from pathlib import Path

    from loongforge.embodied.eval.orchestrator import coordinator as coord

    launches: dict[int, int] = {}

    def fake_launch(config: Any, shard: Any, index: int, url: str, output_dir: Any, tmp: Any) -> _FakeProc:
        launches[index] = launches.get(index, 0) + 1
        worker = Path(output_dir) / f"worker_{index}"
        worker.mkdir(parents=True, exist_ok=True)
        patch = (shard.get("overrides") or {}).get("benchmark") or {}
        episode = int(patch.get("episode_offset") or patch.get("start_seed") or index)
        record = {"episode_id": f"task/{episode}", "success": False}
        (worker / "results.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        return _FakeProc(1)  # benchmark exits non-zero on a model failure

    monkeypatch.setattr(coord, "_launch_unit", fake_launch)
    monkeypatch.setattr(coord, "start_server", lambda args, env=None: _FakeProc(0))
    monkeypatch.setattr(coord, "wait_healthz", lambda *args, **kwargs: None)
    monkeypatch.setattr(coord, "_aggregate", lambda *args, **kwargs: {})
    monkeypatch.setattr(coord, "short_shards", lambda *args, **kwargs: [])

    config = {
        "benchmark": {"name": "libero", "suite": "libero_10", "task_ids": [0], "episodes_per_task": 2},
        "servers": [{"gpus": [0]}],
        "pools": {"cpu_sim": {"n_sims_per_server": 1}},
        "run": {"output_dir": str(tmp_path)},
    }
    result = coord.run(config, "")
    assert all(count == 1 for count in launches.values())  # no retry
    assert result["failed_units"] == []


def test_shard_config_makes_ckpt_setting_unique_per_worker() -> None:
    """RoboTwin's official result dir is keyed by ckpt_setting; workers must not collide."""
    from loongforge.embodied.eval.orchestrator import coordinator as coord

    base = {"benchmark": {"name": "robotwin"}, "run": {"ckpt_setting": "pi05_robotwin2"}}
    shard = _shard("robotwin/seed=1", {}, 1)
    unique = coord._shard_config(base, shard, "ws://h:1", "/tmp/out", worker_index=3)
    plain = coord._shard_config(base, shard, "ws://h:1", "/tmp/out")
    assert unique["run"]["ckpt_setting"] == "pi05_robotwin2_w3"
    assert plain["run"]["ckpt_setting"] == "pi05_robotwin2"
