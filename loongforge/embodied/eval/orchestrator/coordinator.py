# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Coordinator: one command drives a G-server x N-sim evaluation with dynamic dispatch.

A config is *coordinated* when it carries ``coordinator`` / ``servers`` / ``pools``
sections; ``run.py`` dispatches such configs here. The coordinator:

1. starts one policy server per GPU (from ``servers[0].gpus``),
2. asks the benchmark family's planner (``planning.py``) for the shards,
3. runs a **bounded pool** of ``len(server_gpus) * pool.n_sims_per_server``
   concurrent unit processes, pulling the next shard from a global queue as one
   finishes -- so a fast slot picks up the next unit instead of idling (this is
   what flattens the long tail; a static one-shard-per-slot launch does not),
   pinning each worker's CUDA device for GPU simulators (``co_locate`` or
   ``separate``; see ``pools.gpu_sim``), and
4. aggregates the per-unit results and checks coverage.

Each unit is an ordinary ``orchestrator.run`` subprocess pointed at a shared
server via ``server.external_url``. Failed units are retried once, and shards
that come up short are re-run (up to ``MAX_REPAIR_ROUNDS``) instead of aborting.
"""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from loongforge.embodied.eval.orchestrator.config import apply_config
from loongforge.embodied.eval.orchestrator.server_manager import (
    build_argparser as build_server_argparser,
    start_server,
    wait_healthz,
)
from loongforge.embodied.eval.orchestrator.planning import Shard, benchmark_family, plan_shards
from loongforge.embodied.eval.transport.client import PolicyClient
from loongforge.embodied.eval.orchestrator.validation import (
    format_report,
    read_records_by_unit,
    short_shards,
    validate,
)

PORT_STRIDE = 20
DEFAULT_PORT_BASE = 6000  # outside the kernel ephemeral range (10000-61000), see docs §4
GPU_SIM_FAMILIES = {"robotwin", "simplerenv", "maniskill"}
PLACEMENT_MODES = {"co_locate", "separate"}
UNITS_PER_SLOT = 4  # shards per worker slot: slack so fast slots can steal work
DEFAULT_MAX_RETRIES = 2  # episode-level infra retries (episode_error / timeouts) per unit
MAX_REPAIR_ROUNDS = 3  # auto re-dispatch rounds for episodes the plan asked for but that never ran
WATCHDOG_INTERVAL_SEC = 15.0  # how often the coordinator checks its servers are alive
WATCHDOG_RESTART_TIMEOUT_SEC = 600.0  # cap on waiting for a restarted server to get healthy
WATCHDOG_PROBE_INTERVAL_SEC = 60.0  # how often a live server is probed with a real ping
WATCHDOG_UNHEALTHY_LIMIT = 2  # consecutive failed probes before a hung server is restarted
MAX_SERVER_RESTARTS = 3  # a server that keeps dying is not worth restarting forever
PROBE_CONNECT_TIMEOUT_SEC = 3.0  # the probe must never stall the watchdog for long
PROBE_PING_TIMEOUT_SEC = 15.0  # a ping only waits behind ~N queued inferences (~1s at N=16)
# PolicyClient pops these while connecting (a proxy breaks a local websocket); the
# probe runs inside the coordinator, so put them back afterwards.
_PROXY_ENV_KEYS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
UNIT_TIMEOUT_SLACK_SEC = 300  # head-room over the episode budget before a unit is killed
POLICY_TIMEOUT_SAFETY = 2.0  # queue wait is (N-1) x service; keep 2x head-room
MAX_ATTEMPTS = 2  # a failed unit is re-queued once before it is reported as failed


def _server_env(index: int, gpu: int) -> Dict[str, str]:
    """Env overrides for one server: its GPU, plus a per-server timing sink.

    Calibration enables server timing for the whole coordinator; without the
    suffix every server would write the same file and clobber the counters.
    """
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    timing_base = os.environ.get("LOONGFORGE_EVAL_TIMING_PATH")
    if timing_base:
        env["LOONGFORGE_EVAL_TIMING_PATH"] = f"{timing_base}.server{index}"
    return env


def _shard_config(
    config: Dict[str, Any],
    shard: Shard,
    external_url: str,
    output_dir: str,
    gpu: int | None = None,
    pin_gpu: bool = False,
    worker_index: int | None = None,
) -> Dict[str, Any]:
    """Build one unit's config: a copy of the base, minus coordinator, plus this shard's patch.

    Benchmark-agnostic -- the planner already expressed the slice as config overrides.
    ``pin_gpu`` injects the worker's CUDA device for GPU simulators; without it a
    SAPIEN-based benchmark silently lands every worker on GPU 0.
    ``worker_index`` makes ``run.ckpt_setting`` unique: RoboTwin names its official
    result directory by ``(ckpt_setting, second)``, so concurrent workers would
    otherwise overwrite each other's ``_result.txt``.
    """
    shard_cfg = copy.deepcopy(config)
    for key in ("coordinator", "servers", "pools"):
        shard_cfg.pop(key, None)
    for section, values in (shard.get("overrides") or {}).items():
        shard_cfg.setdefault(section, {}).update(values)
    benchmark = shard_cfg.setdefault("benchmark", {})
    for key in ("suites", "suite_overrides"):  # coordinator-only keys
        benchmark.pop(key, None)
    run_cfg = shard_cfg.setdefault("run", {})
    run_cfg["output_dir"] = output_dir
    run_cfg["timestamped_output"] = False  # deterministic path so the coordinator can aggregate
    if worker_index is not None:
        base_setting = str(run_cfg.get("ckpt_setting") or "unit")
        run_cfg["ckpt_setting"] = f"{base_setting}_w{worker_index}"
    if pin_gpu and gpu is not None:
        run_cfg["cuda_visible_devices"] = str(gpu)
    # Retry infra-level episode failures (episode_error / timeouts / server_unreachable) by
    # default: a transient sim<->server disconnect must not be scored as a model failure.
    # Model failures (not_successful_within_max_steps) are never retried.
    run_cfg.setdefault("max_retries", DEFAULT_MAX_RETRIES)
    server_cfg = shard_cfg.setdefault("server", {})
    server_cfg["external_url"] = external_url
    if external_url:
        # Some runners (e.g. RoboTwin) build their client from server.host/server.port
        # and ignore external_url, so derive both from the URL the coordinator assigned.
        parsed = urlparse(external_url)
        if parsed.hostname:
            server_cfg["host"] = parsed.hostname
        if parsed.port:
            server_cfg["port"] = parsed.port
    # external_url=None means "no policy server" (offline expert-check scan):
    # the worker never builds a client, so host/port stay untouched -- and the
    # unit must not start one either (a scan needs no model at all).
    if not external_url and (shard_cfg.get("benchmark") or {}).get("offline_expert_check_scan"):
        server_cfg["disabled"] = True
    return shard_cfg


def _aggregate(output_dir: Path, n_units: int) -> Dict[str, Any]:
    """Merge worker_*/results.jsonl into one deduped summary."""
    seen: Dict[str, bool] = {}
    for i in range(n_units):
        path = output_dir / f"worker_{i}" / "results.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "episode_id" in record:
                seen[str(record["episode_id"])] = bool(record.get("success"))
    total = len(seen)
    successes = sum(seen.values())
    return {"n_episodes": total, "successes": successes, "success_rate": (successes / total) if total else None}


def _warn_policy_timeout(config: Dict[str, Any], n_per_server: int) -> None:
    """Warn when the per-call timeout is below the queue-wait floor.

    A server serialises requests, so the last of N sims waits about ``(N-1) * t_service``
    per step. If ``timeouts.policy_call_ms`` is below ``N * service_ms * safety`` the
    units hit spurious ``policy_timeout`` failures that look like model failures.
    ``timeouts.service_ms`` is the mean per-request service time from the calibration
    report (``service_mean_ms``); without it the check is skipped.
    """
    timeouts = config.get("timeouts") or {}
    service_ms = timeouts.get("service_ms")
    timeout_ms = timeouts.get("policy_call_ms")
    if not service_ms or not timeout_ms:
        return
    floor_ms = n_per_server * float(service_ms) * POLICY_TIMEOUT_SAFETY
    if float(timeout_ms) < floor_ms:
        print(
            f"[coordinator] WARNING: timeouts.policy_call_ms={timeout_ms}ms is below the queue-wait floor "
            f"{floor_ms:.0f}ms (N={n_per_server} x service={service_ms}ms x {POLICY_TIMEOUT_SAFETY}) -- "
            "units will hit spurious policy_timeout failures; raise it",
            file=sys.stderr,
            flush=True,
        )


def _resolve_pool(config: Dict[str, Any]) -> tuple[str, Dict[str, Any], bool]:
    """Return (pool_name, pool_config, is_gpu_sim) for this benchmark family.

    A pool written under the other family's key is invisible to the lookup below,
    so say so instead of letting it fall through to ``N=1``: a LIBERO config
    copied from a RoboTwin template (``pools.gpu_sim``) then just runs slow.
    """
    pools = config.get("pools") or {}
    is_gpu_sim = benchmark_family(config) in GPU_SIM_FAMILIES
    name, other = ("gpu_sim", "cpu_sim") if is_gpu_sim else ("cpu_sim", "gpu_sim")
    pool = pools.get(name) or {}
    if not pool and pools.get(other):
        print(
            f"[coordinator] pools.{other} is set but this benchmark family reads pools.{name}; "
            f"ignoring pools.{other} (N falls back to coordinator.n_workers or 1)",
            flush=True,
        )
    return name, pool, is_gpu_sim


def _resolve_placement(config: Dict[str, Any]) -> tuple[List[int], int, List[int], str, bool]:
    """Resolve (server_gpus, n_sims_per_server, sim_gpus, placement, is_gpu_sim).

    Backwards compatible: without a pool section the worker count falls back to
    ``coordinator.n_workers`` and sims are co-located with their server.
    """
    servers = config.get("servers") or []
    coordinator = config.get("coordinator") or {}
    server_gpus = (servers[0].get("gpus") if servers else None) or coordinator.get("gpus") or [0]
    server_gpus = [int(g) for g in server_gpus]

    pools = config.get("pools") or {}
    _pool_name, pool, is_gpu_sim = _resolve_pool(config)
    n_per_server = (
        pool.get("n_sims_per_server")
        or (pools.get("cpu_sim") or {}).get("n_sims_per_server")
        or coordinator.get("n_workers")
        or 1
    )
    n_per_server = max(1, int(n_per_server))

    mode = str(pool.get("placement") or config.get("placement") or "co_locate")
    if mode not in PLACEMENT_MODES:
        raise ValueError(f"placement must be one of {sorted(PLACEMENT_MODES)}, got {mode!r}")

    if mode == "co_locate" or not is_gpu_sim:
        sim_gpus = list(server_gpus)
    else:
        sim_gpus = [int(g) for g in (pool.get("sim_gpus") or config.get("sim_gpus") or [])]
        if not sim_gpus:
            raise ValueError("placement='separate' requires pools.gpu_sim.sim_gpus (the GPUs that host sims)")
        overlap = sorted(set(sim_gpus) & set(server_gpus))
        if overlap:
            raise ValueError(f"placement='separate' but these GPUs host both a server and sims: {overlap}")
    return server_gpus, n_per_server, sim_gpus, mode, is_gpu_sim


def _build_slots(
    server_gpus: List[int], n_per_server: int, sim_gpus: List[int], mode: str, urls: List[str]
) -> List[tuple[str, int]]:
    """One (server_url, sim_gpu) pair per worker slot.

    ``co_locate`` pins a server's workers to that server's GPU; ``separate``
    spreads all workers over ``sim_gpus`` while keeping the server URLs paired
    round-robin.

    Slots are laid out **sim-layer major** (one slot per GPU, then the next
    layer) rather than grouped by GPU. ``_run_batch`` pops free slots from the
    end of the list, so this ordering makes the first ``len(urls)`` dispatches
    land on distinct GPUs: a job with fewer units than slots still spreads
    across every card instead of packing onto the highest-numbered ones. For a
    full load (units >= slots) the slot *set* is unchanged, so throughput is
    identical.
    """
    slots: List[tuple[str, int]] = []
    for layer in range(n_per_server):
        for index, url in enumerate(urls):
            gpu = server_gpus[index] if mode == "co_locate" else sim_gpus[(layer * len(urls) + index) % len(sim_gpus)]
            slots.append((url, gpu))
    return slots


def _validate_gpu_resources(config: Dict[str, Any], n_per_server: int, is_gpu_sim: bool) -> None:
    """Fail fast when the requested sim count cannot fit next to the model.

    Only runs when the config provides the hardware facts; the numbers come
    from the calibration report (``resources.gpu_sim``).
    """
    if not is_gpu_sim:
        return
    resources = (config.get("resources") or {}).get("gpu_sim") or {}
    vram_total = resources.get("vram_total_mb")
    vram_per_sim = resources.get("vram_per_sim_mb")
    vram_model = resources.get("vram_model_mb")
    if vram_model is None and resources.get("vram_model_gb") is not None:
        vram_model = float(resources["vram_model_gb"]) * 1024
    if not (vram_total and vram_per_sim and vram_model is not None):
        return
    need = n_per_server * float(vram_per_sim) + float(vram_model)
    if need > float(vram_total):
        raise ValueError(
            f"GPU-sim placement needs {need:.0f} MiB per card "
            f"(N={n_per_server} x {vram_per_sim} MiB + model {vram_model:.0f} MiB) "
            f"but resources.gpu_sim.vram_total_mb is {vram_total}"
        )



def _assert_ports_free(ports: List[int]) -> None:
    """Fail fast when a service or health port is already taken.

    Binding is the only reliable test: a killed server can leave its socket
    behind with no visible owner (the listener then shows up as ``*:PORT`` in
    ``ss`` but no pid), and the run would otherwise spend the whole server
    startup budget before reporting "Policy server exited before healthy: 1".
    ``SO_REUSEADDR`` is kept so TIME_WAIT sockets do not count as conflicts.
    """
    conflicts: List[int] = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                # Bind every interface: the policy server listens on 0.0.0.0 even
                # when server.host is 127.0.0.1.
                sock.bind(("", int(port)))
            except OSError:
                conflicts.append(int(port))
    if conflicts:
        raise SystemExit(
            f"these ports are already in use: {conflicts} (first requested {ports[0]}); "
            "stop the process holding them or raise LOONFORGE_COORD_PORT_BASE"
        )


def _launch_unit(
    config: Dict[str, Any],
    shard: tuple,
    index: int,
    external_url: str,
    output_dir: Path,
    tmp: Path,
    gpu: int | None = None,
    pin_gpu: bool = False,
) -> subprocess.Popen:
    """Launch one unit as an ``orchestrator.run`` subprocess pointed at its server."""
    shard_cfg = _shard_config(
        config, shard, external_url, str(output_dir / f"worker_{index}"),
        gpu=gpu, pin_gpu=pin_gpu, worker_index=index,
    )
    cfg_path = tmp / f"unit_{index}.json"
    cfg_path.write_text(json.dumps(shard_cfg), encoding="utf-8")
    # Resume runs refuse a missing output_dir (typo protection). A resumed plan can
    # assign a new worker index, so create the directory before the unit starts.
    (output_dir / f"worker_{index}").mkdir(parents=True, exist_ok=True)
    log = open(output_dir / f"worker_{index}.log", "a", buffering=1)
    return subprocess.Popen(
        [sys.executable, "-m", "loongforge.embodied.eval.orchestrator.run", "--config", str(cfg_path)],
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _unit_timeout_sec(config: Dict[str, Any], shards: List[tuple]) -> float:
    """Wall-clock budget for one unit.

    A unit runs ``episodes_per_task`` episodes of one task, so its budget is that
    many ``per_episode_sec`` timeouts plus connection setup and slack. This is a
    backstop against a unit that hangs in native code (where the runner's SIGALRM
    cannot fire) -- without it ``_await_slot`` would block forever. Override with
    ``run.unit_timeout_sec``.
    """
    explicit = float((config.get("run") or {}).get("unit_timeout_sec") or 0)
    if explicit > 0:
        return explicit
    per_episode = float((config.get("timeouts") or {}).get("per_episode_sec") or 1800)
    connect = float((config.get("server") or {}).get("connect_timeout_sec") or 60)
    max_episodes = max((int(shard["expected"]) for shard in shards), default=1)
    return max_episodes * per_episode + connect + UNIT_TIMEOUT_SLACK_SEC


def _await_slot(
    pool: Dict[subprocess.Popen, tuple], started_at: Dict[int, float], timeout_sec: float
) -> Tuple[subprocess.Popen, bool]:
    """Block until a unit finishes or blows its wall-clock budget.

    Returns ``(proc, timed_out)``; a timed-out unit is killed here so its slot frees
    and it goes through the usual retry/repair path.
    """
    while True:
        for proc in list(pool):
            if proc.poll() is not None:
                return proc, False
        if timeout_sec > 0:
            now = time.time()
            for proc, (_, index) in list(pool.items()):
                if now - started_at[index] > timeout_sec:
                    print(f"[coordinator] unit {index} exceeded {timeout_sec:.0f}s; killing it", flush=True)
                    proc.kill()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                    return proc, True
        time.sleep(1)


def _terminate(proc: subprocess.Popen, timeout: float = 30) -> None:
    """Terminate a process, escalating to SIGKILL if it does not exit."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


class _ServerWatchdog(threading.Thread):
    """Restart a policy server that has died, in place (same port).

    Units connect through ``server.external_url``, so they cannot restart the
    server they depend on -- only its owner (the coordinator) can. Restarting on
    the *same* port keeps every unit's URL valid, so in-flight units just
    reconnect on their next retry.
    """

    def __init__(
        self,
        servers: List[tuple],
        gpus: List[int],
        interval_sec: float = WATCHDOG_INTERVAL_SEC,
        probe_interval_sec: float = WATCHDOG_PROBE_INTERVAL_SEC,
    ) -> None:
        super().__init__(daemon=True, name="server-watchdog")
        self._servers = servers  # list of (proc, args); replaced in place on restart
        self._gpus = gpus
        self._interval = interval_sec
        self._probe_interval = probe_interval_sec
        self._stop_event = threading.Event()  # NB: not ``_stop`` -- Thread reserves that name
        self._unhealthy: Dict[int, int] = {}
        self._last_probe: Dict[int, float] = {}
        self._restart_attempts: Dict[int, int] = {}
        self.given_up: set = set()
        self.restarts = 0

    def stop(self) -> None:
        """Run stop."""
        self._stop_event.set()

    def run(self) -> None:
        """Poll each server: restart a dead process, and a live-but-hung one."""
        while not self._stop_event.wait(self._interval):
            now = time.time()
            for index, (proc, args) in enumerate(self._servers):
                if proc.poll() is not None:
                    self._unhealthy[index] = 0
                    self._restart(index)
                    continue
                # A live process is not necessarily a *serving* one: the healthz endpoint
                # is a separate thread, so it stays green even if the policy loop is
                # wedged. Only a real request proves liveness.
                if now - self._last_probe.get(index, 0.0) < self._probe_interval:
                    continue
                self._last_probe[index] = now
                if self._alive(args):
                    self._unhealthy[index] = 0
                    continue
                self._unhealthy[index] = self._unhealthy.get(index, 0) + 1
                if self._unhealthy[index] >= WATCHDOG_UNHEALTHY_LIMIT:
                    self._unhealthy[index] = 0
                    self._restart(index)

    def _alive(self, args: Any) -> bool:
        """A bounded ping round-trip against a live server."""
        saved = {key: os.environ[key] for key in _PROXY_ENV_KEYS if key in os.environ}
        client = None
        try:
            client = PolicyClient(host=args.host, port=args.port, timeout=PROBE_CONNECT_TIMEOUT_SEC)
            client.ping(timeout=PROBE_PING_TIMEOUT_SEC)
            return True
        except Exception:
            return False
        finally:
            os.environ.update(saved)  # PolicyClient pops these; restore them
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def _restart(self, index: int) -> None:
        """Run _restart."""
        if self._stop_event.is_set() or index in self.given_up:
            return
        self._restart_attempts[index] = self._restart_attempts.get(index, 0) + 1
        if self._restart_attempts[index] > MAX_SERVER_RESTARTS:
            # A server that cannot come back (bad GPU, CUDA OOM, broken ckpt) is not
            # worth reloading forever -- stop, say so, and let the run report it.
            self.given_up.add(index)
            print(
                f"[watchdog] server {index} failed to come back {MAX_SERVER_RESTARTS} times; giving up "
                "-- this run needs a human",
                file=sys.stderr,
                flush=True,
            )
            return
        old_proc, args = self._servers[index]
        if old_proc.poll() is None:
            # Hung, not dead: it still holds the port, so it has to go before we can
            # rebind -- otherwise every restart fails with "address already in use".
            print(f"[watchdog] server {index} is hung; terminating it before restart", flush=True)
            _terminate(old_proc)
        print(f"[watchdog] server {index} died; restarting on port {args.port}", flush=True)
        try:
            proc = start_server(args, _server_env(index, self._gpus[index]))
            wait_healthz(
                f"http://{args.host}:{args.health_port}/healthz",
                min(args.server_start_timeout_sec, WATCHDOG_RESTART_TIMEOUT_SEC),
                proc,
            )
        except Exception as exc:  # a failed restart is retried on the next tick
            print(f"[watchdog] restart of server {index} failed: {exc}", flush=True)
            return
        if self._stop_event.is_set():
            # We were shut down while this restart was in flight: the run's cleanup
            # has already terminated the old servers, so terminate this one too
            # rather than leaking a process that would hold a GPU.
            _terminate(proc)
            return
        self._servers[index] = (proc, args)
        self.restarts += 1
        print(f"[watchdog] server {index} back up (pid={proc.pid})", flush=True)


def _unit_completed(output_dir: Path, index: int, expected: int) -> bool:
    """True when a unit already wrote the episodes the plan asked for.

    Some official evaluators exit non-zero for a *model* failure (e.g. RoboTwin
    prints ``Fail!``) even though every episode ran and was recorded. Re-running
    such a unit wastes a full episode budget and produces duplicate records.
    """
    path = output_dir / f"worker_{index}" / "results.jsonl"
    if not path.exists():
        return False
    seen = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "episode_id" in record:
            seen.add(str(record["episode_id"]))
    return len(seen) >= max(1, int(expected or 1))



def _run_batch(
    config: Dict[str, Any],
    shards: List[tuple],
    indices: List[int],
    slots: List[tuple[str, int]],
    output_dir: Path,
    tmp: Path,
    timeout_sec: float = 0.0,
    pin_gpu: bool = False,
) -> Tuple[Dict[int, int], List[int]]:
    """Run ``indices`` (into ``shards``) through a bounded pool fed by a global queue.

    ``slots`` pairs each worker slot with its server URL and sim GPU. A unit that
    exits non-zero -- or that blows its ``timeout_sec`` budget -- is re-queued
    once, then reported as failed. Returns ``(returncodes_by_index, failed_indices)``.
    """
    returncodes: Dict[int, int] = {}
    failures: List[int] = []
    pending = list(indices)
    free_slots = list(range(len(slots)))
    attempts: Dict[int, int] = {}
    started_at: Dict[int, float] = {}
    pool: Dict[subprocess.Popen, tuple] = {}
    while pending or pool:
        while pending and free_slots:
            slot = free_slots.pop()
            index = pending.pop(0)
            url, gpu = slots[slot]
            started_at[index] = time.time()
            if pin_gpu:
                proc = _launch_unit(
                    config, shards[index], index, url, output_dir, tmp, gpu=gpu, pin_gpu=True
                )
            else:
                proc = _launch_unit(config, shards[index], index, url, output_dir, tmp)
            pool[proc] = (slot, index)
        finished, timed_out = _await_slot(pool, started_at, timeout_sec)
        slot, index = pool.pop(finished)
        free_slots.append(slot)
        returncodes[index] = finished.returncode
        expected = int((shards[index] or {}).get("expected") or 1)
        completed = _unit_completed(output_dir, index, expected)
        if (finished.returncode != 0 or timed_out) and not completed:
            attempts[index] = attempts.get(index, 0) + 1
            if attempts[index] < MAX_ATTEMPTS:
                pending.append(index)  # re-queue once; a fresh unit may just be a flaky sim
            else:
                failures.append(index)
        # A unit that wrote every expected episode is done even when the
        # benchmark's own exit code is non-zero (model failure, not infra).
    return returncodes, failures


def run(config: Dict[str, Any], config_path: str = "") -> Dict[str, Any]:
    """Run a coordinated (G server : G x N worker) evaluation with dynamic dispatch."""
    server_gpus, n_per_server, sim_gpus, placement, is_gpu_sim = _resolve_placement(config)
    concurrency = max(1, len(server_gpus) * n_per_server)
    _validate_gpu_resources(config, n_per_server, is_gpu_sim)
    plan = plan_shards(config, concurrency)
    expected_by_shard = {shard["id"]: int(shard["expected"]) for shard in plan}
    shards_by_id = {shard["id"]: shard for shard in plan}

    output_dir = Path(str((config.get("run") or {}).get("output_dir") or "/tmp/vla_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    finished = output_dir / "coordinator_result.json"
    if finished.exists() and not (config.get("run") or {}).get("resume"):
        raise SystemExit(
            f"{finished} already exists -- refusing to overwrite a finished run. "
            "Use a fresh run.output_dir, or set run.resume: true to continue it."
        )
    _warn_policy_timeout(config, n_per_server)

    base_args = apply_config(build_server_argparser().parse_args([]), config)
    base_args.config = config_path

    # Offline expert-check scan needs no policy at all: skip starting servers
    # (saves the model load of G servers and leaves the GPUs to the sims).
    # Only the TOP-LEVEL config (which carries pools) runs serverless; a unit
    # subprocess inherits the scan key but must still take the normal unit path.
    scan_mode = bool(
        (config.get("benchmark") or {}).get("offline_expert_check_scan")
    ) and bool(config.get("pools"))
    port_base = int(os.environ.get("LOONGFORGE_COORD_PORT_BASE", str(DEFAULT_PORT_BASE)))
    servers = []
    if not scan_mode:
        _assert_ports_free(
            [port_base + PORT_STRIDE * index + offset
             for index in range(len(server_gpus)) for offset in (0, 1)]
        )
    for index, gpu in ([] if scan_mode else enumerate(server_gpus)):
        port = port_base + PORT_STRIDE * index
        health_port = port + 1
        server_log = str(output_dir / f"policy_server_{index}.log")
        # The policy server reads its port from the config file, so a per-server
        # config is the only way to give each server its own port (``--port`` is
        # not a CLI flag of ``loongforge_server``).
        server_cfg = copy.deepcopy(config)
        server_cfg.setdefault("server", {})
        server_cfg["server"]["port"] = port
        server_cfg["server"]["health_port"] = health_port
        server_cfg["server"]["log"] = server_log
        server_cfg_path = output_dir / f"server_{index}.json"
        server_cfg_path.write_text(json.dumps(server_cfg), encoding="utf-8")

        server_args = copy.copy(base_args)
        server_args.config = str(server_cfg_path)
        server_args.port = port
        server_args.health_port = health_port
        server_args.server_log = server_log
        servers.append((start_server(server_args, _server_env(index, gpu)), server_args))
    urls = [f"ws://{args.host}:{args.port}" for _, args in servers]
    if scan_mode:
        # One (no-server) slot per sim slot, pinned to the sim GPU.
        slots = [(None, gpu) for gpu in sim_gpus for _ in range(n_per_server)]
    else:
        slots = _build_slots(server_gpus, n_per_server, sim_gpus, placement, urls)

    unit_timeout = _unit_timeout_sec(config, plan)
    all_shards: List[Shard] = list(plan)
    failed_units: List[int] = []
    repair_rounds = 0
    # watchdog knobs (mainly so a test can shrink the intervals)
    coordinator_cfg = config.get("coordinator") or {}
    watchdog = _ServerWatchdog(  # started once the servers are healthy
        servers,
        server_gpus,
        interval_sec=float(coordinator_cfg.get("watchdog_interval_sec") or WATCHDOG_INTERVAL_SEC),
        probe_interval_sec=float(coordinator_cfg.get("watchdog_probe_interval_sec") or WATCHDOG_PROBE_INTERVAL_SEC),
    )
    watchdog_started = False
    started = time.time()
    try:
        for proc, args in servers:
            wait_healthz(f"http://{args.host}:{args.health_port}/healthz", args.server_start_timeout_sec, proc)
        started = time.time()  # worker wall only (exclude model load)
        if servers:
            watchdog.start()
            watchdog_started = True

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _, failed_units = _run_batch(
                config, all_shards, list(range(len(all_shards))), slots, output_dir, tmp, unit_timeout,
                pin_gpu=is_gpu_sim,
            )
            # Coverage gate: re-run the shards that came up short rather than aborting.
            # A re-run lands in a fresh worker dir, so the overlap is deduped by
            # ``episode_id`` (and a runner with resume only redoes what is missing).
            # Only a still-incomplete run after MAX_REPAIR_ROUNDS needs a human.
            for _ in range(MAX_REPAIR_ROUNDS):
                records = read_records_by_unit(output_dir, len(all_shards))
                launched = [shard["id"] for shard in all_shards]
                short = short_shards(expected_by_shard, launched, records)
                if not short:
                    break
                first = len(all_shards)
                all_shards.extend(shards_by_id[shard_id] for shard_id in short)
                _, extra_failed = _run_batch(
                    config, all_shards, list(range(first, len(all_shards))), slots, output_dir, tmp,
                    unit_timeout, pin_gpu=is_gpu_sim,
                )
                failed_units.extend(extra_failed)
                repair_rounds += 1
    finally:
        watchdog.stop()
        try:
            # Only join if it was actually started: joining a never-started thread raises,
            # and that exception would skip the server teardown below (leaking GPUs).
            if watchdog_started:
                watchdog.join(timeout=WATCHDOG_INTERVAL_SEC + 5)
        finally:
            for proc, _ in servers:
                _terminate(proc)

    launched_ids = [shard["id"] for shard in all_shards]
    validation = validate(expected_by_shard, launched_ids, read_records_by_unit(output_dir, len(all_shards)))
    print(f"[coordinator] {format_report(validation)}", flush=True)
    if validation["unexpected"]:
        print(
            f"[coordinator] note: {validation['unexpected']} episode(s) ran outside the planned seed range "
            "(replacements for skipped unstable seeds)",
            flush=True,
        )
    if not validation["ok"]:
        print(
            f"[coordinator] WARNING: coverage still incomplete after {repair_rounds} repair round(s) "
            "-- this run needs a human; the score is partial",
            file=sys.stderr,
            flush=True,
        )
    elif validation["infra_failed"]:
        print(
            f"[coordinator] WARNING: {validation['infra_failed']} episode(s) failed at the infra level "
            "(they did not really run) -- the score understates the model; see validation.failures",
            file=sys.stderr,
            flush=True,
        )
    # Persist the plan + launch order so ``python -m ...validation`` can re-check offline.
    (output_dir / "plan.json").write_text(json.dumps({"shards": plan, "launched": launched_ids}), encoding="utf-8")
    result = {
        "benchmark": (config.get("benchmark") or {}).get("suites")
        or (config.get("benchmark") or {}).get("suite")
        or (config.get("benchmark") or {}).get("name"),
        "gpus": server_gpus,
        "sim_gpus": sim_gpus,
        "placement": placement,
        "n_sims_per_server": n_per_server,
        "concurrency": concurrency,
        "n_units": len(plan),
        "n_units_launched": len(all_shards),
        "repair_rounds": repair_rounds,
        "unit_timeout_sec": unit_timeout,
        "failed_units": failed_units,
        "server_restarts": watchdog.restarts,
        "servers_given_up": sorted(watchdog.given_up),
        "elapsed_sec": round(time.time() - started, 1),
        "output_dir": str(output_dir),
        "validation": validation,
        "summary": _aggregate(output_dir, len(all_shards)),
    }
    (output_dir / "coordinator_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
