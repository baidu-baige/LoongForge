# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Formal calibration entry: sweep N, build the report, pick a fallback N.

CPU-sim (LIBERO / CALVIN) and GPU-sim (RoboTwin / SimplerEnv / ManiSkill) share
the same command:

    python -m loongforge.embodied.eval.orchestrator.calibration.calibrate --config run.yaml --n 1,2,4,8 --gpus 0,1,2,3,4,5,6,7

- **CPU-sim**: one coordinator per N, each pinned to its own GPU, launched in
  parallel; N is the per-server worker count.
- **GPU-sim**: N = sims per server, and every point uses all ``--gpus`` (one
  server per GPU). Points run **sequentially** because each one needs the whole
  GPU set; each point records per-GPU peak memory.

Both modes collect throughput, server busy fraction and service time into the
report produced by ``report.py``, derive a fallback N (the smallest
N reaching >=95% of the best throughput), validate it and cache it keyed by
(model, benchmark, gpu).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Sequence

from loongforge.embodied.eval.orchestrator.calibration.config import (
    calibration_key,
    save_calibration,
    validate_parallel_config,
)
from loongforge.embodied.eval.orchestrator.calibration.report import build_report, render_markdown
from loongforge.embodied.eval.orchestrator.planning import benchmark_family

GPU_SIM_FAMILIES = {"robotwin", "simplerenv", "maniskill"}
DEFAULT_PORT_BASE = 6000


def is_gpu_sim(config: Dict[str, Any]) -> bool:
    """True for SAPIEN-family benchmarks (sims consume GPU memory)."""
    return benchmark_family(config) in GPU_SIM_FAMILIES


def fallback_n(rows: Sequence[Dict[str, Any]], plateau_frac: float = 0.95) -> Any:
    """Smallest N whose throughput is >= ``plateau_frac`` of the best measured."""
    best = max((r.get("throughput_ep_per_min") or 0.0) for r in rows) if rows else 0.0
    if not best:
        return None
    for row in sorted(rows, key=lambda r: r["N"]):
        if (row.get("throughput_ep_per_min") or 0.0) >= plateau_frac * best:
            return row["N"]
    return None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_timing(path: Path) -> Dict[str, Any]:
    """Aggregate one point's server timing files.

    With multiple servers the coordinator writes ``<path>.server<i>`` per server
    (otherwise they would clobber each other); a single server falls back to the
    plain path.
    """
    files = sorted(path.parent.glob(f"{path.name}.server*")) if path.parent.exists() else []
    if not files and path.exists():
        files = [path]
    count = 0
    weighted = 0.0
    max_ms = 0.0
    for timing_file in files:
        data = _read_json(timing_file)
        n = data.get("count") or 0
        mean = data.get("mean") or 0.0
        count += n
        weighted += n * mean
        max_ms = max(max_ms, float(data.get("max") or 0.0))
    if not files:
        return {}
    return {
        "count": count or None,
        "mean": (weighted / count) if count else None,
        "max": max_ms or None,
        "servers": len(files),
    }


def _sweep_rows(ns: Sequence[int], root: Path) -> List[Dict[str, Any]]:
    """Collect per-N metrics from each coordinator's result + server timing."""
    rows: List[Dict[str, Any]] = []
    for n in ns:
        result = _read_json(root / f"N{n}" / "out" / "coordinator_result.json")
        timing = _read_timing(root / f"N{n}" / "timing.json")
        peak = _read_json(root / f"N{n}" / "gpu_peak.json")
        elapsed = result.get("elapsed_sec")
        summary = result.get("summary") or {}
        episodes = summary.get("n_episodes")
        requests, mean_ms = timing.get("count"), timing.get("mean")
        busy = round(requests * mean_ms / 1000.0 / elapsed, 3) if (requests and mean_ms and elapsed) else None
        rows.append({
            "N": n,
            "workers": result.get("concurrency") or result.get("n_workers"),
            "wall_sec": elapsed,
            "episodes": episodes,
            "successes": summary.get("successes"),
            "success_rate": summary.get("success_rate"),
            "requests": requests,
            "throughput_ep_per_min": round(episodes / elapsed * 60, 2) if (episodes and elapsed) else None,
            "server_busy_frac": busy,
            "service_mean_ms": round(mean_ms, 1) if mean_ms else None,
            "gpu_peak_mb": peak or None,
        })
    return rows


def _point_config(
    config: Dict[str, Any],
    n: int,
    index: int,
    root: Path,
    *,
    gpu_sim: bool,
    gpus: Sequence[int],
) -> tuple[Dict[str, Any], Path]:
    """Build one sweep point's coordinator config (fresh output dir, N set)."""
    cfg = copy.deepcopy(config)
    cfg.pop("calibrate", None)
    out = root / f"N{n}"
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)  # coordinator refuses a finished output_dir
    out.mkdir(parents=True, exist_ok=True)

    run_cfg = cfg.setdefault("run", {})
    run_cfg["output_dir"] = str(out / "out")
    run_cfg["timestamped_output"] = False

    server = cfg.setdefault("server", {})
    server["port"] = DEFAULT_PORT_BASE
    server["health_port"] = DEFAULT_PORT_BASE + 1
    server["log"] = str(out / "policy_server.log")

    if gpu_sim:
        cfg.pop("coordinator", None)
        cfg["servers"] = [{"gpus": [int(g) for g in gpus]}]
        pool = cfg.setdefault("pools", {}).setdefault("gpu_sim", {})
        pool["n_sims_per_server"] = int(n)
        needed = len(gpus) * int(n)
        benchmark = cfg.setdefault("benchmark", {})
        episodes = int(benchmark.get("episodes_per_task") or 1)
        if episodes < needed:
            # Calibration needs one distinct seed per worker slot.
            benchmark["episodes_per_task"] = needed
    else:
        cfg["coordinator"] = {"n_workers": int(n)}

    cfg_path = root / f"N{n}.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg, cfg_path


class _GpuPeakSampler(threading.Thread):
    """Sample per-GPU memory while one sweep point runs."""

    def __init__(self, gpus: Sequence[int], out_path: Path, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self._gpus = list(gpus)
        self._out = out_path
        self._interval = interval
        self._stop_event = threading.Event()
        self._peak: Dict[str, int] = {}

    def run(self) -> None:
        while not self._stop_event.is_set():
            for gpu in self._gpus:
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    used = int(out.strip().splitlines()[0])
                except Exception:
                    continue
                self._peak[str(gpu)] = max(self._peak.get(str(gpu), 0), used)
            self._stop_event.wait(self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2)
        self._out.write_text(json.dumps(self._peak, indent=2), encoding="utf-8")


def _launch(config_path: Path, env: Dict[str, str], log: Any) -> subprocess.Popen:
    log.write(f"=== {config_path} ===\n")
    log.flush()
    return subprocess.Popen(
        [sys.executable, "-m", "loongforge.embodied.eval.orchestrator.run", "--config", str(config_path)],
        cwd=str(config_path.parent),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _timing_env(out: Path) -> Dict[str, str]:
    return {
        "LOONGFORGE_EVAL_TIMING": "1",
        "LOONGFORGE_EVAL_TIMING_PATH": str(out / "timing.json"),
        "LOONGFORGE_EVAL_TIMING_FLUSH_EVERY": "1",
    }


def _run_cpu_sweep(
    config: Dict[str, Any], config_path: str, ns: Sequence[int], gpus: Sequence[int], root: Path
) -> List[int]:
    """One coordinator per N, each on its own GPU, all in parallel."""
    procs = []
    for index, n in enumerate(ns):
        _, cfg_path = _point_config(config, n, index, root, gpu_sim=False, gpus=gpus)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpus[index % len(gpus)])
        env["LOONGFORGE_COORD_PORT_BASE"] = str(DEFAULT_PORT_BASE + 100 * index)
        env.update(_timing_env(root / f"N{n}"))
        log = open(root / f"N{n}.log", "a", buffering=1)
        procs.append(_launch(cfg_path, env, log))
    return [p.wait() for p in procs]


def _run_gpu_sweep(
    config: Dict[str, Any], config_path: str, ns: Sequence[int], gpus: Sequence[int], root: Path
) -> List[int]:
    """Every point uses all GPUs (one server per GPU), so points run sequentially."""
    returncodes: List[int] = []
    for index, n in enumerate(ns):
        _, cfg_path = _point_config(config, n, 0, root, gpu_sim=True, gpus=gpus)
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)  # coordinator assigns one GPU per server
        # Shift the base per point: servers from the previous point may still be in
        # TIME_WAIT, and rebinding the same ports would fail.
        env["LOONGFORGE_COORD_PORT_BASE"] = str(DEFAULT_PORT_BASE + 100 * index)
        env.update(_timing_env(root / f"N{n}"))
        sampler = _GpuPeakSampler(gpus, root / f"N{n}" / "gpu_peak.json")
        sampler.start()
        log = open(root / f"N{n}.log", "a", buffering=1)
        proc = _launch(cfg_path, env, log)
        returncodes.append(proc.wait())
        sampler.stop()
        log.close()
    return returncodes


def run(
    config: Dict[str, Any],
    config_path: str = "",
    ns: Sequence[int] = (1, 2, 4, 8, 16),
    gpus: Sequence[int] | None = None,
    out_dir: str = "",
) -> Dict[str, Any]:
    """Run the N sweep, build the report, and cache a fallback calibrated config."""
    root = Path(out_dir or (config.get("calibrate") or {}).get("out_dir") or "/tmp/vla_calibrate")
    root.mkdir(parents=True, exist_ok=True)
    gpu_sim = is_gpu_sim(config)
    if gpus:
        gpu_list = [int(g) for g in gpus]
    elif gpu_sim:
        gpu_list = [int(g) for g in (config.get("servers") or [{}])[0].get("gpus") or []]
        if not gpu_list:
            raise ValueError("GPU-sim calibration needs --gpus or servers[0].gpus")
    else:
        gpu_list = list(range(len(ns)))

    returncodes = (
        _run_gpu_sweep(config, config_path, ns, gpu_list, root)
        if gpu_sim
        else _run_cpu_sweep(config, config_path, ns, gpu_list, root)
    )

    rows = _sweep_rows(ns, root)
    pool_name = "gpu_sim" if gpu_sim else "cpu_sim"
    env_facts = dict((config.get("resources") or {}).get(pool_name) or {})
    if gpu_sim:
        env_facts.setdefault("gpu_count", len(gpu_list))
        env_facts.setdefault(
            "placement",
            str((config.get("pools") or {}).get("gpu_sim", {}).get("placement") or "co_locate"),
        )
    report = build_report(rows, _read_timing(root / f"N{ns[0]}" / "timing.json"), env=env_facts)
    report_dir = root / "calibration_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    if env_facts:
        (report_dir / "env.json").write_text(json.dumps(env_facts, indent=2), encoding="utf-8")

    # deterministic fallback: validate + cache it (an agent's decision supersedes).
    n_fallback = fallback_n(rows)
    pool_cfg: Dict[str, Any] = {"n_sims_per_server": n_fallback}
    if gpu_sim:
        pool_cfg["placement"] = env_facts.get("placement", "co_locate")
    calibrated = {"calibrated": {"pools": {pool_name: pool_cfg}}}
    service_ms = next((r.get("service_mean_ms") or 0.0 for r in rows if r.get("N") == n_fallback), 0.0)
    validation = validate_parallel_config(calibrated, env_facts, float(service_ms)) if n_fallback else {"result": "SKIP"}
    key = calibration_key(
        str((config.get("model") or {}).get("name") or "model"),
        str((config.get("benchmark") or {}).get("suite") or "benchmark"),
        "-".join(str(g) for g in gpu_list) if gpu_sim else str(gpu_list[0]),
    )
    if n_fallback:
        save_calibration(str(report_dir / "cache.json"), key, calibrated, report)

    return {
        "returncodes": returncodes,
        "mode": "gpu_sim" if gpu_sim else "cpu_sim",
        "sweep": rows,
        "fallback_n": n_fallback,
        "validation": validation,
        "cache_key": key,
        "report_dir": str(report_dir),
    }


def main() -> None:
    """Run main."""
    parser = argparse.ArgumentParser(
        description="Sweep N, build the calibration report, cache a fallback N (CPU-sim and GPU-sim)."
    )
    parser.add_argument("--config", required=True, help="eval config (model/benchmark/server)")
    parser.add_argument("--n", required=True, help="comma-separated N values, e.g. 1,2,4,8,16,32")
    parser.add_argument(
        "--gpus",
        default="",
        help="comma-separated GPU indices. CPU-sim: one GPU per N; GPU-sim: the full server GPU set.",
    )
    parser.add_argument("--out", default="", help="output directory for the sweep")
    args = parser.parse_args()

    from loongforge.embodied.eval.orchestrator.config import load_config

    config_path = str(Path(args.config).resolve())
    config = load_config(config_path)
    ns = [int(x) for x in args.n.split(",")]
    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    print(json.dumps(run(config, config_path, ns, gpus, args.out), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
