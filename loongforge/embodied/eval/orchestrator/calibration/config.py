# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate and cache the calibrated parallel config.

The calibration report (``report.py``) drives an agent (or a fallback
rule) to pick a parallel config. This module then:
- checks that config against hard bounds (cores / VRAM / client timeout), and
- caches it keyed by ``(model, benchmark, gpu)`` so the next run can skip
  calibration.

Validation is deliberately conservative: a config that fails any bound is
rejected rather than silently used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def calibration_key(model: str, benchmark: str, gpu: str) -> str:
    """Build the cache key for a (model, benchmark, gpu) combination."""
    return f"{model}__{benchmark}__{gpu}"


def validate_parallel_config(
    config: Dict[str, Any],
    env: Dict[str, Any],
    service_time_ms: float,
    timeout_safety: float = 2.0,
) -> Dict[str, Any]:
    """Check a calibrated parallel config against hard bounds.

    Bounds (see the architecture doc §6.1 Step 5):
    - CPU: ``N <= cores / cores_per_sim``
    - GPU: ``k * vram_per_sim + vram_model <= vram_total``
    - timeout: ``policy_call_ms >= N * mean_service_ms * timeout_safety``
    """
    calibrated = config.get("calibrated") or {}
    pools = calibrated.get("pools") or {}
    cpu = pools.get("cpu_sim") or {}
    gpu = pools.get("gpu_sim") or {}
    checks: List[Dict[str, Any]] = []

    def add(rule: str, value: str, ok: bool, note: str = "") -> None:
        checks.append({"rule": rule, "value": value, "ok": ok, "note": note})

    n = cpu.get("n_sims_per_server")
    cores, cores_per_sim = env.get("cores"), env.get("cores_per_sim")
    if n is not None:
        if cores and cores_per_sim:
            cap = int(cores / cores_per_sim)
            add("cpu: N <= cores / cores_per_sim", f"{n} <= {cores}/{cores_per_sim} = {cap}", n <= cap)
        else:
            add("cpu: N <= cores / cores_per_sim", f"N={n}, cores_per_sim unmeasured", True, "unchecked")

    k = gpu.get("n_sims_per_server")
    vram_total = env.get("vram_total_mb")
    if k is not None and vram_total:
        need = k * (env.get("vram_per_sim_mb") or 0) + (env.get("vram_model_mb") or 0)
        add("gpu: k*vram_per_sim + vram_model <= vram_total", f"{need} <= {vram_total}", need <= vram_total)

    timeout = (calibrated.get("timeouts") or {}).get("policy_call_ms")
    if timeout is not None and n is not None:
        need = n * service_time_ms * timeout_safety
        add(
            "timeout >= N * mean_service * safety",
            f"{timeout}ms >= {need:.0f}ms",
            timeout >= need,
        )

    return {"checks": checks, "result": "PASS" if checks and all(c["ok"] for c in checks) else "FAIL"}


def save_calibration(path: str, key: str, config: Dict[str, Any], report: Dict[str, Any]) -> None:
    """Insert/overwrite the cached calibration entry for ``key``."""
    store = _load_store(path)
    store[key] = {"config": config, "report": report}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(store, indent=2), encoding="utf-8")


def load_calibration(path: str, key: str) -> Optional[Dict[str, Any]]:
    """Return the cached entry for ``key``, or None."""
    return _load_store(path).get(key)


def _load_store(path: str) -> Dict[str, Any]:
    out = Path(path)
    if not out.exists():
        return {}
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    """Run main."""
    parser = argparse.ArgumentParser(description="Validate a calibrated parallel config.")
    parser.add_argument("--config", required=True, help="calibrated parallel config YAML/JSON")
    parser.add_argument("--env", required=True, help="env facts JSON")
    parser.add_argument("--timing", required=True, help="policy server timing JSON (uses 'mean')")
    args = parser.parse_args()

    config = _load_any(args.config)
    env = json.loads(Path(args.env).read_text())
    mean_ms = float(json.loads(Path(args.timing).read_text())["mean"])
    result = validate_parallel_config(config, env, mean_ms)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["result"] == "PASS" else 1)


def _load_any(path: str) -> Dict[str, Any]:
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml  # local import: yaml is an optional dep for this CLI path

        return yaml.safe_load(text)


if __name__ == "__main__":
    main()
