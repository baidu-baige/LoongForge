# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for parallel-config validation and the calibration cache."""

from __future__ import annotations

from loongforge.embodied.eval.orchestrator.calibration.config import (
    calibration_key,
    load_calibration,
    save_calibration,
    validate_parallel_config,
)


def _config(n=16, timeout=60000, gpu_k=None):
    pools = {"cpu_sim": {"n_sims_per_server": n}}
    if gpu_k is not None:
        pools["gpu_sim"] = {"n_sims_per_server": gpu_k}
    return {"calibrated": {"pools": pools, "timeouts": {"policy_call_ms": timeout}}}


def test_key_format() -> None:
    """Run test_key_format."""
    assert calibration_key("fastwam", "libero_10", "A800") == "fastwam__libero_10__A800"


def test_valid_config_passes() -> None:
    """Run test_valid_config_passes."""
    env = {"cores": 128, "cores_per_sim": 1.5, "vram_total_mb": 81920, "vram_model_mb": 24700, "vram_per_sim_mb": 0}
    result = validate_parallel_config(_config(n=16), env, service_time_ms=50.9)
    assert result["result"] == "PASS"


def test_cpu_bound_violation_fails() -> None:
    """Run test_cpu_bound_violation_fails."""
    env = {"cores": 8, "cores_per_sim": 1.5}  # cap = 5
    result = validate_parallel_config(_config(n=16), env, service_time_ms=50.9)
    assert result["result"] == "FAIL"
    assert any(c["rule"].startswith("cpu:") and not c["ok"] for c in result["checks"])


def test_vram_bound_violation_fails() -> None:
    """Run test_vram_bound_violation_fails."""
    env = {"vram_total_mb": 81920, "vram_model_mb": 24700, "vram_per_sim_mb": 20000}
    result = validate_parallel_config(_config(gpu_k=8), env, service_time_ms=50.9)
    assert result["result"] == "FAIL"
    assert any(c["rule"].startswith("gpu:") and not c["ok"] for c in result["checks"])


def test_timeout_bound_violation_fails() -> None:
    """Run test_timeout_bound_violation_fails."""
    env = {"cores": 128}
    result = validate_parallel_config(_config(n=100, timeout=5000), env, service_time_ms=50.9)
    assert result["result"] == "FAIL"
    assert any(c["rule"].startswith("timeout") and not c["ok"] for c in result["checks"])


def test_unmeasured_cores_per_sim_is_unchecked() -> None:
    """Run test_unmeasured_cores_per_sim_is_unchecked."""
    env = {"cores": 128}
    result = validate_parallel_config(_config(n=16), env, service_time_ms=50.9)
    assert any("unchecked" in c["note"] for c in result["checks"])


def test_cache_round_trip(tmp_path) -> None:
    """Run test_cache_round_trip."""
    store = str(tmp_path / "cache.json")
    key = calibration_key("fastwam", "libero_10", "A800")
    assert load_calibration(store, key) is None
    save_calibration(store, key, {"calibrated": {"pools": {}}}, {"kind": "m0_calibration"})
    entry = load_calibration(store, key)
    assert entry is not None and entry["report"]["kind"] == "m0_calibration"
