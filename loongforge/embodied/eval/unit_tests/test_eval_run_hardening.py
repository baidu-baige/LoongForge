# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Run-directory, resume and manifest guards for evaluation runs."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from pathlib import Path

import pytest

from loongforge.embodied.eval.metrics.repro_metadata import (
    benchmark_commit,
    checkpoint_sha256,
    ensure_manifest_matches_checkpoint,
    git_dirty,
    load_run_manifest,
    write_run_manifest,
)
from loongforge.embodied.eval.orchestrator.config import CONFIG_MAPPING
from loongforge.embodied.eval.orchestrator import server_manager
from loongforge.embodied.eval.orchestrator.run import _timestamped_run_config, _write_effective_config
from loongforge.embodied.eval.orchestrator.runners._common import guard_output_dir


def test_resume_reuses_the_existing_output_dir(tmp_path: Path) -> None:
    """resume must not create an empty timestamped sibling."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    config = {"run": {"resume": True, "output_dir": str(output_dir)}}

    assert _timestamped_run_config(config) is config


def test_resume_requires_an_existing_output_dir(tmp_path: Path) -> None:
    """A resume run cannot start from a directory that does not exist."""
    config = {"run": {"resume": True, "output_dir": str(tmp_path / "missing")}}

    with pytest.raises(SystemExit):
        _timestamped_run_config(config)


def test_guard_refuses_to_append_without_resume_or_restart(tmp_path: Path) -> None:
    """Existing results must not be silently extended."""
    (tmp_path / "results.jsonl").write_text("", encoding="utf-8")
    args = argparse.Namespace(output_dir=str(tmp_path), resume=False, restart=False)

    with pytest.raises(SystemExit):
        guard_output_dir(args)

    args.resume = True
    assert guard_output_dir(args) == tmp_path


def test_config_mapping_exposes_resume_and_report_switches() -> None:
    """F4/F10: the YAML switches must reach the runner args."""
    assert CONFIG_MAPPING["run.resume"] == "resume"
    assert CONFIG_MAPPING["run.timestamped_output"] == "timestamped_output"
    assert CONFIG_MAPPING["run.generate_report"] == "generate_report"


def test_manifest_roundtrip_and_checkpoint_guard(tmp_path: Path) -> None:
    """The manifest records the checkpoint and rejects a mismatched resume."""
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"weights-v1")
    args = argparse.Namespace(ckpt_path=str(checkpoint))

    manifest_path = write_run_manifest(
        tmp_path,
        config={"run": {"seed": 7}},
        args=args,
        protocol={"seed": 7, "noise_seed": 42},
        protocol_version="1.0",
        benchmark_modules=("definitely-not-a-package",),
    )

    assert manifest_path.exists()
    manifest = load_run_manifest(tmp_path)
    assert manifest is not None
    assert manifest["checkpoint"] == str(checkpoint)
    assert manifest["checkpoint_sha256"]
    assert manifest["protocol"]["noise_seed"] == 42
    assert (tmp_path / "resolved_config.yaml").exists()

    ensure_manifest_matches_checkpoint(tmp_path, str(checkpoint))

    other = tmp_path / "other.pt"
    other.write_bytes(b"weights-v2")
    with pytest.raises(SystemExit):
        ensure_manifest_matches_checkpoint(tmp_path, str(other))

    checkpoint.write_bytes(b"weights-v1-mutated")
    with pytest.raises(SystemExit):
        ensure_manifest_matches_checkpoint(tmp_path, str(checkpoint))


def test_directory_checkpoint_sha_is_content_sensitive(tmp_path: Path) -> None:
    """Directory checkpoints (HF-style) must be hashed, not silently skipped."""
    checkpoint = tmp_path / "model-v1"
    checkpoint.mkdir()
    weights = checkpoint / "weights.safetensors"
    weights.write_bytes(b"weights-v1")

    first = checkpoint_sha256(str(checkpoint))
    assert first is not None

    # Same tree at a different path has the same digest.
    duplicate = tmp_path / "model-copy"
    duplicate.mkdir()
    (duplicate / "weights.safetensors").write_bytes(b"weights-v1")
    assert checkpoint_sha256(str(duplicate)) == first

    weights.write_bytes(b"weights-v2")
    assert checkpoint_sha256(str(checkpoint)) != first


def test_manifest_resolved_config_reflects_effective_args(tmp_path: Path) -> None:
    """The manifest must record effective values, not only the raw YAML."""
    args = argparse.Namespace(
        ckpt_path="",
        max_steps=800,
        episodes_per_task=10,
        resume=True,
        restart=False,
        timestamped_output=False,
        generate_report=True,
    )
    config = {
        "benchmark": {"max_steps": 300, "suite": "libero_object"},
        "model": {"model_type": "xvla"},
    }

    write_run_manifest(
        tmp_path,
        config=config,
        args=args,
        protocol={"seed": 7},
        protocol_version="1.0",
    )

    resolved = load_run_manifest(tmp_path)["resolved_config"]
    assert resolved["benchmark"]["max_steps"] == 800  # effective value wins
    assert resolved["benchmark"]["suite"] == "libero_object"  # unmapped key kept
    assert resolved["model"]["model_type"] == "xvla"
    assert resolved["run"]["resume"] is True
    assert resolved["run"]["generate_report"] is True


def _init_git_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-q", "-m", "init"],
        cwd=path,
        check=True,
    )
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, text=True, stdout=subprocess.PIPE
    )
    return proc.stdout.strip()


def test_benchmark_commit_supports_namespace_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Namespace packages (e.g. libero) have origin=None but a source directory."""
    repo = tmp_path / "repo"
    package = repo / "lf_eval_ns_probe"
    package.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    expected = _init_git_repo(repo)

    monkeypatch.syspath_prepend(str(repo))
    importlib.invalidate_caches()
    assert benchmark_commit("lf_eval_ns_probe") == expected


def test_git_dirty_detects_modified_files(tmp_path: Path) -> None:
    """A dirty source tree must be recorded as dirty, not unknown."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "file.txt").write_text("clean\n", encoding="utf-8")
    _init_git_repo(repo)

    git_dirty.cache_clear()
    try:
        assert git_dirty(str(repo)) is False
        (repo / "file.txt").write_text("modified\n", encoding="utf-8")
        git_dirty.cache_clear()
        assert git_dirty(str(repo)) is True
    finally:
        git_dirty.cache_clear()


def _server_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        host="127.0.0.1",
        health_port=1,
        server_start_timeout_sec=1,
        server_log=str(tmp_path / "policy_server.log"),
    )


def test_server_start_retries_a_transient_bind_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep hand-off can hit EADDRINUSE; one retry must recover."""
    (tmp_path / "policy_server.log").write_text(
        "OSError: [Errno 98] address already in use\n", encoding="utf-8"
    )
    calls = {"start": 0, "wait": 0}

    class _Proc:
        pid = 123
        returncode = 1

    def _start(_args: object) -> _Proc:
        calls["start"] += 1
        return _Proc()

    def _wait(*_args: object) -> None:
        calls["wait"] += 1
        if calls["wait"] == 1:
            raise RuntimeError("Policy server exited before healthy")

    monkeypatch.setattr(server_manager, "start_server", _start)
    monkeypatch.setattr(server_manager, "wait_healthz", _wait)
    monkeypatch.setattr(server_manager.ManagedServer, "stop", lambda self: None)
    monkeypatch.setattr(server_manager.time, "sleep", lambda _seconds: None)

    server_manager.ManagedServer(_server_args(tmp_path)).start()

    assert calls == {"start": 2, "wait": 2}


def test_server_start_does_not_retry_real_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only bind failures are retried; model-load crashes propagate immediately."""
    (tmp_path / "policy_server.log").write_text(
        "torch.OutOfMemoryError: CUDA out of memory\n", encoding="utf-8"
    )
    calls = {"start": 0}

    class _Proc:
        pid = 123
        returncode = 1

    def _start(_args: object) -> _Proc:
        calls["start"] += 1
        return _Proc()

    monkeypatch.setattr(server_manager, "start_server", _start)
    monkeypatch.setattr(
        server_manager,
        "wait_healthz",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("Policy server exited before healthy")),
    )
    monkeypatch.setattr(server_manager.ManagedServer, "stop", lambda self: None)

    with pytest.raises(RuntimeError):
        server_manager.ManagedServer(_server_args(tmp_path)).start()

    assert calls["start"] == 1


def test_effective_config_persists_sweep_overrides(tmp_path: Path) -> None:
    """The server must read the expanded config, not the original YAML."""
    output_dir = tmp_path / "run"
    config = {
        "run": {"output_dir": str(output_dir)},
        "server": {"port": 17200, "health_port": 17201},
        "benchmark": {"suite": "libero_object"},
    }

    path = Path(_write_effective_config(config, fallback_path="ignored.yaml"))

    assert path == output_dir / "effective_config.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["server"]["port"] == 17200
    assert written["benchmark"]["suite"] == "libero_object"
