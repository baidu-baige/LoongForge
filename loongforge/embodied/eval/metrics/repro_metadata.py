# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Reproducibility metadata and per-run manifests for evaluation.

The manifest captures what a run actually used (resolved configuration,
checkpoint identity, software revision) so that a results directory can be
audited without the original YAML or shell history.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_git(repo: Path, args: Iterable[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


@lru_cache(maxsize=4)
def git_sha(repo: Optional[str] = None) -> Optional[str]:
    """Return the current git revision of the source tree."""
    return _run_git(Path(repo) if repo else _REPO_ROOT, ["rev-parse", "HEAD"])


@lru_cache(maxsize=4)
def git_dirty(repo: Optional[str] = None) -> Optional[bool]:
    """Return whether tracked files have uncommitted changes.

    Untracked files are excluded: scanning them on a network filesystem can
    exceed the git timeout and turn a boolean into ``None``.
    """
    status = _run_git(
        Path(repo) if repo else _REPO_ROOT,
        ["status", "--porcelain", "--untracked-files=no"],
    )
    return None if status is None else bool(status)


@lru_cache(maxsize=8)
def benchmark_commit(*module_names: str) -> Optional[str]:
    """Return the git revision of an installed benchmark package, if any.

    Tries each module name in order; returns ``None`` when the package is not
    installed or is not checked out from a git repository.
    """
    for name in module_names:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        directories = []
        if spec.origin and spec.origin not in {"built-in", "frozen"}:
            directories.append(Path(spec.origin).resolve().parent)
        # Namespace packages (e.g. ``libero``) have ``origin=None`` but still
        # expose their source directories through ``submodule_search_locations``.
        for location in spec.submodule_search_locations or []:
            directories.append(Path(location).resolve())
        for directory in directories:
            for candidate in (directory, directory.parent, directory.parent.parent):
                sha = _run_git(candidate, ["rev-parse", "HEAD"])
                if sha:
                    return sha
    return None


def docker_image() -> Optional[str]:
    """Return the container image name when it is exposed by the environment."""
    for key in ("LOONGFORGE_DOCKER_IMAGE", "DOCKER_IMAGE", "IMAGE_NAME"):
        value = os.environ.get(key)
        if value:
            return value
    return None


@lru_cache(maxsize=1)
def python_env_hash() -> str:
    """Return a short stable hash of the Python environment."""
    parts = [sys.version, platform.platform()]
    for package in ("torch", "numpy", "robosuite", "libero", "calvin-env"):
        try:
            parts.append(f"{package}={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _directory_sha256(directory: Path) -> str:
    """Return a deterministic content hash for a checkpoint directory.

    Files are visited in sorted relative-path order; each entry contributes its
    relative path, size, and content. This makes the digest independent of
    directory traversal order and of file timestamps.
    """
    digest = hashlib.sha256()
    files = sorted(
        (path for path in directory.rglob("*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    for path in files:
        relative = path.relative_to(directory).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_sha256(path: str) -> Optional[str]:
    """Return the SHA-256 of a checkpoint file or directory.

    Directories are hashed over their sorted file tree (path, size, content),
    which covers the common case of HuggingFace-style checkpoint directories.
    Returns ``None`` when the path is missing or empty.
    """
    if not path:
        return None
    checkpoint = Path(path)
    if checkpoint.is_file():
        digest = hashlib.sha256()
        with checkpoint.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if checkpoint.is_dir():
        return _directory_sha256(checkpoint)
    return None


def repro_fields(benchmark_modules: Iterable[str] = ()) -> Dict[str, Any]:
    """Return the static reproducibility fields embedded in every record."""
    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "benchmark_commit": benchmark_commit(*tuple(benchmark_modules)),
        "docker_image": docker_image(),
        "python_env_hash": python_env_hash(),
    }


def _set_nested(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``target[a][b] = value`` for ``dotted_key='a.b'``."""
    parts = dotted_key.split(".")
    node = target
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _resolved_config(config: Optional[Dict[str, Any]], args: Any) -> Dict[str, Any]:
    """Return the raw YAML overlaid with the runner's effective values.

    Values are overlaid through ``CONFIG_MAPPING`` so the manifest reflects
    defaults and CLI/config resolution rather than only what the YAML spelled
    out. Unmapped YAML entries (e.g. ``model.*``) are preserved.
    """
    resolved = copy.deepcopy(config) if config else {}
    try:
        from loongforge.embodied.eval.orchestrator.config import CONFIG_MAPPING
    except Exception:  # Keep the manifest usable in isolation (e.g. tests).
        CONFIG_MAPPING = {}
    for source, target in CONFIG_MAPPING.items():
        if hasattr(args, target):
            _set_nested(resolved, source, getattr(args, target))
    if not config:
        resolved["runner_args"] = {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, list, dict, type(None)))
        }
    return resolved


def write_run_manifest(
    output_dir: Any,
    *,
    config: Optional[Dict[str, Any]],
    args: Any,
    protocol: Optional[Dict[str, Any]],
    protocol_version: str,
    benchmark_modules: Iterable[str] = (),
) -> Path:
    """Write ``run_manifest.json`` (and the resolved YAML) into ``output_dir``."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = str(getattr(args, "ckpt_path", "") or "")
    resolved_config = _resolved_config(config, args)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": protocol_version,
        "resolved_config": resolved_config,
        "protocol": protocol or {},
        **repro_fields(benchmark_modules),
        "checkpoint": checkpoint or None,
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if resolved_config:
        try:
            import yaml

            (output / "resolved_config.yaml").write_text(
                yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
            )
        except Exception:  # PyYAML is optional for JSON configs.
            pass
    return manifest_path


def load_run_manifest(output_dir: Any) -> Optional[Dict[str, Any]]:
    """Load ``run_manifest.json`` from a results directory, if present."""
    manifest_path = Path(output_dir) / "run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def ensure_manifest_matches_checkpoint(output_dir: Any, checkpoint: str) -> None:
    """Abort a resumed run whose results directory belongs to another checkpoint."""
    manifest = load_run_manifest(output_dir)
    if not manifest or not checkpoint:
        return
    recorded_path = manifest.get("checkpoint")
    if recorded_path and Path(str(recorded_path)).expanduser().resolve() != Path(checkpoint).expanduser().resolve():
        raise SystemExit(
            "output_dir belongs to a different checkpoint: "
            f"{recorded_path!r} != {checkpoint!r}; use a new directory or pass run.restart=true"
        )
    recorded_sha = manifest.get("checkpoint_sha256")
    if recorded_sha:
        current_sha = checkpoint_sha256(str(Path(checkpoint).expanduser()))
        if current_sha and current_sha != recorded_sha:
            raise SystemExit(
                "output_dir checkpoint content changed (sha256 mismatch); "
                "use a new directory or pass run.restart=true"
            )
