# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Artifact retention policy helpers."""

from typing import Dict


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _artifact_flag(cfg: Dict, name: str, default: bool) -> bool:
    if name in cfg:
        return _as_bool(cfg[name])
    return _as_bool(cfg.get("artifacts", {}).get(name, default))


def debug_artifacts_enabled(cfg: Dict) -> bool:
    """Whether to keep human-readable diagnostic artifacts."""
    return _artifact_flag(cfg, "debug_artifacts", True)


def keep_intermediate_artifacts(cfg: Dict) -> bool:
    """Whether to keep handoff files that are not needed after the next stage."""
    return _artifact_flag(cfg, "keep_intermediate", True)
