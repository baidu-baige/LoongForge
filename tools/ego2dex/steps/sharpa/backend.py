# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Resolve only the pinned official hand assets; no alternative E2R pipeline."""

import os
from pathlib import Path


def upstream_root():
    path = Path(os.environ.get("EGO2ROBOT_SHARPA_UPSTREAM", "/official_sharpa")).resolve()
    if not (path / "deployment/mujoco_replay/assets/sharpa_assets").is_dir():
        raise FileNotFoundError(f"Pinned official Sharpa assets are unavailable: {path}")
    return path
