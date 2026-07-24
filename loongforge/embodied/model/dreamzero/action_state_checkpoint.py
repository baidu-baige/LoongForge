# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Locate DreamZero action/state tensors inside checkpoint artifacts.

Base Wan DiT release checkpoints do not contain DreamZero's state/action
encoder and action decoder tensors. DreamZero training checkpoints store those
tensors under ``action_head.model.*``.
"""

from __future__ import annotations

import json
from pathlib import Path

ACTION_STATE_TARGETS = (
    "state_encoder.layer1.W",
    "state_encoder.layer1.b",
    "state_encoder.layer2.W",
    "state_encoder.layer2.b",
    "action_encoder.W1.W",
    "action_encoder.W1.b",
    "action_encoder.W2.W",
    "action_encoder.W2.b",
    "action_encoder.W3.W",
    "action_encoder.W3.b",
    "action_decoder.layer1.W",
    "action_decoder.layer1.b",
    "action_decoder.layer2.W",
    "action_decoder.layer2.b",
)
ACTION_STATE_SOURCE_KEYS = tuple(f"action_head.model.{name}" for name in ACTION_STATE_TARGETS)

_INDEX_FILENAMES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
)


def candidate_action_state_files(path: Path | str) -> dict[str, list[Path]]:
    """Return safetensors files that may contain each required source key."""
    root = Path(path)
    if root.is_file():
        return {key: [root] for key in ACTION_STATE_SOURCE_KEYS}
    if not root.is_dir():
        raise FileNotFoundError(root)

    for index_name in _INDEX_FILENAMES:
        index_path = root / index_name
        if not index_path.exists():
            continue
        with index_path.open("r") as f:
            weight_map = json.load(f).get("weight_map", {})
        return {
            key: [root / weight_map[key]] if key in weight_map else []
            for key in ACTION_STATE_SOURCE_KEYS
        }

    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files under {root}")
    return {key: files for key in ACTION_STATE_SOURCE_KEYS}
