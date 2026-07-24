# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Configuration loading for DreamZero cache precomputation."""

from __future__ import annotations

import os
import re
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from loongforge.embodied.data.datasets.dreamzero.transforms.data_configuration_dreamzero import (
    DreamZeroDataConfig,
)
from loongforge.embodied.model.dreamzero.model_configuration_dreamzero import DreamZeroConfig

_MODEL_CONFIG_FIELDS = {field.name for field in fields(DreamZeroConfig)}

_DATA_CONFIG_FIELDS = {field.name for field in fields(DreamZeroDataConfig)}

_PRECOMPUTE_CONFIG_DEFAULTS = {
    "tokenizer_path": "",
    "video_backend": "decord",
}

_OC_ENV_PATTERN = re.compile(r"\$\{oc\.env:([^}]+)\}")


def _expand_oc_env(value: Any) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise KeyError(f"environment variable {name!r} is required by config")
            return os.environ[name]

        return _OC_ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_expand_oc_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_oc_env(item) for key, item in value.items()}
    return value


def _load_config(path: Path) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_oc_env(raw)
    if "model" in raw or "data" in raw:
        model_values = dict(raw.get("model", {}) or {})
        data_values = dict(raw.get("data", {}) or {})
    else:
        model_values = {
            key: value for key, value in raw.items() if key in _MODEL_CONFIG_FIELDS
        }
        data_values = {
            key: value for key, value in raw.items() if key in _DATA_CONFIG_FIELDS
        }

    model_values.pop("_target_", None)
    model_values.pop("model_type", None)
    model_values.setdefault("device", "cpu")
    data_values.pop("_target_", None)
    data_values.setdefault("metadata_version", None)

    unknown_model = set(model_values) - _MODEL_CONFIG_FIELDS
    unknown_data = set(data_values) - _DATA_CONFIG_FIELDS
    if unknown_model or unknown_data:
        raise ValueError(
            f"unsupported DreamZero precompute config keys in {path}: "
            f"model={sorted(unknown_model)} data={sorted(unknown_data)}"
        )

    model_config = DreamZeroConfig(**model_values)
    data_config = DreamZeroDataConfig(**data_values)
    config = {
        **vars(model_config),
        **vars(data_config),
        **_PRECOMPUTE_CONFIG_DEFAULTS,
    }
    return SimpleNamespace(**config)
