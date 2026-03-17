# Copyright 2026 The OmniTraining Authors.
# SPDX-License-Identifier: Apache-2.0

"""resume state"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Set

STATE_VERSION = 1


def _now() -> str:
    # return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    beijing_tz = timezone(timedelta(hours=8))
    return datetime.now(beijing_tz).strftime("%Y-%m-%dT%H:%M:%S%z")


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def load_state(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {
            "version": STATE_VERSION,
            "created_at": _now(),
            "updated_at": _now(),
            "models": {},
        }

    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("resume state is not a dict")
    except Exception:
        return {
            "version": STATE_VERSION,
            "created_at": _now(),
            "updated_at": _now(),
            "models": {},
        }

    data.setdefault("version", STATE_VERSION)
    data.setdefault("created_at", _now())
    data.setdefault("updated_at", _now())
    data.setdefault("models", {})
    if not isinstance(data.get("models"), dict):
        data["models"] = {}
    return data


def save_state(path: str, state: Dict[str, Any]) -> None:
    if not path:
        return
    _ensure_parent(path)
    state["updated_at"] = _now()
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def mark_model(state: Dict[str, Any], model_name: str, passed: bool, meta: Dict[str, Any]) -> None:
    models = state.setdefault("models", {})
    record = models.get(model_name, {})
    record.update({
        "status": "completed",
        "passed": bool(passed),
        "updated_at": _now(),
    })
    if meta:
        record.update(meta)
    models[model_name] = record


def get_completed_models(state: Dict[str, Any], policy: str = "skip_completed") -> Set[str]:
    models = state.get("models", {}) or {}
    completed = set()
    for name, info in models.items():
        if not isinstance(info, dict):
            continue
        status = info.get("status")
        passed = info.get("passed")
        if policy == "skip_passed":
            if status == "completed" and passed is True:
                completed.add(name)
        else:
            if status == "completed":
                completed.add(name)
    return completed
