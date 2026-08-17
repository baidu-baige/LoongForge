# Copyright 2026 The LoongForge Authors.
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
    existing_tasks = record.get("tasks") or []
    existing_training_type = record.get("training_type") or []
    record.update({
        "status": "completed",
        "passed": bool(passed),
        "updated_at": _now(),
    })
    if meta:
        meta = dict(meta)
        if "tasks" in meta:
            merged_tasks = set(existing_tasks)
            merged_tasks.update(meta.get("tasks") or [])
            meta["tasks"] = sorted(merged_tasks)
        if "training_type" in meta:
            merged_training_type = set(existing_training_type)
            merged_training_type.update(meta.get("training_type") or [])
            meta["training_type"] = sorted(merged_training_type)
        if "tasks_passed" in meta:
            merged_tasks_passed = dict(record.get("tasks_passed") or {})
            merged_tasks_passed.update(meta.get("tasks_passed") or {})
            meta["tasks_passed"] = merged_tasks_passed
        record.update(meta)
    models[model_name] = record


def get_completed_models(
    state: Dict[str, Any],
    policy: str = "skip_completed",
    required_tasks: Any = None,
    required_training_type: Any = None,
) -> Set[str]:
    models = state.get("models", {}) or {}
    completed = set()
    required_tasks_set = set(required_tasks) if required_tasks else None
    required_training_type_set = set(required_training_type) if required_training_type else None

    for name, info in models.items():
        if not isinstance(info, dict):
            continue
        status = info.get("status")
        passed = info.get("passed")
        tasks_passed = info.get("tasks_passed") or {}
        if policy == "skip_passed":
            if required_tasks_set is not None and tasks_passed:
                if not all(tasks_passed.get(task) is True for task in required_tasks_set):
                    continue
            else:
                if not (status == "completed" and passed is True):
                    continue
        else:
            if status != "completed":
                continue
            if required_tasks_set is not None and tasks_passed:
                if not set(tasks_passed.keys()).issuperset(required_tasks_set):
                    continue

        if required_tasks_set is not None and "tasks" in info:
            recorded_tasks = info.get("tasks") or []
            if not set(recorded_tasks).issuperset(required_tasks_set):
                continue

        if required_training_type_set is not None and "training_type" in info:
            recorded_training_type = info.get("training_type") or []
            if not set(recorded_training_type).issuperset(required_training_type_set):
                continue

        completed.add(name)

    return completed
