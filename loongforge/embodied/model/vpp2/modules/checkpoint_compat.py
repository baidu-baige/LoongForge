# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Read historical VPP2 action checkpoints without importing BASE packages."""

ACTION_FORMAT = "vpp2_wan21_action_v1"
_LEGACY_ACTION_FORMATS = {"fastwam_wan21_action_v1", "wan21_action_v1"}
_LEGACY_NAMESPACES = ("fastwam.", "robodojo_joint2b.")
_LEGACY_TARGET_PARTS = {
    "fastwam": "vpp2",
    "FastWAM": "VPP2",
    "fastwam_processor": "vpp2_processor",
    "FastWAMProcessor": "VPP2Processor",
    "create_fastwam_wan21_14b": "create_vpp2_wan21_14b",
}


def canonicalize_config(value):
    """Convert legacy Hydra targets recursively, preserving other values."""
    if isinstance(value, list):
        return [canonicalize_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: canonicalize_config(item) for key, item in value.items()}
    target = result.get("_target_")
    if isinstance(target, str):
        for prefix in _LEGACY_NAMESPACES:
            if target.startswith(prefix):
                target = "vpp2." + target[len(prefix) :]
                break
        if target.startswith("vpp2."):
            target = ".".join(_LEGACY_TARGET_PARTS.get(part, part) for part in target.split("."))
        result["_target_"] = target
    return result


def action_config(payload):
    """Validate and canonicalize a serialized action-checkpoint envelope."""
    if payload.get("format") not in {ACTION_FORMAT, *_LEGACY_ACTION_FORMATS}:
        raise ValueError("Expected an exported VPP2 action checkpoint")
    return canonicalize_config(payload["resolved_config"])
