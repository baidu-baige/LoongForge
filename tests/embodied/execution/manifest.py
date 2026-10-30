# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Script manifest loading: config/scripts.yaml -> {name: script path relative to examples/embodied}."""

import os

import yaml

# Path of the manifest file relative to the embodied suite root (tests/embodied).
MANIFEST_FILE = os.path.join("config", "scripts.yaml")


def examples_dir(embodied_root):
    """tests/embodied -> tests -> repo root -> examples/embodied."""
    repo_root = os.path.dirname(os.path.dirname(embodied_root))
    return os.path.join(repo_root, "examples", "embodied")


def load_manifest(embodied_root):
    """Return {model_name: script_relpath} in file order."""
    path = os.path.join(embodied_root, MANIFEST_FILE)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} should be a YAML mapping of '<model name>: <training script relative path>'")
    entries = {}
    for model_name, script in data.items():
        if not isinstance(script, str) or not script.strip():
            raise ValueError(f"The script path for '{model_name}' in {path} should be a non-empty string: {script!r}")
        entries[str(model_name)] = script.strip()
    return entries


def list_models(embodied_root):
    return list(load_manifest(embodied_root))

if __name__ == "__main__":
    embodied_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = load_manifest(embodied_root)
    for model_name, script in manifest.items():
        print(f"{model_name}: {script}")
