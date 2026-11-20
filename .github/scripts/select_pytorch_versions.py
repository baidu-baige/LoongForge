# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Select PyTorch matrix versions from pip's compatible wheel candidates."""

import json
import os
import re
import sys
from urllib.request import urlopen

from packaging.specifiers import SpecifierSet


def select_versions(candidates: list[str], override: str = "") -> list[str]:
    stable = {
        version.split("+", 1)[0]
        for version in candidates
        if re.fullmatch(r"\d+\.\d+\.\d+(?:\+cpu)?", version)
    }
    if override.strip():
        selected = list(dict.fromkeys(version.strip() for version in override.split(",")))
        if invalid := [version for version in selected if version not in stable]:
            raise ValueError(f"No compatible stable CPU wheel for {invalid!r}")
        return selected

    latest = {}
    for version in sorted(stable, key=lambda value: tuple(map(int, value.split("."))), reverse=True):
        series = version.rsplit(".", 1)[0]
        latest.setdefault(series, version)
        if len(latest) == 2:
            return list(latest.values())
    raise ValueError("Expected two stable PyTorch series with compatible CPU wheels")


def build_python_torch_matrix(
    versions: list[str],
    python_versions: list[str],
    requires_python: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    """Pair each PyTorch release with Python versions accepted by its metadata."""
    matrix = []
    for torch_version in versions:
        python_spec = SpecifierSet(requires_python[torch_version])
        for python_version in python_versions:
            if python_spec.contains(python_version, prereleases=True):
                matrix.append(
                    {
                        "python-version": python_version,
                        "torch-version": torch_version,
                    }
                )

    if not matrix:
        raise ValueError("No stable PyTorch release supports the configured Python versions")
    return {"include": matrix}


def fetch_requires_python(torch_version: str) -> str:
    """Read Requires-Python from the official PyPI metadata for a release."""
    url = f"https://pypi.org/pypi/torch/{torch_version}/json"
    with urlopen(url, timeout=30) as response:
        metadata = json.load(response)
    requires_python = metadata["info"].get("requires_python")
    if not requires_python:
        raise ValueError(f"PyPI metadata has no Requires-Python for torch {torch_version}")
    return requires_python


if __name__ == "__main__":
    try:
        versions = select_versions(json.load(sys.stdin)["versions"], os.environ.get("TORCH_VERSIONS", ""))
        python_versions = [
            version.strip()
            for version in os.environ.get("PYTHON_VERSIONS", "3.10,3.11,3.12").split(",")
            if version.strip()
        ]
        requires_python = {version: fetch_requires_python(version) for version in versions}
        matrix = build_python_torch_matrix(versions, python_versions, requires_python)
    except ValueError as error:
        sys.exit(str(error))

    print(f"Selected PyTorch versions: {', '.join(versions)}")
    print(f"Compatibility matrix: {json.dumps(matrix, separators=(',', ':'))}")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        print(f"matrix={json.dumps(matrix, separators=(',', ':'))}", file=output)
