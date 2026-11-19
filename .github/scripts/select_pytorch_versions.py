# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Select PyTorch matrix versions from pip's compatible wheel candidates."""

import json
import os
import re
import sys


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


if __name__ == "__main__":
    try:
        versions = select_versions(json.load(sys.stdin)["versions"], os.environ.get("TORCH_VERSIONS", ""))
    except ValueError as error:
        sys.exit(str(error))

    print(f"Selected PyTorch versions: {', '.join(versions)}")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        print(f"matrix={json.dumps({'torch-version': versions}, separators=(',', ':'))}", file=output)
