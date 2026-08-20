# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Configuration helpers shared by local and CI regression entry points."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


LOGICAL_ENVS = ("a", "p")


def load_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Read a simple KEY=VALUE file without executing it as shell code."""
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid config line {line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"invalid config key on line {line_number}: {key!r}")
        values[key] = value.strip().strip("\"'")
    return values


def baseline_models(
    env: str,
    baseline_root: str | os.PathLike[str],
    requested: list[str] | None = None,
) -> list[str]:
    """Return models with a baseline for one logical environment.

    The physical directory can be configured with ``LOONGFORGE_BASELINE_<ENV>``;
    its name is intentionally kept out of workflow arguments and Check names.
    """
    if env not in LOGICAL_ENVS:
        raise ValueError(f"unsupported environment: {env}")
    root = Path(baseline_root)
    physical = os.getenv(f"LOONGFORGE_BASELINE_{env.upper()}")
    if not physical:
        physical = {"a": "A800", "p": "BZZ"}[env]
    baseline_dir = root / "default" / physical
    available = {path.stem for path in baseline_dir.glob("*.json")}
    if requested is None:
        return sorted(available)
    return sorted(set(requested) & available)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=LOGICAL_ENVS, required=True)
    parser.add_argument("--baseline-root", default="tests/baseline")
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    selected = baseline_models(args.env, args.baseline_root, args.models or None)
    if args.models:
        missing = sorted(set(args.models) - set(selected))
        if missing:
            print(
                f"not configured for {args.env}: {', '.join(missing)}",
                file=sys.stderr,
            )
    print(" ".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
