#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate Release Request model selections against LoongForge tests."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RUNNABLE_RE = re.compile(r"(?m)^MODEL_RUNNABLE:\s*[Tt]rue(?:\s|#|$)")
EMBODIED_MODEL_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*):\s+\S+")


def validate_llm_model(root: Path, model: str) -> str | None:
    if not MODEL_RE.fullmatch(model):
        return "llm_model is invalid"
    config = root / "tests" / "llm_vlm" / "configs" / f"{model}.yaml"
    if not config.is_file():
        return "llm_model has no LoongForge definition"
    if not RUNNABLE_RE.search(config.read_text(encoding="utf-8")):
        return "llm_model is not runnable"
    return None


def validate_vla_model(root: Path, model: str) -> str | None:
    if not MODEL_RE.fullmatch(model):
        return "vla_model is invalid"
    manifest = root / "tests" / "embodied" / "config" / "scripts.yaml"
    if not manifest.is_file():
        return "vla_model manifest is unavailable"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = EMBODIED_MODEL_RE.match(line)
        if match and match.group(1) == model:
            return None
    return "vla_model has no LoongForge definition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--vla-model", required=True)
    parser.add_argument("--source", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.source).resolve()
    errors = [
        validate_llm_model(root, args.llm_model),
        validate_vla_model(root, args.vla_model),
    ]
    errors = [error for error in errors if error is not None]
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
