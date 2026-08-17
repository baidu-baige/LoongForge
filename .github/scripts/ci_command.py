# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Parse the public ``/ok-to-test`` command into a dispatch payload."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import asdict, dataclass


MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


@dataclass(frozen=True)
class TestRequest:
    environments: list[str]
    models: list[str]
    build_image: str


class CommandError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CommandError(message)


def parse_command(comment: str) -> TestRequest:
    line = next((line.strip() for line in comment.splitlines() if line.strip()), "")
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        raise CommandError(str(exc)) from exc
    if not tokens or tokens[0] != "/ok-to-test":
        raise CommandError("comment must start with /ok-to-test")

    parser = _Parser(add_help=False, allow_abbrev=False)
    parser.add_argument("--env", required=True, choices=("a", "p", "all"))
    parser.add_argument("--model", default="")
    parser.add_argument("--build-image", default="", choices=("", "a", "h", "p"))
    args = parser.parse_args(tokens[1:])

    models = [item.strip() for item in args.model.split(",") if item.strip()]
    invalid = [model for model in models if not MODEL_RE.fullmatch(model)]
    if invalid:
        raise CommandError(f"invalid model name: {invalid[0]}")
    environments = ["a", "p"] if args.env == "all" else [args.env]
    return TestRequest(environments=environments, models=models, build_image=args.build_image)


def main() -> int:
    try:
        request = parse_command(sys.stdin.read())
    except CommandError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(asdict(request), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
