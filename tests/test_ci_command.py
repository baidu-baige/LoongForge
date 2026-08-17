# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / ".github" / "scripts" / "ci_command.py"
SPEC = importlib.util.spec_from_file_location("ci_command", MODULE_PATH)
ci_command = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules["ci_command"] = ci_command
SPEC.loader.exec_module(ci_command)


def test_all_environments_expand_to_configured_runners():
    request = ci_command.parse_command(
        "/ok-to-test --env all --model llama3_8b,qwen3_14b --build-image p"
    )

    assert request.environments == ["a", "p"]
    assert request.models == ["llama3_8b", "qwen3_14b"]
    assert request.build_image == "p"


def test_models_are_optional_and_default_to_baselines():
    request = ci_command.parse_command("/ok-to-test --env a")

    assert request.models == []


@pytest.mark.parametrize(
    "comment",
    [
        "/ok-to-test --env b",
        "/ok-to-test --env a --unknown value",
        "/ok-to-test --env a --model 'x; uname'",
        "please /ok-to-test --env a",
    ],
)
def test_invalid_or_unsafe_commands_are_rejected(comment: str):
    with pytest.raises(ci_command.CommandError):
        ci_command.parse_command(comment)
