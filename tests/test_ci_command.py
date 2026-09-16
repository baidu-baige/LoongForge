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


def test_torch_suite_selects_image_build():
    request = ci_command.parse_command(
        "/ok-to-test --suite torch --model pi05_ddp --build-image"
    )

    assert request.suite == "torch"
    assert request.models == ["pi05_ddp"]
    assert request.build_image is True


def test_models_are_optional_and_default_to_baselines():
    request = ci_command.parse_command("/ok-to-test --suite mcore")

    assert request.suite == "mcore"
    assert request.models == []
    assert request.build_image is False


@pytest.mark.parametrize(
    "comment",
    [
        "/ok-to-test --suite unknown",
        "/ok-to-test --suite mcore --unknown value",
        "/ok-to-test --suite mcore --model 'x; uname'",
        "/ok-to-test --suite torch --build-image p",
        "please /ok-to-test --suite mcore",
    ],
)
def test_invalid_or_unsafe_commands_are_rejected(comment: str):
    with pytest.raises(ci_command.CommandError):
        ci_command.parse_command(comment)
