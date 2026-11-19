# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tests.ci_test_utils import load_module

select_versions = load_module(
    "select_pytorch_versions",
    Path(__file__).resolve().parents[1] / ".github/scripts/select_pytorch_versions.py",
).select_versions


def test_latest_patch_in_each_of_two_newest_stable_series():
    candidates = [
        "2.9.12+cpu", "2.10.0+cpu", "2.10.10+cpu", "2.10.2", "2.10.10",
        "2.11.0rc1+cpu", "2.11.0.dev20261119+cpu", "2.11.0+cu130", "2.8.0",
    ]

    assert select_versions(candidates) == ["2.10.10", "2.9.12"]


def test_manual_versions_must_have_compatible_wheels():
    candidates = ["2.9.0+cpu", "2.10.0+cpu"]

    assert select_versions(candidates, "2.9.0, 2.10.0, 2.9.0") == ["2.9.0", "2.10.0"]
    with pytest.raises(ValueError, match="No compatible stable CPU wheel"):
        select_versions(candidates, "2.11.0")


def test_fails_when_fewer_than_two_stable_series_are_available():
    with pytest.raises(ValueError, match="Expected two stable PyTorch series"):
        select_versions(["2.10.0+cpu", "2.10.1+cpu", "2.11.0rc1+cpu"])
