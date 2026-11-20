# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tests.ci_test_utils import load_module

selector = load_module(
    "select_pytorch_versions",
    Path(__file__).resolve().parents[1] / ".github/scripts/select_pytorch_versions.py",
)
select_versions = selector.select_versions
build_python_torch_matrix = selector.build_python_torch_matrix


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


def test_matrix_respects_each_torch_releases_python_requirement():
    matrix = build_python_torch_matrix(
        ["2.15.0", "2.14.1"],
        ["3.10", "3.11", "3.12"],
        {"2.15.0": ">=3.11", "2.14.1": ">=3.10"},
    )

    assert matrix["include"] == [
        {"python-version": "3.11", "torch-version": "2.15.0"},
        {"python-version": "3.12", "torch-version": "2.15.0"},
        {"python-version": "3.10", "torch-version": "2.14.1"},
        {"python-version": "3.11", "torch-version": "2.14.1"},
        {"python-version": "3.12", "torch-version": "2.14.1"},
    ]


def test_matrix_fails_when_no_python_pair_is_supported():
    with pytest.raises(ValueError, match="No stable PyTorch release supports"):
        build_python_torch_matrix(
            ["2.15.0"],
            ["3.10"],
            {"2.15.0": ">=3.11"},
        )
