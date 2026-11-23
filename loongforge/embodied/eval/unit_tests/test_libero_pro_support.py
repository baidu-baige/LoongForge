# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LIBERO-Pro (perturbation suites + BDDL instructions) support."""

from __future__ import annotations

from pathlib import Path

import pytest

from loongforge.embodied.eval.adapters.libero import base_suite_name
from loongforge.embodied.eval.orchestrator.runners.libero_runner import _bddl_language


@pytest.mark.parametrize(
    ("suite", "expected"),
    [
        ("libero_spatial", "libero_spatial"),
        ("libero_object", "libero_object"),
        ("libero_goal", "libero_goal"),
        ("libero_10", "libero_10"),
        ("libero_90", "libero_90"),
        ("libero_spatial_lan", "libero_spatial"),
        ("libero_spatial_object", "libero_spatial"),
        ("libero_spatial_swap", "libero_spatial"),
        ("libero_spatial_task", "libero_spatial"),
        ("libero_object_lan", "libero_object"),
        ("libero_object_object", "libero_object"),
        ("libero_object_swap", "libero_object"),
        ("libero_object_task", "libero_object"),
        ("libero_goal_lan", "libero_goal"),
        ("libero_goal_object", "libero_goal"),
        ("libero_goal_swap", "libero_goal"),
        ("libero_goal_task", "libero_goal"),
        ("libero_10_lan", "libero_10"),
        ("libero_10_object", "libero_10"),
        ("libero_10_swap", "libero_10"),
        ("libero_10_task", "libero_10"),
    ],
)
def test_base_suite_name(suite: str, expected: str) -> None:
    assert base_suite_name(suite) == expected


def test_base_suite_name_keeps_unknown_names() -> None:
    # The adapter's own validation reports unknown names; the mapper must not
    # silently rewrite them into a valid suite.
    assert base_suite_name("libero_unknown") == "libero_unknown"
    assert base_suite_name("libero_goal_custom") == "libero_goal_custom"


def test_bddl_language_reads_declared_instruction(tmp_path: Path) -> None:
    bddl = tmp_path / "open_the_middle_drawer_of_the_cabinet.bddl"
    bddl.write_text(
        "(define (problem LIBERO_Kitchen_Tabletop_Manipulation)\n"
        "  (:domain robosuite)\n"
        "  (:language open the cabinet's middle drawer)\n"
        "  (:goal (and (open (bottom_drawer))))\n"
        ")\n",
        encoding="utf-8",
    )
    assert _bddl_language(bddl) == "open the cabinet's middle drawer"


def test_bddl_language_missing_block_returns_empty(tmp_path: Path) -> None:
    bddl = tmp_path / "no_language.bddl"
    bddl.write_text("(define (problem X) (:domain robosuite))\n", encoding="utf-8")
    assert _bddl_language(bddl) == ""


def test_bddl_language_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _bddl_language(tmp_path / "does_not_exist.bddl") == ""
