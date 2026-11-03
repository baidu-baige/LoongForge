# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_vlm_dataset_inputs import REPO_ROOT, _load_functions


def test_dreamzero_vae_loader_executes_and_returns_module():
    loader = Mock()
    spec = SimpleNamespace(loader=loader)
    module = object()
    util = SimpleNamespace(
        spec_from_file_location=Mock(return_value=spec),
        module_from_spec=Mock(return_value=module),
    )
    namespace = _load_functions(
        "tools/dreamzero_data_preprocess/cache_precompute/features.py",
        {"_load_vae_module"},
        {"_REPO_ROOT": REPO_ROOT, "importlib": SimpleNamespace(util=util)},
    )

    assert namespace["_load_vae_module"]() is module
    path = util.spec_from_file_location.call_args.args[1]
    assert Path(path).is_file()
    util.module_from_spec.assert_called_once_with(spec)
    loader.exec_module.assert_called_once_with(module)


@pytest.mark.parametrize("relative_path", [
    "loongforge/evaluation/embodied/orchestrator/server_manager.py",
    "loongforge/evaluation/embodied/orchestrator/runners/robotwin_runner.py",
])
def test_evaluation_cli_accepts_custom_eval_root(relative_path):
    namespace = _load_functions(
        relative_path,
        {"build_argparser"},
        {"argparse": argparse, "os": os, "sys": sys},
    )
    args = namespace["build_argparser"]().parse_args(["--eval-root", "/custom"])

    assert args.eval_root == "/custom"
