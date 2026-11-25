# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call

import pytest

from loongforge.cli import main


@pytest.mark.parametrize("legacy_script", [False, True])
def test_training_entry_passes_arguments_through(monkeypatch, legacy_script):
    runtime = ModuleType("loongforge.train")
    expected_args = ["--model-name", "qwen3-8b", "model.num_layers=2"]
    runner = Mock()

    def parse_train_args():
        assert sys.argv[1:] == expected_args
        return runner.parse_train_args()

    runtime.parse_train_args = parse_train_args
    runtime.build_model_trainer = runner.build_model_trainer
    monkeypatch.setitem(sys.modules, "loongforge.train", runtime)
    monkeypatch.setattr(sys, "argv", ["LoongForge", *expected_args])

    if legacy_script:
        script = Path(__file__).resolve().parents[1] / "loongforge" / "train.py"
        runpy.run_path(str(script), run_name="__main__")
    else:
        main()

    assert runner.mock_calls == [
        call.parse_train_args(),
        call.build_model_trainer(runner.parse_train_args.return_value),
        call.build_model_trainer().train(),
    ]
