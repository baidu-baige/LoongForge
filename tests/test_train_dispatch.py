# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import loongforge.train as train
from loongforge.models import catalog


@pytest.mark.parametrize("argv", [
    ["train.py", "--model-name", "GR00T-N1-6"],
    ["train.py", "--model-name", "pi05", "--config-file", "custom.yaml"],
])
def test_torch_model_name_controls_engine_selection(argv):
    with (
        patch.object(sys, "argv", argv),
        patch.object(train, "_run_megatron") as run_megatron,
        patch.object(train, "_run_torch") as run_torch,
    ):
        train.main()
    run_torch.assert_called_once_with()
    run_megatron.assert_not_called()


@pytest.mark.parametrize(("family", "phase"), [
    ("qwen3", "sft"),
    ("qwen_image", "pretrain"),
    ("intern_vl", "pretrain"),
    ("qwen3_vl", "sft"),
])
def test_megatron_dispatches_to_recipe_builder(family, phase):
    from loongforge.training import trainer_builder

    args = SimpleNamespace(model_name="selected-model", model_family=family, training_phase=phase)
    trainer = Mock()
    parser = SimpleNamespace(parse_train_args=Mock(return_value=args))
    with (
        patch.object(trainer_builder, "build_model_trainer", return_value=trainer) as build,
        patch.dict(sys.modules, {
            "loongforge.engines.megatron.parser": parser,
        }),
    ):
        train._run_megatron()
    build.assert_called_once_with(args)
    trainer.train.assert_called_once_with()


@pytest.mark.parametrize(("family", "phase", "message"), [
    (
        "unregistered", "pretrain",
        "Not found trainer for family: unregistered",
    ),
    (
        "qwen_image", "sft",
        "Loongforge not support sft phase for selected-model",
    ),
])
def test_megatron_rejects_unregistered_family_or_phase(family, phase, message):
    from loongforge.training import trainer_builder

    args = SimpleNamespace(model_name="selected-model", model_family=family, training_phase=phase)
    recipe = Mock()
    parser = SimpleNamespace(parse_train_args=Mock(return_value=args))
    with (
        patch.object(
            trainer_builder,
            "build_model_trainer",
            side_effect=ValueError(message),
        ) as build,
        patch.dict(sys.modules, {
            "loongforge.engines.megatron.parser": parser,
        }),
        pytest.raises(ValueError, match=message),
    ):
        train._run_megatron()
    build.assert_called_once_with(args)
    recipe.assert_not_called()


def test_default_config_lookup_is_independent_of_working_directory(tmp_path, monkeypatch):
    expected = catalog.get_config_from_model_name("qwen3-0.6b")
    monkeypatch.chdir(tmp_path)
    actual = catalog.get_config_from_model_name("QWEN3-0.6B")
    assert actual == expected
    directory, name = actual
    assert Path(directory).is_absolute()
    assert (Path(directory) / f"{name}.yaml").is_file()


def test_unknown_model_config_is_rejected():
    with pytest.raises(KeyError, match="Unknown model_name"):
        catalog.get_config_from_model_name("missing")
