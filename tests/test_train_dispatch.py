# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

import loongforge.train as train
from loongforge.models import catalog
from loongforge.training import trainer_builder


@pytest.mark.parametrize(("argv", "engine"), [
    (["train.py", "--model-name", "GR00T-N1-6"], "torch"),
    (["train.py", "--model-name", "pi05", "--config-file", "custom.yaml"], "torch"),
    (["train.py", "--model-name", "qwen3-8b"], "megatron"),
    (["train.py", "--config-file", "model.yaml"], "megatron"),
    (["train.py"], "megatron"),
])
def test_model_name_selects_the_default_engine(argv, engine):
    trainer = Mock()
    with (
        patch.object(sys, "argv", argv),
        patch.object(train, "build_model_trainer", return_value=trainer) as build,
    ):
        train.main()
    build.assert_called_once_with(engine)
    trainer.train.assert_called_once_with()


@pytest.fixture
def torch_engine(monkeypatch):
    """An engine adapter that returns a fixed lookup key."""
    key = {}

    def adapter():
        return key["family"], key["phase"], key["trainer_type"], key["configs"]

    monkeypatch.setitem(trainer_builder.ENGINES, "torch", adapter)
    monkeypatch.setitem(trainer_builder.TRAINERS_BY_ENGINE, "torch", {})
    monkeypatch.setitem(trainer_builder.TRAINERS_BY_NAME, "torch", {})
    return key


def register(family, phase, name, recipe=None):
    def _register(fn):
        return trainer_builder.register_model_trainer(
            family, phase, fn, engine="torch"
        )

    fn = recipe or Mock(name=name)
    fn.__name__ = name
    _register(fn)
    return fn


@pytest.mark.parametrize(("family", "phase", "message"), [
    ("unregistered", "pretrain", "Not found trainer for family: unregistered"),
    ("phase_demo", "pretrain", "Loongforge not support pretrain phase for family phase_demo"),
])
def test_unregistered_family_or_phase_is_rejected(torch_engine, family, phase, message):
    register("phase_demo", "sft", "phase_recipe")
    torch_engine.update(family=family, phase=phase, trainer_type=None, configs=())
    with pytest.raises(ValueError, match=message):
        trainer_builder.build_model_trainer("torch")


def test_trainer_type_selects_a_registered_recipe_without_a_family(torch_engine):
    recipe = register([], "sft", "TensorWiseTrainer")
    torch_engine.update(
        family="named_demo", phase="sft", trainer_type="TensorWiseTrainer", configs=(1,)
    )
    assert trainer_builder.build_model_trainer("torch") is recipe.return_value


def test_unknown_trainer_type_is_rejected(torch_engine):
    register([], "sft", "TensorWiseTrainer")
    torch_engine.update(
        family="named_demo", phase="sft", trainer_type="MissingTrainer", configs=()
    )
    with pytest.raises(ValueError, match="Unknown --trainer-type 'MissingTrainer'"):
        trainer_builder.build_model_trainer("torch")


def test_unknown_engine_registration_is_rejected():
    with pytest.raises(ValueError, match="Unknown engine 'nope'"):
        trainer_builder.register_model_trainer("demo", "sft", engine="nope")


def test_duplicate_trainer_name_is_rejected(torch_engine):
    register("dup_demo", "sft", "same_name")
    with pytest.raises(ValueError, match="Duplicate trainer name 'same_name' for torch sft"):
        register("other_demo", "sft", "same_name")


def test_same_trainer_registered_twice_is_allowed(torch_engine):
    recipe = register("twice_demo", "sft", "twice_recipe")
    assert register("twice_demo", "sft", "twice_recipe", recipe=recipe) is recipe
    torch_engine.update(family="twice_demo", phase="sft", trainer_type=None, configs=())
    assert trainer_builder.build_model_trainer("torch") is recipe.return_value


def test_default_config_lookup_is_independent_of_working_directory(tmp_path, monkeypatch):
    expected = catalog.get_config_from_model_name("qwen3-0.6b")
    spec = catalog.get_model_spec("qwen3-0.6b")
    assert spec.engine == "megatron"
    assert spec.model_family == catalog.LanguageModelFamilies.QWEN
    monkeypatch.chdir(tmp_path)
    actual = catalog.get_config_from_model_name("QWEN3-0.6B")
    assert actual == expected
    directory, name = actual
    assert Path(directory).is_absolute()
    assert (Path(directory) / f"{name}.yaml").is_file()


def test_unknown_model_config_is_rejected():
    with pytest.raises(KeyError, match="Unknown model_name"):
        catalog.get_config_from_model_name("missing")


def test_catalog_groups_models_by_engine_and_family():
    qwen_config = catalog.MODEL_CONFIG_REGISTRY["megatron"][catalog.LanguageModelFamilies.QWEN]
    assert qwen_config["qwen3-8b"] == "qwen3/qwen3_8b.yaml"

    pi05_config = catalog.MODEL_CONFIG_REGISTRY["torch"][catalog.EmbodiedModelFamilies.PI05]
    assert pi05_config["pi05"][0] == "pi05/pi05.yaml"
    assert len(pi05_config["pi05"]) == 3

    spec = catalog.get_model_spec("GR00T-N1-6")
    assert spec.engine == "torch"
    assert spec.model_family == catalog.EmbodiedModelFamilies.GR00T_N1_6
    assert spec.yaml_file == "gr00t_n1_6/gr00t_n1_6.yaml"
