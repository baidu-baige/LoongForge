# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training entry dispatch tests."""

import argparse
import io
import re
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from loongforge import train
from loongforge.training import trainer_builder


def _builder_that_parses(parsed, engines):
    """Emulate an engine parser: record the engine, then parse the argv."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model-name")

    def build(engine):
        engines.append(engine)
        parsed.append(parser.parse_known_args())
        return Mock()

    return build


class TrainingEntryTests(unittest.TestCase):
    def test_help_and_missing_values_reach_the_engine_parser(self):
        cases = (
            (["--help", "--model-name"], "megatron", 0),
            (["--help", "--model-name", "pi05"], "megatron", 0),
            (["--model-name", "pi05", "--help", "--model-name"], "torch", 0),
            (["--model-name", "pi05", "--help", "--model-name", "qwen3-8b"], "torch", 0),
            (["--model-name", "qwen3-8b", "-h", "--model-name", "pi05"], "megatron", 0),
            (["--model-n", "pi05", "--help", "--model-name"], "megatron", 0),
            (["--model-name"], "megatron", 2),
        )
        for argv, engine, exit_code in cases:
            with self.subTest(argv=argv):
                engines, parsed = [], []
                with (
                    patch.object(sys, "argv", ["train.py", *argv]),
                    patch.object(
                        train,
                        "build_model_trainer",
                        side_effect=_builder_that_parses(parsed, engines),
                    ),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as result,
                ):
                    train.main()
                self.assertEqual(result.exception.code, exit_code)
                self.assertEqual(engines, [engine])

    def test_exact_model_name_controls_routing(self):
        cases = (
            (["--model-n", "pi05"], "megatron", None, ["--model-n", "pi05"]),
            (["--model=pi05"], "megatron", None, ["--model=pi05"]),
            (["--model-n", "qwen3-8b"], "megatron", None, ["--model-n", "qwen3-8b"]),
            (["--model-name", "qwen3-8b", "--model-n", "pi05"],
             "megatron", "qwen3-8b", ["--model-n", "pi05"]),
            (["--model-name", "pi05", "--model-n", "qwen3-8b"],
             "torch", "pi05", ["--model-n", "qwen3-8b"]),
            (["--model-name", "pi05", "--model-name", "qwen3-8b"],
             "megatron", "qwen3-8b", []),
            (["--model-name", "qwen3-8b", "--model-name", "pi05"],
             "torch", "pi05", []),
            (["--model-name", "pi05", "--", "--model-name", "qwen3-8b"],
             "torch", "pi05", ["--", "--model-name", "qwen3-8b"]),
        )
        for argv, engine, model_name, unknown in cases:
            with self.subTest(argv=argv):
                engines, parsed = [], []
                with (
                    patch.object(sys, "argv", ["train.py", *argv]),
                    patch.object(
                        train,
                        "build_model_trainer",
                        side_effect=_builder_that_parses(parsed, engines),
                    ),
                ):
                    self.assertIsNone(train.main())
                    self.assertEqual(sys.argv, ["train.py", *argv])
                self.assertEqual(engines, [engine])
                self.assertEqual(parsed[0][0].model_name, model_name)
                self.assertEqual(parsed[0][1], unknown)

    def _parse_torch_args(self, argv):
        pytest.importorskip("torch")
        from loongforge.training.torch_args import build_arg_parser

        return build_arg_parser().parse_args(argv)

    def test_torch_parser_defaults_to_sft_phase(self):
        pytest.importorskip("torch")
        from dataclasses import fields

        from loongforge.training.torch_args import TrainingArgs

        phases = {f.name: f for f in fields(TrainingArgs)}
        self.assertEqual(phases["training_phase"].default, "sft")
        # CLI defaults are SUPPRESS, so an unset phase falls back to the dataclass default.
        self.assertFalse(hasattr(self._parse_torch_args([]), "training_phase"))

    def test_torch_parser_accepts_an_explicit_phase(self):
        self.assertEqual(
            self._parse_torch_args(["--training-phase", "pretrain"]).training_phase,
            "pretrain",
        )

    def test_torch_parser_rejects_unknown_phase_and_abbreviations(self):
        for argv in (["--training-phase", "rl"], ["--model-n", "pi05"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self._parse_torch_args(argv)


@pytest.mark.parametrize(("argv", "engine"), [
    (["train.py", "--model-name", "GR00T-N1-6"], "torch"),
    (["train.py", "--model-name", "pi05", "--config-file", "custom.yaml"], "torch"),
    (["train.py", "--model-name", "qwen3-8b"], "megatron"),
    (["train.py", "--config-file", "model.yaml"], "megatron"),
    (["train.py"], "megatron"),
    (["train.py", "--help"], "megatron"),
    (["train.py", "--model-name", "unknown"], None),
])
def test_model_name_selects_the_default_engine(argv, engine):
    trainer = Mock()
    with (
        patch.object(sys, "argv", argv),
        patch.object(train, "build_model_trainer", return_value=trainer) as build,
    ):
        if engine is None:
            with pytest.raises(KeyError, match="Unknown model_name"):
                train.main()
            build.assert_not_called()
            return
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


@pytest.mark.parametrize("trainer_type", ["FamilyTrainer", "MissingTrainer"])
def test_unknown_trainer_type_is_rejected(torch_engine, trainer_type):
    register([], "sft", "TensorWiseTrainer")
    register("named_demo", "sft", "FamilyTrainer")
    torch_engine.update(
        family="named_demo", phase="sft", trainer_type=trainer_type, configs=()
    )
    # Only trainers registered without a family can be selected by name.
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Unknown --trainer-type '{trainer_type}'. Available: ['TensorWiseTrainer']"
        ),
    ):
        trainer_builder.build_model_trainer("torch")


def test_duplicate_trainer_name_is_rejected(torch_engine):
    register([], "sft", "same_name")
    with pytest.raises(ValueError, match="Duplicate trainer name 'same_name' for torch sft"):
        register([], "sft", "same_name")


def test_recipe_registry_keeps_family_case_and_override_rules(monkeypatch):
    registry = {}
    monkeypatch.setitem(trainer_builder.TRAINERS_BY_ENGINE, "megatron", registry)
    original = Mock(__name__="original")
    replacement = Mock(__name__="replacement")
    sft = Mock(__name__="sft")
    register = trainer_builder.register_model_trainer
    assert register(["LLAMA", "qwen3"], "pretrain", engine="megatron")(original) is original
    assert registry == {
        "llama": {"pretrain": original},
        "qwen3": {"pretrain": original},
    }
    register("llama", "pretrain", engine="megatron")(replacement)
    assert registry["llama"]["pretrain"] is original
    register("llama", "pretrain", override=True, engine="megatron")(replacement)
    assert registry["llama"]["pretrain"] is replacement
    assert register("LLAMA", "sft", training_func=sft, engine="megatron") is None
    assert registry["llama"]["sft"] is sft


def test_builder_uses_the_registered_phase_of_the_selected_engine(monkeypatch):
    recipe = Mock()
    args = SimpleNamespace(model_name="qwen3-8b", model_family="qwen3", training_phase="pretrain")
    monkeypatch.setattr(
        trainer_builder,
        "TRAINERS_BY_ENGINE",
        {"megatron": {"qwen3": {"pretrain": recipe}}, "torch": {}},
    )
    monkeypatch.setitem(
        trainer_builder.ENGINES,
        "megatron",
        lambda: ("qwen3", "pretrain", None, (args,)),
    )
    trainer_builder.build_model_trainer("megatron")
    recipe.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
