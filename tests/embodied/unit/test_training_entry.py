# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training dispatch and SFT callback regression tests."""

import argparse
import importlib.util
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from loongforge import train


def _builder_that_parses(parsed, engines, use_abbrev):
    """Emulate an engine parser: record the engine, then parse the argv."""
    parser = argparse.ArgumentParser(allow_abbrev=use_abbrev)
    parser.add_argument("--model-name")

    def build(engine):
        engines.append(engine)
        parsed.append(parser.parse_known_args())
        return Mock()

    return build


class TrainingEntryTests(unittest.TestCase):
    def test_entry_does_not_import_torch(self):
        subprocess.run(
            [sys.executable, "-c",
             "import sys; import loongforge.train; "
             "assert not any(name in sys.modules for name in ('torch', 'megatron')); "
             "assert not any(name.startswith('loongforge.engines.') for name in sys.modules)"],
            check=True,
        )

    def test_megatron_model_uses_the_megatron_engine(self):
        trainer = Mock()
        with (
            patch.object(train, "build_model_trainer", return_value=trainer) as build,
            patch.object(sys, "argv", ["train.py", "--model-name", "qwen3-8b"]),
        ):
            self.assertIsNone(train.main())
        build.assert_called_once_with("megatron")
        trainer.train.assert_called_once_with()

    def test_unknown_trainer_type_is_rejected(self):
        from loongforge.training import trainer_builder

        for name in ("WallOss05Trainer", "unknown"):
            with (
                self.subTest(name=name),
                patch.dict(
                    trainer_builder.ENGINES,
                    {"torch": lambda: ("pi05", "sft", name, ())},
                ),
                self.assertRaises(ValueError),
            ):
                trainer_builder.build_model_trainer("torch")

    def test_wall_oss_uses_shared_finetune_recipe(self):
        from loongforge.training import trainer_builder

        recipe = Mock()
        configs = (object(), object(), object())
        with (
            patch.dict(
                trainer_builder.ENGINES,
                {"torch": lambda: ("wall_oss_0_5", "sft", None, configs)},
            ),
            patch.object(
                trainer_builder,
                "TORCH_TRAINERS_BY_FAMILY",
                {"wall_oss_0_5": {"sft": recipe}},
            ),
            patch.object(sys, "argv", ["train.py", "--model-name", "wall_oss_0_5"]),
        ):
            train.main()
        recipe.assert_called_once_with(*configs)
        recipe.return_value.train.assert_called_once_with()

    def test_config_file_keeps_megatron_dispatch(self):
        with (
            patch.object(sys, "argv", ["train.py", "--config-file", "model.yaml"]),
            patch.object(train, "build_model_trainer") as build,
        ):
            train.main()
        build.assert_called_once_with("megatron")

    def test_help_and_unknown_model_reach_the_megatron_parser(self):
        for argv in (["--help"], ["--model-name", "unknown"], []):
            with (
                self.subTest(argv=argv),
                patch.object(sys, "argv", ["train.py", *argv]),
                patch.object(train, "build_model_trainer") as build,
            ):
                self.assertIsNone(train.main())
                self.assertEqual(sys.argv, ["train.py", *argv])
                build.assert_called_once_with("megatron")

    def test_torch_model_name_keeps_the_original_unique_abbreviation(self):
        with (
            patch.object(sys, "argv", ["train.py", "--model-n", "pi05"]),
            patch.object(train, "build_model_trainer") as build,
        ):
            self.assertIsNone(train.main())
        build.assert_called_once_with("torch")

    def test_help_and_missing_values_reach_the_engine_parser(self):
        cases = (
            (["--help", "--model-name"], "megatron", 0),
            (["--help", "--model-name", "pi05"], "megatron", 0),
            (["--model-name", "pi05", "--help", "--model-name"], "torch", 0),
            (["--model-name", "pi05", "--help", "--model-name", "qwen3-8b"], "torch", 0),
            (["--model-name", "qwen3-8b", "-h", "--model-name", "pi05"], "megatron", 0),
            (["--model-n", "pi05", "--help", "--model-name"], "torch", 0),
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
                        side_effect=_builder_that_parses(parsed, engines, engine == "torch"),
                    ),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as result,
                ):
                    train.main()
                self.assertEqual(result.exception.code, exit_code)
                self.assertEqual(engines, [engine])

    def test_exact_model_name_controls_routing_with_abbreviations(self):
        cases = (
            (["--model-n", "pi05"], "torch", "pi05", []),
            (["--model=pi05"], "torch", "pi05", []),
            (["--model-n", "qwen3-8b"], "megatron", None, ["--model-n", "qwen3-8b"]),
            (["--model-name", "qwen3-8b", "--model-n", "pi05"],
             "megatron", "qwen3-8b", ["--model-n", "pi05"]),
            (["--model-name", "pi05", "--model-n", "qwen3-8b"],
             "torch", "qwen3-8b", []),
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
                        side_effect=_builder_that_parses(parsed, engines, engine == "torch"),
                    ),
                ):
                    self.assertIsNone(train.main())
                    self.assertEqual(sys.argv, ["train.py", *argv])
                self.assertEqual(engines, [engine])
                self.assertEqual(parsed[0][0].model_name, model_name)
                self.assertEqual(parsed[0][1], unknown)


@unittest.skipUnless(
    sys.version_info >= (3, 10) and importlib.util.find_spec("torch") is not None,
    "SFT imports require Python >=3.10 and the Torch training dependencies",
)
class SFTRecipeTests(unittest.TestCase):
    def setUp(self):
        from loongforge.engines.torch.utils import global_vars

        for name in (
            "_EMBODIED_TRAINING_ARGS", "_EMBODIED_MODEL_CONFIG", "_EMBODIED_DATA_CONFIG",
        ):
            patcher = patch.object(global_vars, name, None)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_tensor_wise_config_and_trainer_wiring(self):
        from omegaconf.errors import ConfigKeyError

        from loongforge.engines.torch.optimizer.tensor_wise_optimizer.config import TensorWiseConfig
        from loongforge.training.embodied.custom import lingbot_vla_v2  # noqa: F401
        from loongforge.training.embodied.custom.tensor_wise_trainer import TensorWiseTrainer
        from loongforge.training.embodied.sft import FinetuneTrainer
        from loongforge.training.torch_args import parse_train_args

        argv = [
            "train.py", "--model-name", "lingbot_vla_v2",
            "--trainer-type", "LingbotVlaV2Trainer",
            "--distributed-strategy", "ddp", "--optimizer", "muon",
            "--no-save-training-state", "model.tensor_wise.grad_reduce_dtype=fp32",
        ]
        with patch.object(sys, "argv", argv):
            args, model_cfg, _ = parse_train_args()
        self.assertIsInstance(model_cfg.tensor_wise, TensorWiseConfig)
        self.assertEqual(model_cfg.tensor_wise.grad_reduce_dtype, "fp32")
        self.assertEqual(model_cfg.tensor_wise.comm_bucket_mb, 256)
        from loongforge.training import trainer_builder

        self.assertTrue(issubclass(TensorWiseTrainer, FinetuneTrainer))
        self.assertTrue(
            issubclass(
                trainer_builder.TORCH_TRAINERS_BY_NAME["sft"][args.trainer_type],
                TensorWiseTrainer,
            )
        )
        with patch.object(sys, "argv", argv + ["model.tensor_wise.unknown=1"]):
            with self.assertRaises(ConfigKeyError):
                parse_train_args()

    def test_lingbot_data_registry_uses_master_signature(self):
        import inspect

        from loongforge.data.embodied.lingbot_vla_v2 import build_lingbot_vla_v2_dataset
        from loongforge.data.embodied.transforms.collator import build_preprocessor
        from loongforge.data.embodied.transforms.registry import get_transform_builder
        from loongforge.models.embodied.lingbot_vla_v2 import upstream_adapter

        inspect.signature(build_lingbot_vla_v2_dataset).bind(object(), object(), object())
        self.assertEqual(get_transform_builder("lingbot_vla_v2")(object()), [])
        upstream_collator = Mock(return_value={"tokens": [1, 2]})
        with patch.object(upstream_adapter, "resolve_vla_data_collator", return_value=lambda: upstream_collator):
            collator = build_preprocessor("lingbot_vla_v2", object(), object())
        batch = collator([{"sample": 1}])
        self.assertEqual(batch.data, {"tokens": [1, 2]})
        upstream_collator.assert_called_once_with([{"sample": 1}])

    def test_model_factory_receives_model_config(self):
        from loongforge.training.embodied import sft

        trainer = sft.FinetuneTrainer(SimpleNamespace(train_iters=1), object(), object())
        with patch.object(sft, "build_model") as build:
            self.assertIs(trainer._build_model(), build.return_value)
        build.assert_called_once_with(trainer.model_cfg)

    def test_model_batch_hook_receives_training_context(self):
        from loongforge.training.embodied import sft

        trainer = sft.FinetuneTrainer(SimpleNamespace(train_iters=1), object(), object())
        hook = Mock(return_value=None)
        trainer.model = SimpleNamespace(on_after_train_batch_fetch=hook)
        batch = object()
        with patch.object(sft.torch, "rand") as rand:
            trainer._on_after_train_batch_fetch(batch, 0)
            trainer._on_after_train_batch_fetch(batch, 1)
        rand.assert_not_called()
        self.assertEqual(hook.call_count, 2)
        for micro_step, call in enumerate(hook.call_args_list):
            self.assertEqual(call.kwargs, {
                "args": trainer.training_args,
                "completed_steps": trainer.completed_steps,
                "micro_step": micro_step,
                "batch": batch,
            })


if __name__ == "__main__":
    unittest.main()
