# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training dispatch and SFT callback regression tests."""

import importlib.util
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from loongforge import train


class TrainingEntryTests(unittest.TestCase):
    def test_entry_does_not_import_torch(self):
        subprocess.run(
            [sys.executable, "-c",
             "import sys; import loongforge.train; assert 'torch' not in sys.modules"],
            check=True,
        )

    def test_unknown_trainer_fails_before_import(self):
        with patch.object(train.importlib, "import_module") as import_module:
            for name in ("", "WallOss05Trainer", "unknown"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    train.resolve_torch_trainer(name)
            import_module.assert_not_called()

    def test_wall_oss_uses_shared_finetune_recipe(self):
        args = SimpleNamespace(trainer_type="FinetuneTrainer")
        configs = (args, object(), object())
        trainer_type = Mock()
        parser = SimpleNamespace(parse_train_args=Mock(return_value=configs))
        recipe = SimpleNamespace(FinetuneTrainer=trainer_type)
        with (
            patch.dict(sys.modules, {"loongforge.training.torch_args": parser}),
            patch.object(train.importlib, "import_module", return_value=recipe) as load,
            patch.object(sys, "argv", ["train.py", "--model-name", "wall_oss_0_5"]),
        ):
            train.main()
        load.assert_called_once_with("loongforge.training.embodied.sft")
        trainer_type.assert_called_once_with(*configs)
        trainer_type.return_value.train.assert_called_once_with()

    def test_config_file_keeps_megatron_dispatch(self):
        with (
            patch.object(sys, "argv", ["train.py", "--config-file", "model.yaml"]),
            patch.object(train, "_run_megatron") as run_megatron,
            patch.object(train, "_run_torch") as run_torch,
        ):
            train.main()
        run_megatron.assert_called_once_with()
        run_torch.assert_not_called()


@unittest.skipUnless(
    sys.version_info >= (3, 10) and importlib.util.find_spec("torch") is not None,
    "SFT imports require Python >=3.10 and the Torch training dependencies",
)
class SFTRecipeTests(unittest.TestCase):
    def setUp(self):
        from loongforge.training.embodied import global_vars

        for name in (
            "_EMBODIED_TRAINING_ARGS", "_EMBODIED_MODEL_CONFIG", "_EMBODIED_DATA_CONFIG",
        ):
            patcher = patch.object(global_vars, name, None)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_tensor_wise_config_and_trainer_wiring(self):
        from omegaconf.errors import ConfigKeyError

        from loongforge.engines.torch.optimizer.tensor_wise_optimizer.config import TensorWiseConfig
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
        tensor_wise = train.resolve_torch_trainer("TensorWiseTrainer")
        self.assertTrue(issubclass(tensor_wise, FinetuneTrainer))
        self.assertTrue(issubclass(train.resolve_torch_trainer(args.trainer_type), tensor_wise))
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
