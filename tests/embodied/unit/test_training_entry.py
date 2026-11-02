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
    def test_model_factory_receives_training_args(self):
        from loongforge.training.embodied import sft

        trainer = sft.FinetuneTrainer(SimpleNamespace(train_iters=1), object(), object())
        with patch.object(sft, "build_model") as build:
            self.assertIs(trainer._build_model(), build.return_value)
        build.assert_called_once_with(trainer.model_cfg, trainer.training_args)

    def test_model_rng_instruction_is_applied_once(self):
        from loongforge.training.embodied import sft

        trainer = sft.FinetuneTrainer(SimpleNamespace(train_iters=1), object(), object())
        hook = Mock(side_effect=[(17, 5), None])
        trainer.model = SimpleNamespace(on_after_train_batch_fetch=hook)
        batch = object()
        with patch.object(sft, "set_seed") as seed, patch.object(sft.torch, "rand") as rand:
            trainer._on_after_train_batch_fetch(batch, 0)
            trainer._on_after_train_batch_fetch(batch, 1)
        seed.assert_called_once_with(17)
        rand.assert_called_once_with(5)
        self.assertEqual(hook.call_count, 2)


if __name__ == "__main__":
    unittest.main()
