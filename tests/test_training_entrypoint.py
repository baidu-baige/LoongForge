# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from loongforge.__main__ import resolve_train
from loongforge.checkpoint.manifest import is_hf_checkpoint, read_torch_metadata, write_torch_metadata
from loongforge.contracts.checkpoint import TorchCheckpointMetadata
from loongforge.engine.common import run_train
from loongforge.models.catalog import MCORE_CONFIGS, TORCH_CONFIGS, get_model_spec

ROOT = Path(__file__).resolve().parents[1]


class TrainingEntrypointTest(unittest.TestCase):
    def test_catalog_and_recipe_overrides(self):
        for model in (*MCORE_CONFIGS, *TORCH_CONFIGS):
            with self.subTest(model=model):
                self.assertTrue(get_model_spec(model).config_file.is_file())
        for recipe, engine, flag in (
            ("pi05_sft.yaml", "torch", "--lr-base"),
            ("qwen3_0.6b_pretrain.yaml", "mcore", "--lr"),
        ):
            spec = resolve_train(None, None, ROOT / "configs/recipes" / recipe, [flag, "0.02"])
            parser = argparse.ArgumentParser()
            parser.add_argument(flag, type=float)
            parsed, _ = parser.parse_known_args(spec.args)
            self.assertEqual(vars(parsed)[flag[2:].replace("-", "_")], 0.02)
            self.assertEqual(spec.engine, engine)
        spec = resolve_train("mcore", None, None, ["--config-file", str(get_model_spec("qwen3-0.6b").config_file)])
        self.assertEqual(spec.engine, "mcore")

    def test_rejects_conflicts_and_malformed_recipes(self):
        with self.assertRaisesRegex(ValueError, "supports torch"):
            resolve_train("mcore", "pi05", None, [])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_train(None, "pi05", None, ["--model-name", "xvla"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("engine: torch\nmodel: pi05\nargs: [12]\n")
            with self.assertRaisesRegex(ValueError, "argument strings"):
                resolve_train(None, None, path, [])

    def test_dry_run_never_imports_engines(self):
        code = (
            "from loongforge.__main__ import main; import sys; "
            "main(['train', '--model', 'pi05', '--dry-run']); "
            "assert not any(m in sys.modules for m in "
            "('torch', 'megatron', 'loongforge.engine.torch.entrypoint', 'loongforge.engine.mcore.entrypoint'))"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)["engine"], "torch")

    def test_shared_utils_import_stays_megatron_free(self):
        code = """
import sys, types
try:
    import torch  # noqa: F401
except ImportError:
    torch = types.ModuleType('torch')
    torch.distributed = types.ModuleType('torch.distributed')
    sys.modules['torch'] = torch
    sys.modules['torch.distributed'] = torch.distributed
import loongforge.utils
leaked = [m for m in sys.modules if m == 'megatron' or m.startswith(('megatron.', 'loongforge.engine.mcore'))]
assert not leaked, leaked
"""
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, check=True)

    @unittest.skipUnless(importlib.util.find_spec("omegaconf"), "requires the Torch config dependency")
    def test_torch_recipe_reaches_typed_parser(self):
        code = """
from pathlib import Path
import sys
from loongforge.__main__ import resolve_train
from loongforge.engine.torch.parser import parse_train_args
spec = resolve_train(None, None, Path('configs/recipes/pi05_sft.yaml'), [
    '--lr-base', '0.0001', '--train-iters', '20',
    'model.action_dim=6', 'data.image_size=128',
])
sys.argv = ['loongforge train', *spec.args]
training, model, data = parse_train_args()
assert training.lr_base == 0.0001 and training.train_iters == 20
assert model.action_dim == 6 and data.image_size == 128
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)

    @unittest.skipUnless(importlib.util.find_spec("hatchling"), "requires the wheel build dependency")
    def test_wheel_runs_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(
                [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", directory],
                cwd=ROOT, capture_output=True, text=True, check=True,
            )
            installed = root / "installed"
            with ZipFile(next(root.glob("*.whl"))) as wheel:
                self.assertIn("tools/convert_checkpoint/module_convertor/model.py", wheel.namelist())
                wheel.extractall(installed)
            code = """
from pathlib import Path
from loongforge.__main__ import main
from loongforge.models.catalog import MCORE_CONFIGS, TORCH_CONFIGS, get_model_spec
for model in (*MCORE_CONFIGS, *TORCH_CONFIGS):
    assert get_model_spec(model).config_file.is_file()
main(['train', '--model', 'pi05', '--dry-run'])
"""
            subprocess.run(
                [sys.executable, "-c", code], cwd=root,
                env={**os.environ, "PYTHONPATH": str(installed)},
                capture_output=True, text=True, check=True,
            )

    def test_dispatch_restores_arguments_even_on_failure(self):
        previous = sys.argv
        for model, module in (("pi05", "torch.entrypoint"), ("qwen3-0.6b", "mcore.entrypoint")):
            spec = resolve_train(None, model, None, ["--train-iters", "1"])
            with patch("loongforge.engine.common.import_module") as load:
                def main():
                    self.assertEqual(sys.argv[1:], list(spec.args))
                    raise RuntimeError("backend failed")
                load.return_value.main.side_effect = main
                with self.assertRaisesRegex(RuntimeError, "backend failed"):
                    run_train(spec)
                load.assert_called_once_with("loongforge.engine." + module)
            self.assertIs(sys.argv, previous)

    def test_metadata_publish_preserves_previous_marker_on_error(self):
        meta = TorchCheckpointMetadata(10, 1, "dcp", 2, False)
        with tempfile.TemporaryDirectory() as directory:
            write_torch_metadata(directory, meta)
            with patch("loongforge.checkpoint.manifest.json.dump", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    write_torch_metadata(directory, TorchCheckpointMetadata(20, 2, "dcp", 2, False))
            self.assertEqual(read_torch_metadata(directory), meta)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["resume_meta.json"])
            self.assertFalse(is_hf_checkpoint(directory))
            (Path(directory) / "model.safetensors.index.json").write_text("{}")
            self.assertTrue(is_hf_checkpoint(directory))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_public_parameter_groups_update_only_trainable_weights(self):
        import torch
        from loongforge.engine.torch.training_args import TrainingArgs
        from loongforge.optim.param_groups import build_param_groups

        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 1))
        model[0].bias.requires_grad_(False)
        args = TrainingArgs(lr_base=0.01, lr_group="0=0.002", weight_decay=0.1, weight_decay_grouping="bias_norm")
        groups = build_param_groups(model, args)
        params = [p for group in groups for p in group["params"]]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(len(params), len(expected))
        self.assertEqual({id(p) for p in params}, expected)
        for group in groups:
            for p in group["params"]:
                if id(p) in {id(model[1].weight), id(model[1].bias), id(model[2].bias)}:
                    self.assertEqual(group["weight_decay"], 0.0)
        frozen = model[0].bias.detach().clone()
        before = model[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(groups, lr=args.lr_base)
        for _ in range(2):
            optimizer.zero_grad()
            loss = model(torch.ones(3, 4)).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.equal(frozen, model[0].bias))
        self.assertFalse(torch.equal(before, model[0].weight))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_public_collectives_on_two_cpu_ranks(self):
        code = """
import torch
import torch.distributed as dist
from loongforge.distributed.context import rank, world_size
from loongforge.distributed.collectives import all_reduce_mean
dist.init_process_group('gloo')
try:
    assert world_size() == 2
    value = torch.tensor(float(rank()), requires_grad=True)
    mean = all_reduce_mean(value)
    assert mean.item() == 0.5 and not mean.requires_grad
    assert value.item() == rank()
finally:
    dist.destroy_process_group()
"""
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "collectives.py"
            script.write_text(code)
            subprocess.run(
                [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", str(script)],
                cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
                capture_output=True, text=True, check=True, timeout=60,
            )


if __name__ == "__main__":
    unittest.main()
