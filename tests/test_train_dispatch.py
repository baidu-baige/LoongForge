# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import loongforge.train as train
from loongforge.data import TrainingPhase
from loongforge.models import catalog

ROOT = Path(__file__).resolve().parents[1]
PHASES = (TrainingPhase.PRETRAIN, TrainingPhase.SFT)


def _literal_assignment(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def _previous_method(family, phase):
    """Module path that the removed @register_model_trainer factories ran for (family, phase), or None."""
    if family in catalog.LanguageModelFamilies.names():
        return {"pretrain": "loongforge.training.llm.pretrain", "sft": "loongforge.training.llm.sft"}[phase]
    if family == catalog.VisionLanguageModelFamilies.INTERN_VL:
        return {"sft": "loongforge.training.vlm.sft_internvl"}.get(phase)
    if family == catalog.VisionLanguageModelFamilies.ERNIE4_5_VL:
        return {"sft": "loongforge.training.vlm.sft_ernie"}.get(phase)
    if family in catalog.VisionLanguageModelFamilies.names():
        return {"pretrain": "loongforge.training.vlm.pretrain_vlm", "sft": "loongforge.training.vlm.sft_vlm"}[phase]
    if family in (catalog.DiffusionModelFamilies.WAN2_1_I2V, catalog.DiffusionModelFamilies.WAN2_2_I2V):
        return {"pretrain": "loongforge.training.diffusion.pretrain_wan"}.get(phase)
    if family == catalog.DiffusionModelFamilies.QWEN_IMAGE:
        return {"pretrain": "loongforge.training.diffusion.pretrain_qwen_image"}.get(phase)
    return None


class TrainDispatchTest(unittest.TestCase):
    def test_torch_trainer_resolves_only_selected_factory(self):
        cases = (
            ("Gr00tN1d6Trainer", "sft", "gr00t_n1_6"),
            ("Gr00tN1d7Trainer", "sft", "gr00t_n1_7"),
            ("LingBotFinetuneTrainer", "sft_lingbot_va", "lingbot_va"),
        )
        for trainer, method_name, model in cases:
            method = object()
            factory = SimpleNamespace(build_hooks=object())
            with self.subTest(trainer=trainer), \
                    patch.object(train.importlib, "import_module", side_effect=[method, factory]) as load:
                self.assertEqual(train.resolve_torch_trainer(trainer), (method, factory.build_hooks))
                self.assertEqual([call.args[0] for call in load.call_args_list], [
                    f"loongforge.training.embodied.{method_name}",
                    f"loongforge.training.embodied.custom.{model}.factory",
                ])

    def test_default_torch_trainer_and_invalid_selection(self):
        port = type("TrainerHooks", (), {})
        module = SimpleNamespace(TrainerHooks=port)
        with patch.dict(sys.modules, {"loongforge.engines.torch.train_step": module}), \
                patch.object(train.importlib, "import_module") as load:
            method, build = train.resolve_torch_trainer("FinetuneTrainer")
            self.assertIs(method, load.return_value)
            self.assertIsInstance(build(None), port)
            load.assert_called_once_with("loongforge.training.embodied.sft")
            load.reset_mock()
            for name in ("", "missing"):
                with self.assertRaises(ValueError):
                    train.resolve_torch_trainer(name)
            load.assert_not_called()

    def test_routes_registered_models_and_config_only_megatron(self):
        cases = (
            (["train.py", "--model-name", "qwen3-0.6b"], "megatron"),
            (["train.py", "--model-name", "pi05"], "torch"),
            (["train.py", "--model-name", "GR00T-N1-6"], "torch"),
            (["train.py", "--model-name", "pi05", "--config-file", "custom.yaml"], "torch"),
            (["train.py", "--config-file", "custom.yaml"], "megatron"),
        )
        for argv, engine in cases:
            result = object()
            with (
                self.subTest(argv=argv),
                patch.object(sys, "argv", argv),
                patch.object(train, "_run_megatron", return_value=result) as run_megatron,
                patch.object(train, "_run_torch", return_value=result) as run_torch,
            ):
                self.assertIs(train.main(), result)
                selected, skipped = (run_megatron, run_torch) if engine == "megatron" else (run_torch, run_megatron)
                selected.assert_called_once_with()
                skipped.assert_not_called()

    def test_megatron_family_and_phase_resolve_to_previous_method(self):
        families = [
            *catalog.LanguageModelFamilies.names(),
            *catalog.VisionLanguageModelFamilies.names(),
            *catalog.DiffusionModelFamilies.names(),
        ]
        for family in families:
            for phase in PHASES:
                expected = _previous_method(family, phase)
                with self.subTest(family=family, phase=phase), \
                        patch.object(train.importlib, "import_module") as load:
                    if expected is None:
                        with self.assertRaises(ValueError):
                            train.resolve_method(family, phase)
                        load.assert_not_called()
                    else:
                        train.resolve_method(family, phase)
                        load.assert_called_once_with(expected)

    def test_split_model_config_reads_top_level_or_model_section(self):
        llm = catalog.LanguageModelFamilies.names()[0]
        vlm = catalog.VisionLanguageModelFamilies.names()[0]
        top = SimpleNamespace(model_type=llm)
        self.assertEqual(catalog.split_model_config(top), (llm, top))
        nested = SimpleNamespace(model_type=vlm)
        self.assertEqual(catalog.split_model_config(SimpleNamespace(model_type=vlm, model=nested)), (vlm, nested))
        self.assertEqual(catalog.split_model_config(SimpleNamespace(model=nested)), (vlm, nested))
        with self.assertRaises(ValueError):
            catalog.split_model_config(SimpleNamespace(model_type=vlm))

    def test_megatron_methods_train_through_pretrain(self):
        for module in sorted(set(train.METHOD_REGISTRY.values())):
            with self.subTest(module=module):
                tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
                train_fn = next(
                    (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"), None
                )
                self.assertIsNotNone(train_fn)
                self.assertEqual([arg.arg for arg in train_fn.args.args], ["args"])
                called = {
                    node.func.id for node in ast.walk(train_fn)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                }
                self.assertIn("pretrain", called)

    def test_rejects_unknown_and_ambiguous_models(self):
        with patch.object(sys, "argv", ["train.py", "--model-name", "missing"]), self.assertRaises(SystemExit):
            train.main()
        with (
            patch.object(sys, "argv", ["train.py", "--model-name", "same-name"]),
            patch.dict(catalog.MODEL_CONFIG_REGISTRY, {
                "same-name": {"engine": "megatron"}, "same_name": {"engine": "torch"},
            }),
            self.assertRaises(SystemExit),
        ):
            train.main()

    def test_default_paths_are_independent_of_working_directory(self):
        for name, relative in (
            ("qwen3-0.6b", "configs/models/qwen3/qwen3_0_6b.yaml"),
            ("pi05", "configs/models/pi05/pi05.yaml"),
            ("cosmos3-nano", "configs/models/cosmos3/nano.yaml"),
        ):
            with self.subTest(name=name):
                self.assertEqual(Path(catalog.get_config_path(name)), ROOT / relative)
        with patch.dict(catalog.MODEL_CONFIG_REGISTRY, {
            "missing-file": {"engine": "torch", "config_path": "configs", "config_name": "missing"},
        }), self.assertRaises(FileNotFoundError):
            catalog.get_config_path("missing-file")

    def test_all_bindings_point_to_existing_configs_and_types(self):
        for name, entry in catalog.MODEL_CONFIG_REGISTRY.items():
            with self.subTest(name=name):
                self.assertIn(entry["engine"], {"megatron", "torch"})
                self.assertTrue(Path(catalog.get_config_path(name)).is_file())
                if entry["engine"] != "torch":
                    continue
                for field in ("model_config", "data_config"):
                    module, cls = entry[field].rsplit(".", 1)
                    tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text())
                    self.assertIn(cls, {node.name for node in tree.body if isinstance(node, ast.ClassDef)})

    def test_torch_catalog_model_types_are_registered(self):
        packages = _literal_assignment(ROOT / "loongforge/models/embodied/registry.py", "MODEL_PACKAGES")
        for name, entry in catalog.MODEL_CONFIG_REGISTRY.items():
            if entry["engine"] != "torch":
                continue
            config_text = Path(catalog.get_config_path(name)).read_text(encoding="utf-8")
            match = re.search(r"^\s*model_type:\s*([^#\s]+)", config_text, re.MULTILINE)
            self.assertIsNotNone(match, name)
            model_type = match.group(1).strip("'\\\"")
            self.assertIn(model_type, packages, name)

    def test_config_types_are_loaded_only_for_selected_model(self):
        model_cls, data_cls = type("ModelConfig", (), {}), type("DataConfig", (), {})
        modules = [SimpleNamespace(Pi05ModelConfig=model_cls), SimpleNamespace(Pi05DataConfig=data_cls)]
        with patch.object(catalog, "import_module", side_effect=modules) as load:
            self.assertEqual(catalog.get_config_types("pi05"), (model_cls, data_cls))
            self.assertEqual([call.args[0] for call in load.call_args_list], [
                "loongforge.models.embodied.pi05.model_configuration_pi05",
                "loongforge.data.embodied.pi05.data_configuration_pi05",
            ])
        with self.assertRaises(ValueError):
            catalog.get_config_types("qwen3-0.6b")

    def test_catalog_import_does_not_load_training_dependencies(self):
        subprocess.run([sys.executable, "-c", """
import sys
from loongforge.models.catalog import get_model_entry, get_config_path
import loongforge.data
assert get_model_entry('pi05')['engine'] == 'torch'
assert get_model_entry('QWEN3-0.6B')['engine'] == 'megatron'
assert get_config_path('qwen3-0.6b').endswith('/configs/models/qwen3/qwen3_0_6b.yaml')
assert not any(name.split('.')[0] in {'torch', 'megatron', 'transformers', 'omegaconf'} for name in sys.modules)
assert not any(name.startswith('loongforge.engines.') or name.startswith('loongforge.training.') for name in sys.modules)
"""], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
