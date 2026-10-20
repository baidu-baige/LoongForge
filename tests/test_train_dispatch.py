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
from loongforge import constants
from loongforge.models import catalog
from loongforge.training.megatron import registry
from loongforge.training.torch import registry as torch_registry

ROOT = Path(__file__).resolve().parents[1]
PHASES = (constants.TrainingPhase.PRETRAIN, constants.TrainingPhase.SFT)


def _literal_assignment(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def _previous_method(family, phase):
    """Method that the removed @register_model_trainer factories ran for (family, phase), or None."""
    if family in constants.LanguageModelFamilies.names():
        return {"pretrain": "pretrain_llm", "sft": "sft_llm"}[phase]
    if family == constants.VisionLanguageModelFamilies.INTERN_VL:
        return {"sft": "sft_internvl"}.get(phase)
    if family == constants.VisionLanguageModelFamilies.ERNIE4_5_VL:
        return {"sft": "sft_ernie"}.get(phase)
    if family in constants.VisionLanguageModelFamilies.names():
        return {"pretrain": "pretrain_vlm", "sft": "sft_vlm"}[phase]
    if family in (constants.CustomModelFamilies.WAN2_1_I2V, constants.CustomModelFamilies.WAN2_2_I2V):
        return {"pretrain": "pretrain_wan"}.get(phase)
    if family == constants.CustomModelFamilies.QWEN_IMAGE:
        return {"pretrain": "pretrain_qwen_image"}.get(phase)
    return None


class TrainDispatchTest(unittest.TestCase):
    def test_torch_trainer_resolves_only_selected_factory(self):
        cases = (
            ("GrootN1d6Trainer", "sft_embodied", "groot_n1_6"),
            ("GrootN1d7Trainer", "sft_embodied", "groot_n1_7"),
            ("LingBotFinetuneTrainer", "sft_lingbot_va", "lingbot_va"),
        )
        for trainer, method_name, model in cases:
            method = object()
            factory = SimpleNamespace(build_optimization=object())
            with self.subTest(trainer=trainer), \
                    patch.object(torch_registry.importlib, "import_module", side_effect=[method, factory]) as load:
                self.assertEqual(torch_registry.resolve_torch_trainer(trainer), (method, factory.build_optimization))
                self.assertEqual([call.args[0] for call in load.call_args_list], [
                    f"loongforge.training.torch.methods.{method_name}",
                    f"loongforge.extensions.torch.{model}.factory",
                ])

    def test_default_torch_trainer_and_invalid_selection(self):
        port = type("TorchOptimization", (), {})
        module = SimpleNamespace(TorchOptimization=port)
        with patch.dict(sys.modules, {"loongforge.engines.torch.train_step": module}), \
                patch.object(torch_registry.importlib, "import_module") as load:
            method, build = torch_registry.resolve_torch_trainer("FinetuneTrainer")
            self.assertIs(method, load.return_value)
            self.assertIsInstance(build(None), port)
            load.assert_called_once_with("loongforge.training.torch.methods.sft_embodied")
            load.reset_mock()
            for name in ("", "missing"):
                with self.assertRaises(ValueError):
                    torch_registry.resolve_torch_trainer(name)
            load.assert_not_called()

    def test_routes_registered_models_and_config_only_megatron(self):
        cases = (
            (["train.py", "--model-name", "qwen3-0.6b"], "loongforge.training.megatron.runner"),
            (["train.py", "--model-name", "pi05"], "loongforge.training.torch.runner"),
            (["train.py", "--model-name", "GROOT-N1-6"], "loongforge.training.torch.runner"),
            (["train.py", "--model-name", "pi05", "--config-file", "custom.yaml"],
             "loongforge.training.torch.runner"),
            (["train.py", "--config-file", "custom.yaml"], "loongforge.training.megatron.runner"),
        )
        for argv, entrypoint in cases:
            with (
                self.subTest(argv=argv),
                patch.object(sys, "argv", argv),
                patch.object(train, "import_module") as load,
            ):
                train.main()
                load.assert_called_once_with(entrypoint)

    def test_megatron_family_and_phase_resolve_to_previous_method(self):
        families = [
            *constants.LanguageModelFamilies.names(),
            *constants.VisionLanguageModelFamilies.names(),
            *constants.CustomModelFamilies.names(),
        ]
        for family in families:
            for phase in PHASES:
                expected = _previous_method(family, phase)
                with self.subTest(family=family, phase=phase), \
                        patch.object(registry.importlib, "import_module") as load:
                    if expected is None:
                        with self.assertRaises(ValueError):
                            registry.resolve_method(family, phase)
                        load.assert_not_called()
                    else:
                        registry.resolve_method(family, phase)
                        load.assert_called_once_with(f"loongforge.training.megatron.methods.{expected}")

    def test_split_model_config_reads_top_level_or_model_section(self):
        llm = constants.LanguageModelFamilies.names()[0]
        vlm = constants.VisionLanguageModelFamilies.names()[0]
        top = SimpleNamespace(model_type=llm)
        self.assertEqual(constants.split_model_config(top), (llm, top))
        nested = SimpleNamespace(model_type=vlm)
        self.assertEqual(constants.split_model_config(SimpleNamespace(model_type=vlm, model=nested)), (vlm, nested))
        self.assertEqual(constants.split_model_config(SimpleNamespace(model=nested)), (vlm, nested))
        with self.assertRaises(ValueError):
            constants.split_model_config(SimpleNamespace(model_type=vlm))

    def test_megatron_methods_train_through_pretrain(self):
        for module in sorted(set(registry.METHOD_REGISTRY.values())):
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

    def test_torch_catalog_model_types_exist_in_both_embodied_registries(self):
        model_modules = _literal_assignment(ROOT / "loongforge/models/embodied/registry.py", "MODEL_MODULES")
        data_modules = _literal_assignment(ROOT / "loongforge/data/embodied/registry.py", "MODEL_MODULES")
        for name, entry in catalog.MODEL_CONFIG_REGISTRY.items():
            if entry["engine"] != "torch":
                continue
            config_text = Path(catalog.get_config_path(name)).read_text(encoding="utf-8")
            match = re.search(r"^\s*model_type:\s*([^#\s]+)", config_text, re.MULTILINE)
            self.assertIsNotNone(match, name)
            model_type = match.group(1).strip("'\\\"")
            self.assertIn(model_type, model_modules, name)
            self.assertIn(model_type, data_modules, name)

    def test_config_types_are_loaded_only_for_selected_model(self):
        model_cls, data_cls = type("ModelConfig", (), {}), type("DataConfig", (), {})
        modules = [SimpleNamespace(Pi05ModelConfig=model_cls), SimpleNamespace(Pi05DataConfig=data_cls)]
        with patch.object(catalog, "import_module", side_effect=modules) as load:
            self.assertEqual(catalog.get_config_types("pi05"), (model_cls, data_cls))
            self.assertEqual([call.args[0] for call in load.call_args_list], [
                "loongforge.models.embodied.pi05.configuration_pi05",
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
