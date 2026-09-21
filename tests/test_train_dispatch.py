# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import loongforge.train as train
from loongforge.models import catalog

ROOT = Path(__file__).resolve().parents[1]


class TrainDispatchTest(unittest.TestCase):
    def test_routes_registered_models_and_config_only_mcore(self):
        cases = (
            (["train.py", "--model-name", "qwen3-0.6b"], "loongforge.engines.mcore.entrypoint"),
            (["train.py", "--model-name", "pi05"], "loongforge.engines.torch.entrypoint"),
            (["train.py", "--model-name", "GROOT-N1-6"], "loongforge.engines.torch.entrypoint"),
            (["train.py", "--model-name", "pi05", "--config-file", "custom.yaml"],
             "loongforge.engines.torch.entrypoint"),
            (["train.py", "--config-file", "custom.yaml"], "loongforge.engines.mcore.entrypoint"),
        )
        for argv, entrypoint in cases:
            with (
                self.subTest(argv=argv),
                patch.object(sys, "argv", argv),
                patch.object(train, "import_module") as load,
            ):
                train.main()
                load.assert_called_once_with(entrypoint)

    def test_rejects_unknown_and_ambiguous_models(self):
        with patch.object(sys, "argv", ["train.py", "--model-name", "missing"]), self.assertRaises(SystemExit):
            train.main()
        with (
            patch.object(sys, "argv", ["train.py", "--model-name", "same-name"]),
            patch.dict(catalog.MODEL_CONFIG_REGISTRY, {
                "same-name": {"engine": "mcore"}, "same_name": {"engine": "torch"},
            }),
            self.assertRaises(SystemExit),
        ):
            train.main()

    def test_default_paths_are_independent_of_working_directory(self):
        for name, relative in (
            ("qwen3-0.6b", "configs/models/qwen3/qwen3_0_6b.yaml"),
            ("pi05", "configs/models/embodied/pi05.yaml"),
            ("cosmos3-nano", "configs/models/embodied/cosmos3/nano.yaml"),
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
                self.assertIn(entry["engine"], {"mcore", "torch"})
                self.assertTrue(Path(catalog.get_config_path(name)).is_file())
                if entry["engine"] != "torch":
                    continue
                for field in ("model_config", "data_config"):
                    module, cls = entry[field].rsplit(".", 1)
                    tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text())
                    self.assertIn(cls, {node.name for node in tree.body if isinstance(node, ast.ClassDef)})

    def test_config_types_are_loaded_only_for_selected_model(self):
        model_cls, data_cls = type("ModelConfig", (), {}), type("DataConfig", (), {})
        modules = [SimpleNamespace(Pi05ModelConfig=model_cls), SimpleNamespace(Pi05DataConfig=data_cls)]
        with patch.object(catalog, "import_module", side_effect=modules) as load:
            self.assertEqual(catalog.get_config_types("pi05"), (model_cls, data_cls))
            self.assertEqual([call.args[0] for call in load.call_args_list], [
                "loongforge.models.embodied.pi05.model_configuration_pi05",
                "loongforge.data.embodied.datasets.pi05.transforms.data_configuration_pi05",
            ])
        with self.assertRaises(ValueError):
            catalog.get_config_types("qwen3-0.6b")

    def test_catalog_import_does_not_load_training_dependencies(self):
        subprocess.run([sys.executable, "-c", """
import sys
from loongforge.models.catalog import get_model_entry, get_config_path
import loongforge.data
assert get_model_entry('pi05')['engine'] == 'torch'
assert get_model_entry('QWEN3-0.6B')['engine'] == 'mcore'
assert get_config_path('qwen3-0.6b').endswith('/configs/models/qwen3/qwen3_0_6b.yaml')
assert not any(name.split('.')[0] in {'torch', 'megatron', 'transformers', 'omegaconf'} for name in sys.modules)
from types import ModuleType
template = ModuleType('loongforge.data.chat_template')
template.ChatTemplate = object()
sys.modules[template.__name__] = template
assert loongforge.data.ChatTemplate is template.ChatTemplate
"""], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
