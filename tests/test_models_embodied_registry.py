# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check ``MODEL_MODULES`` in ``models/embodied/registry.py`` against the code.

Runs without torch: it only parses files.
"""

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

EMBODIED = Path(__file__).resolve().parents[1] / "loongforge" / "models" / "embodied"


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _model_modules():
    for node in _parse(EMBODIED / "registry.py").body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "MODEL_MODULES":
            return ast.literal_eval(node.value)
    raise AssertionError("MODEL_MODULES not found")


def _registered_types(path):
    """Return the model_type strings passed to ``@register_model(...)`` in ``path``."""
    return {
        node.args[0].value
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "register_model"
    }


class ModelsEmbodiedRegistryTest(unittest.TestCase):
    def test_table_modules_register_their_model_type(self):
        for model_type, module in _model_modules().items():
            path = EMBODIED / (module.replace(".", "/") + ".py")
            self.assertTrue(path.is_file(), f"MODEL_MODULES[{model_type!r}]: {module}")
            self.assertIn(model_type, _registered_types(path), f"MODEL_MODULES[{model_type!r}]")

    def test_every_registered_model_type_is_in_table(self):
        table = _model_modules()
        for path in EMBODIED.rglob("*.py"):
            for model_type in _registered_types(path):
                self.assertIn(model_type, table, f"{path.relative_to(EMBODIED)} registers {model_type!r}")

    def test_data_registry_can_retry_after_import_failure(self):
        import loongforge.data.embodied.registry as data_registry

        model_type = "pi05"
        previous_imported = set(data_registry._IMPORTED)
        data_registry._IMPORTED.discard(model_type)
        try:
            with patch.object(
                data_registry.importlib,
                "import_module",
                side_effect=[ModuleNotFoundError("optional dependency"), None],
            ) as import_module:
                with self.assertRaises(ModuleNotFoundError):
                    data_registry._ensure_imported(model_type)
                self.assertNotIn(model_type, data_registry._IMPORTED)
                data_registry._ensure_imported(model_type)
                self.assertEqual(import_module.call_count, 2)
                self.assertIn(model_type, data_registry._IMPORTED)
        finally:
            data_registry._IMPORTED.clear()
            data_registry._IMPORTED.update(previous_imported)


if __name__ == "__main__":
    unittest.main()
