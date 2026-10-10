# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check ``MODEL_MODULES`` in ``models/embodied/registry.py`` against the code.

Runs without torch: it only parses files.
"""

import ast
from pathlib import Path
import unittest

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

    def test_pi05_sets_dynamo_cache_limit(self):
        # Per-layer compile needs one cache entry per layer; the torch default is 8.
        target = "torch._dynamo.config.cache_size_limit"
        values = [
            node.value.value
            for node in _parse(EMBODIED / "pi05" / "modeling_pi05.py").body
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == target
            and isinstance(node.value, ast.Constant)
        ]
        self.assertTrue(values and values[-1] >= 512, f"pi05 must set {target} >= 512")


if __name__ == "__main__":
    unittest.main()
