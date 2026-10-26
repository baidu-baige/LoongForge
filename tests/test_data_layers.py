# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check the dependency direction between the four layers of ``loongforge``.

Runs without torch: it only parses files. The rules mirror ``AGENTS.md``:
``data`` owns samples and data formats, ``models`` network structure,
``engines`` backend execution, and ``training`` what a run does.
"""

import ast
from functools import lru_cache
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "loongforge"
DATA = PKG / "data"
CHAT_TEMPLATES = DATA / "chat_templates"
FAMILIES = ("llm", "vlm", "embodied", "diffusion")
GLOBAL_READERS = {"get_args", "get_tokenizer", "get_model_config", "get_chat_template"}
# ``data/`` may import these engine modules only.
DATA_ENGINE_ALLOWED = {
    ("loongforge", "engines", "megatron", "distributed", "chunkpipe"): {"loongforge/data/llm/dataloader.py"},
}
# Declarative ``data/`` packages that ``engines/`` may read.
ENGINE_DATA_ALLOWED = (("loongforge", "data", "chat_templates"),)
# Backend state that ``data/`` must receive as arguments instead of reading it.
DATA_BACKEND_STATE_BANNED = (
    ("megatron", "core", "parallel_state"),
    ("megatron", "core", "mpu"),
    ("megatron", "training", "checkpointing"),
)
# Torch argument groups that only the backend consumes; ``data/`` must not read them.
TORCH_BACKEND_MIXINS = (
    "_StepArgs",
    "_RuntimeArgs",
    "_CheckpointIOArgs",
    "_OptimizerArgs",
    "_LoraArgs",
    "_CudaGraphArgs",
    "_ActivationCheckpointArgs",
    "_FP8Args",
    "_DistributedArgs",
)
# ``engines/`` may import these generic model packages only.
ENGINE_MODEL_ALLOWED = (
    ("loongforge", "models", "base"),
    ("loongforge", "models", "catalog"),
    ("loongforge", "models", "multimodal"),
)


def _is_model_implementation(module_name):
    """True when a ``models/`` module denotes an implementation, not a contract."""
    return (
        module_name.startswith("modeling_")
        or module_name.endswith(("_model", "_layer_spec"))
        or module_name in {"model", "modeling"}
        or "kernel" in module_name
    )


def _package_segments(target):
    segments = []
    for index in range(1, len(target)):
        if (PKG.joinpath(*target[1:index + 1])).is_dir():
            segments.append(index)
    return segments


def _rel(path):
    return str(path.relative_to(ROOT))


def _py_files(root):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


@lru_cache(maxsize=None)
def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _is_type_checking(test):
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


class _Imports(ast.NodeVisitor):
    """Collect ``(dotted target, in_type_checking, in_function)`` for every import."""

    def __init__(self, package):
        self.package = package
        self.found = []
        self.type_checking = False
        self.in_function = False

    def _add(self, target):
        self.found.append((tuple(target), self.type_checking, self.in_function))

    def visit_Import(self, node):
        for alias in node.names:
            self._add(alias.name.split("."))

    def visit_ImportFrom(self, node):
        level = node.level
        base = self.package[: len(self.package) - (level - 1)] if level else []
        module = node.module.split(".") if node.module else []
        for alias in node.names:
            self._add(base + module + [alias.name])

    def visit_If(self, node):
        if _is_type_checking(node.test):
            saved, self.type_checking = self.type_checking, True
            for child in node.body:
                self.visit(child)
            self.type_checking = saved
            for child in node.orelse:
                self.visit(child)
        else:
            self.generic_visit(node)

    def _visit_function(self, node):
        saved = self.in_function, self.type_checking
        self.in_function, self.type_checking = True, False
        self.generic_visit(node)
        self.in_function, self.type_checking = saved

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _visit_function


@lru_cache(maxsize=None)
def _imports(path):
    package = list(path.relative_to(ROOT).with_suffix("").parts[:-1])
    visitor = _Imports(package)
    visitor.visit(_parse(path))
    return visitor.found


def _runtime_imports(path):
    return [target for target, type_checking, _ in _imports(path) if not type_checking]


@lru_cache(maxsize=1)
def _torch_backend_argument_names():
    """Field names of the Torch argument groups that the backend alone consumes."""
    names = set()
    for node in ast.walk(_parse(PKG / "training" / "torch_args.py")):
        if isinstance(node, ast.ClassDef) and node.name in TORCH_BACKEND_MIXINS:
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    names.add(item.target.id)
    return names


def _is_training_args(node):
    return (isinstance(node, ast.Name) and node.id == "training_args") or (
        isinstance(node, ast.Attribute) and node.attr == "training_args"
    )


def _starts_with(target, prefix):
    return target[: len(prefix)] == tuple(prefix)


class LayeringTest(unittest.TestCase):
    def test_chat_templates_import_no_other_layer(self):
        banned = [
            ("loongforge", layer)
            for layer in ("engines", "models", "training", "evaluation")
        ] + [("loongforge", "data", family) for family in FAMILIES]
        for path in _py_files(CHAT_TEMPLATES):
            for target, _, _ in _imports(path):
                for prefix in banned:
                    self.assertFalse(_starts_with(target, prefix), f"{_rel(path)} imports {'.'.join(target)}")

    def test_families_do_not_import_each_other(self):
        for family in FAMILIES:
            others = {("loongforge", "data", f) for f in FAMILIES if f != family}
            for path in _py_files(DATA / family):
                for target, _, _ in _imports(path):
                    for prefix in others:
                        self.assertFalse(_starts_with(target, prefix), f"{_rel(path)} imports {'.'.join(target)}")

    def test_data_imports_no_training_and_limited_engines(self):
        banned = [("loongforge", layer) for layer in ("training", "evaluation")]
        for path in _py_files(DATA):
            for target, _, _ in _imports(path):
                for prefix in banned:
                    self.assertFalse(_starts_with(target, prefix), f"{_rel(path)}: {'.'.join(target)}")
                if not _starts_with(target, ("loongforge", "engines")):
                    continue
                allowed = any(
                    _starts_with(target, prefix) and (files is None or _rel(path) in files)
                    for prefix, files in DATA_ENGINE_ALLOWED.items()
                )
                self.assertTrue(allowed, f"{_rel(path)} imports {'.'.join(target)}")

    def test_data_do_not_import_model_implementations(self):
        for path in _py_files(DATA):
            for target in _runtime_imports(path):
                if not _starts_with(target, ("loongforge", "models")):
                    continue
                package = set(_package_segments(target))
                offenders = [
                    part for index, part in enumerate(target)
                    if index not in package and _is_model_implementation(part)
                ]
                self.assertFalse(
                    offenders,
                    f"{_rel(path)} imports model implementation {'.'.join(target)}",
                )

    def test_data_reads_no_backend_arguments(self):
        backend_names = _torch_backend_argument_names()
        self.assertTrue(backend_names, "Torch backend argument names were not found")
        found = set()
        for path in _py_files(DATA):
            for node in ast.walk(_parse(path)):
                if isinstance(node, ast.Attribute) and node.attr in backend_names and _is_training_args(node.value):
                    found.add((_rel(path), node.lineno, node.attr))
        self.assertEqual(found, set(), f"data reads Torch backend arguments: {sorted(found)}")

    def test_data_holds_no_backend_state(self):
        found = set()
        dist_queries = {"get_rank", "get_world_size", "is_initialized"}
        for path in _py_files(DATA):
            for target, _, _ in _imports(path):
                if any(_starts_with(target, prefix) for prefix in DATA_BACKEND_STATE_BANNED):
                    found.add((_rel(path), ".".join(target)))
            for node in ast.walk(_parse(path)):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in dist_queries
                    and (
                        (isinstance(node.value, ast.Name) and node.value.id == "dist")
                        or (
                            isinstance(node.value, ast.Attribute)
                            and node.value.attr == "distributed"
                        )
                    )
                ):
                    found.add((_rel(path), f"torch.distributed.{node.attr}"))
        self.assertEqual(found, set(), f"data reads backend state: {sorted(found)}")

    def test_data_components_take_explicit_configs(self):
        # ``image_enhance.py`` takes an augmentation-parameter dict named ``args``.
        allowed = {"loongforge/data/vlm/ernie/utils/image_enhance.py"}
        offenders = set()
        for path in _py_files(DATA):
            if _rel(path) in allowed:
                continue
            for node in ast.walk(_parse(path)):
                if isinstance(node, ast.FunctionDef) and any(arg.arg == "args" for arg in node.args.args):
                    offenders.add(f"{_rel(path)}:{node.lineno}:{node.name}")
        self.assertEqual(offenders, set(), f"data components take the args namespace: {sorted(offenders)}")

    def test_data_does_not_read_global_state(self):
        for path in _py_files(DATA):
            for node in ast.walk(_parse(path)):
                if isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                elif isinstance(node, ast.alias):
                    name = node.name.split(".")[-1]
                else:
                    continue
                self.assertNotIn(name, GLOBAL_READERS, f"{_rel(path)}:{getattr(node, 'lineno', '?')}")

    def test_only_training_imports_training(self):
        for layer in ("engines", "models", "evaluation", "data"):
            for path in _py_files(PKG / layer):
                for target in _runtime_imports(path):
                    self.assertFalse(
                        _starts_with(target, ("loongforge", "training")),
                        f"{_rel(path)} imports {'.'.join(target)}",
                    )

    def test_models_do_not_import_data(self):
        for path in _py_files(PKG / "models"):
            for target in _runtime_imports(path):
                self.assertFalse(
                    _starts_with(target, ("loongforge", "data")), f"{_rel(path)} imports {'.'.join(target)}"
                )

    def test_models_do_not_import_other_layers(self):
        banned = [("loongforge", layer) for layer in ("training", "evaluation")]
        for path in _py_files(PKG / "models"):
            for target in _runtime_imports(path):
                for prefix in banned:
                    self.assertFalse(_starts_with(target, prefix), f"{_rel(path)} imports {'.'.join(target)}")

    def test_models_do_not_import_engines(self):
        found = set()
        for path in _py_files(PKG / "models"):
            rel = str(path.relative_to(PKG / "models"))
            for target, _, _ in _imports(path):
                if _starts_with(target, ("loongforge", "engines")):
                    found.add((rel, ".".join(target)))
        self.assertEqual(found, set(), "models must not import engines")

    def test_engines_import_only_declarative_data(self):
        for path in _py_files(PKG / "engines"):
            for target in _runtime_imports(path):
                if not _starts_with(target, ("loongforge", "data")):
                    continue
                allowed = any(_starts_with(target, prefix) for prefix in ENGINE_DATA_ALLOWED)
                self.assertTrue(allowed, f"{_rel(path)} imports {'.'.join(target)}")

    def test_engines_import_only_generic_model_packages(self):
        for path in _py_files(PKG / "engines"):
            for target in _runtime_imports(path):
                if not _starts_with(target, ("loongforge", "models")):
                    continue
                allowed = any(_starts_with(target, prefix) for prefix in ENGINE_MODEL_ALLOWED)
                self.assertTrue(allowed, f"{_rel(path)} imports {'.'.join(target)}")

    def test_training_families_are_isolated(self):
        for family in FAMILIES:
            others = {("loongforge", "training", other) for other in FAMILIES if other != family}
            for path in _py_files(PKG / "training" / family):
                for target in _runtime_imports(path):
                    for prefix in others:
                        self.assertFalse(
                            _starts_with(target, prefix),
                            f"{_rel(path)} imports {'.'.join(target)}",
                        )

    def test_models_and_engines_do_not_import_optimizations(self):
        banned = (("loongforge", "training", "embodied", "optimizations"),)
        for layer in ("models", "engines"):
            for path in _py_files(PKG / layer):
                for target in _runtime_imports(path):
                    for prefix in banned:
                        self.assertFalse(
                            _starts_with(target, prefix),
                            f"{_rel(path)} imports {'.'.join(target)}",
                        )

    def test_embodied_optimizations_do_not_import_other_families(self):
        for path in _py_files(PKG / "training" / "embodied" / "optimizations"):
            for target in _runtime_imports(path):
                for family in ("llm", "vlm", "diffusion"):
                    self.assertFalse(
                        _starts_with(target, ("loongforge", "training", family)),
                        f"{_rel(path)} imports {'.'.join(target)}",
                    )

    def test_evaluation_does_not_import_training(self):
        for path in _py_files(PKG / "evaluation"):
            for target in _runtime_imports(path):
                self.assertFalse(
                    _starts_with(target, ("loongforge", "training")),
                    f"{_rel(path)} imports {'.'.join(target)}",
                )


class EmbodiedDataRegistryTest(unittest.TestCase):
    def test_data_registry_can_retry_after_import_failure(self):
        # An optional dependency may be installed later in the same process, so a
        # failed import must not mark the model type as done.
        sys.path.insert(0, str(ROOT))
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
