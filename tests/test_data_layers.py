# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check the layering of ``loongforge`` and the strings in the data registries.

Runs without torch: it only parses files.

Layering rules:
  * ``constants.py`` imports only the standard library.
  * ``chat_templates/`` imports no other loongforge layer at run time.
  * ``data/`` family directories do not import each other.
  * ``data/`` imports no ``training``, and from ``engines`` only the tokenizer
    (plus ``chunkpipe`` in ``llm_dataloader.py``).
  * ``data/`` and ``chat_templates/`` do not read engine global state.
  * ``models/`` does not import ``data`` at module level (``TYPE_CHECKING`` and function bodies are allowed).
  * ``engines/`` imports no ``data`` at run time (only ``TYPE_CHECKING`` is allowed).

Layout rules: no ``utils/`` directory, fixed ``data/`` first level, fixed role
files in every ``data/embodied/<model>/``.

Registry strings must point at real code:
  * ``embodied/registry.py``: ``MODEL_MODULES`` (decorator names, package imports) and ``DATASET_STRATEGIES``
  * ``vlm/__init__.py``: ``TASK_ENCODER_REGISTRY``
  * ``chat_templates/registry.py``: every ``.jinja`` it reads
"""

import ast
from functools import lru_cache
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "loongforge"
DATA = PKG / "data"
EMBODIED = DATA / "embodied"
CHAT_TEMPLATES = PKG / "chat_templates"
FAMILIES = ("llm", "vlm", "embodied", "diffusion")
GLOBAL_READERS = {"get_args", "get_tokenizer", "get_model_config", "get_chat_template"}
# Allowed ``loongforge.engines`` imports from ``data/``: prefix -> files that may use it (None: any).
DATA_ENGINE_ALLOWED = {
    ("loongforge", "engines", "mcore", "tokenizer"): None,
    ("loongforge", "engines", "mcore", "parallel", "chunkpipe"): {"llm_dataloader.py"},
}
# ``MODEL_MODULES`` values that are plain modules, not model packages.
PLAIN_MODULE_MODEL_TYPES = {"dummy"}
NON_MODEL_DIRS = {"datasets", "transforms"}
# ``MODEL_MODULES`` keys that register a collator but no transform builder.
NO_TRANSFORM_MODEL_TYPES = {"dummy"}
EMBODIED_DECORATORS = ("register_collator", "register_transform_builder", "register_sampler_builder")


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
        saved, self.in_function = self.in_function, True
        self.generic_visit(node)
        self.in_function = saved

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _visit_function


@lru_cache(maxsize=None)
def _imports(path):
    """Imports of ``path`` as absolute dotted tuples; ``from a import b`` gives ``a.b``."""
    package = list(path.relative_to(ROOT).with_suffix("").parts[:-1])
    visitor = _Imports(package)
    visitor.visit(_parse(path))
    return visitor.found


def _runtime_imports(path):
    return [target for target, type_checking, _ in _imports(path) if not type_checking]


def _module_level_imports(path):
    return [target for target, type_checking, in_function in _imports(path) if not (type_checking or in_function)]


def _starts_with(target, prefix):
    return target[: len(prefix)] == tuple(prefix)


def _module_file(dotted, root):
    base = root.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _top_level_names(path):
    names = set()
    for node in _parse(path).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _literal_assignment(path, name):
    for node in _parse(path).body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def _decorator_args(path, decorator):
    """Return the strings passed to ``@<decorator>(...)`` in ``path``."""
    return {
        node.args[0].value
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == decorator
    }


def _only_docstring(path):
    body = _parse(path).body
    return len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)


class LayeringTest(unittest.TestCase):
    def test_constants_import_only_stdlib(self):
        path = PKG / "constants.py"
        for target in (t for t, _, _ in _imports(path)):
            self.assertIn(target[0], sys.stdlib_module_names, f"constants.py imports {'.'.join(target)}")

    def test_chat_templates_import_no_other_layer(self):
        banned = [("loongforge", layer) for layer in ("data", "engines", "models", "training")]
        for path in _py_files(CHAT_TEMPLATES):
            for target in _runtime_imports(path):
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
        for path in _py_files(DATA):
            for target in _runtime_imports(path):
                self.assertFalse(_starts_with(target, ("loongforge", "training")), f"{_rel(path)}: {target}")
                if not _starts_with(target, ("loongforge", "engines")):
                    continue
                allowed = any(
                    _starts_with(target, prefix) and (files is None or path.name in files)
                    for prefix, files in DATA_ENGINE_ALLOWED.items()
                )
                self.assertTrue(allowed, f"{_rel(path)} imports {'.'.join(target)}")

    def test_data_and_chat_templates_do_not_read_global_state(self):
        for path in _py_files(DATA) + _py_files(CHAT_TEMPLATES):
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

    def test_models_do_not_import_data_at_module_level(self):
        for path in _py_files(PKG / "models"):
            for target in _module_level_imports(path):
                self.assertFalse(
                    _starts_with(target, ("loongforge", "data")), f"{_rel(path)} imports {'.'.join(target)}"
                )

    def test_engines_import_no_data_at_run_time(self):
        for path in _py_files(PKG / "engines"):
            for target in _runtime_imports(path):
                self.assertFalse(
                    _starts_with(target, ("loongforge", "data")), f"{_rel(path)} imports {'.'.join(target)}"
                )


class LayoutTest(unittest.TestCase):
    def test_no_utils_directory(self):
        for root in (DATA, CHAT_TEMPLATES, PKG / "engines"):
            found = [p for p in root.rglob("utils") if p.is_dir()]
            self.assertEqual(found, [], [_rel(p) for p in found])

    def test_data_first_level(self):
        dirs = {p.name for p in DATA.iterdir() if p.is_dir() and p.name != "__pycache__"}
        files = {p.name for p in DATA.iterdir() if p.is_file()}
        self.assertEqual(dirs, set(FAMILIES))
        self.assertEqual(files, {"__init__.py"} | {f"{f}_dataloader.py" for f in FAMILIES})
        self.assertTrue(_only_docstring(DATA / "__init__.py"))
        self.assertTrue(_only_docstring(CHAT_TEMPLATES / "__init__.py"))

    def test_embodied_model_directories(self):
        modules = _literal_assignment(EMBODIED / "registry.py", "MODEL_MODULES")
        packages = [d for m, d in modules.items() if m not in PLAIN_MODULE_MODEL_TYPES]
        for model_type, directory in modules.items():
            if model_type in PLAIN_MODULE_MODEL_TYPES:
                self.assertTrue((EMBODIED / f"{directory}.py").is_file(), f"MODEL_MODULES[{model_type!r}]")
            else:
                self.assertTrue((EMBODIED / directory).is_dir(), f"MODEL_MODULES[{model_type!r}]: {directory}")
        model_dirs = {
            p.name for p in EMBODIED.iterdir() if p.is_dir() and p.name not in NON_MODEL_DIRS | {"__pycache__"}
        }
        self.assertEqual(sorted(packages), sorted(model_dirs))
        for d in model_dirs:
            for name in ("__init__.py", f"data_configuration_{d}.py", f"{d}_transform.py", f"{d}_collator.py"):
                self.assertTrue((EMBODIED / d / name).is_file(), f"embodied/{d}/{name}")


class RegistryStringsTest(unittest.TestCase):
    def test_embodied_model_modules_register_their_model_type(self):
        modules = _literal_assignment(EMBODIED / "registry.py", "MODEL_MODULES")
        for model_type, directory in modules.items():
            plain = model_type in PLAIN_MODULE_MODEL_TYPES
            files = [EMBODIED / f"{directory}.py"] if plain else _py_files(EMBODIED / directory)
            registered = {p: {d: _decorator_args(p, d) for d in EMBODIED_DECORATORS} for p in files}
            required = ["register_collator"]
            if model_type not in NO_TRANSFORM_MODEL_TYPES:
                required.append("register_transform_builder")
            for decorator in required:
                names = set().union(*(r[decorator] for r in registered.values()))
                self.assertIn(model_type, names, f"MODEL_MODULES[{model_type!r}]: no @{decorator}({model_type!r})")
            if plain:
                continue
            imported = _module_level_imports(EMBODIED / directory / "__init__.py")
            for path, by_decorator in registered.items():
                if not any(by_decorator.values()):
                    continue
                module = tuple(path.relative_to(ROOT).with_suffix("").parts)
                self.assertTrue(
                    any(_starts_with(target, module) for target in imported),
                    f"embodied/{directory}/__init__.py does not import {_rel(path)}",
                )

    def test_embodied_dataset_strategies(self):
        strategies = _literal_assignment(EMBODIED / "registry.py", "DATASET_STRATEGIES")
        for strategy, entry in strategies.items():
            module, func = entry.split(":")
            target = EMBODIED.joinpath(*module.split(".")).with_suffix(".py")
            self.assertTrue(target.is_file(), f"DATASET_STRATEGIES[{strategy!r}]: {module}")
            self.assertIn(func, _top_level_names(target), f"DATASET_STRATEGIES[{strategy!r}]: {func}")

    def test_vlm_task_encoder_registry_strings(self):
        registry = _literal_assignment(DATA / "vlm" / "__init__.py", "TASK_ENCODER_REGISTRY")
        for key, dotted in registry.items():
            module, cls = dotted.rsplit(".", 1)
            target = _module_file(module, ROOT)
            self.assertIsNotNone(target, f"TASK_ENCODER_REGISTRY[{key!r}]: {module}")
            self.assertIn(cls, _top_level_names(target), f"TASK_ENCODER_REGISTRY[{key!r}]: {cls}")

    def test_chat_template_jinja_files_exist(self):
        source = (CHAT_TEMPLATES / "registry.py").read_text(encoding="utf-8")
        names = set(re.findall(r'_read_builtin_chat_template\("([^"]+)"\)', source))
        self.assertTrue(names)
        for name in names:
            self.assertTrue((CHAT_TEMPLATES / "jinja" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
