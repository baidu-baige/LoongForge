# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check the layering of ``loongforge`` and the strings in the data registries.

Runs without torch: it only parses files.

Layering rules:
  * ``constants.py`` imports only the standard library.
  * ``chat_templates/`` imports no other loongforge layer, not even under ``TYPE_CHECKING``.
  * ``data/`` family directories do not import each other.
  * ``data/`` imports no ``models`` implementation, ``training``, ``evaluation`` or ``extensions``.
    It may import model-package declarative modules (config / schema / pure helpers); the
    blacklist below rejects ``modeling_*``, ``*_model``, ``*_layer_spec`` and kernel modules.
  * ``data/`` imports from ``engines`` only ``chunkpipe`` in ``llm_dataloader.py``.
  * Only ``training/`` imports ``training``.
  * ``models/`` imports no ``data`` at run time (``TYPE_CHECKING`` type contracts are allowed),
    and no ``training``, ``evaluation``, ``extensions`` or ``chat_templates``.
  * ``models/`` imports no ``engines`` at run time; the training method injects engine callables.
  * ``engines/`` imports no ``data`` at run time.
  * ``engines/`` imports only ``models/base``, ``models/multimodal`` and ``models/catalog``,
    never a model family.
  * ``training/megatron`` and ``training/torch`` do not import each other's training or engine modules.
  * ``extensions/`` may import ``models`` and ``engines`` but not ``training`` or ``data``;
    ``models`` and ``engines`` do not import ``extensions``.

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
    ("loongforge", "engines", "megatron", "distributed", "chunkpipe"): {"llm_dataloader.py"},
}
# ``engines/`` may import these generic model packages only.
ENGINE_MODEL_ALLOWED = (
    ("loongforge", "models", "base"),
    ("loongforge", "models", "catalog"),
    ("loongforge", "models", "multimodal"),
)
# ``MODEL_MODULES`` values that are plain modules, not model packages.
PLAIN_MODULE_MODEL_TYPES = {"dummy"}
# ``MODEL_MODULES`` keys that register a collator but no transform builder.
NO_TRANSFORM_MODEL_TYPES = {"dummy"}
EMBODIED_DECORATORS = ("register_collator", "register_transform_builder", "register_sampler_builder")


def _is_model_implementation(module_name):
    """True when a ``models/`` module name denotes an implementation, not a contract.

    Both naming conventions in the repository count: ``modeling_<model>.py`` on the
    HF side and ``<model>_model.py`` on the Megatron side (``qwen_model.py``,
    ``encoder_model.py``). Declarative modules (``configuration_*``, ``*_config``,
    ``processing_*``, ``features``, ``embodiment_configs``) stay importable.
    """
    return (
        module_name.startswith("modeling_")
        or module_name.endswith(("_model", "_layer_spec"))
        or module_name in {"model", "modeling"}
        or "kernel" in module_name
    )


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
    """Imports of ``path`` as absolute dotted tuples; ``from a import b`` gives ``a.b``."""
    package = list(path.relative_to(ROOT).with_suffix("").parts[:-1])
    visitor = _Imports(package)
    visitor.visit(_parse(path))
    return visitor.found


def _runtime_imports(path):
    return [target for target, type_checking, _ in _imports(path) if not type_checking]


@lru_cache(maxsize=1)
def _torch_engine_argument_names():
    """Annotated field names of every class in ``engines/torch/arguments.py``."""
    names = set()
    for node in ast.walk(_parse(PKG / "engines" / "torch" / "arguments.py")):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                names.add(item.target.id)
    return names


def _is_training_args(node):
    """True for ``training_args`` and ``<ctx>.training_args``."""
    return (isinstance(node, ast.Name) and node.id == "training_args") or (
        isinstance(node, ast.Attribute) and node.attr == "training_args"
    )


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


class LayeringTest(unittest.TestCase):
    def test_constants_import_only_stdlib(self):
        if not hasattr(sys, "stdlib_module_names"):
            self.skipTest("sys.stdlib_module_names requires Python 3.10+")
        path = PKG / "constants.py"
        for target in (t for t, _, _ in _imports(path)):
            self.assertIn(target[0], sys.stdlib_module_names, f"constants.py imports {'.'.join(target)}")

    def test_chat_templates_import_no_other_layer(self):
        # Stricter than every other rule: chat_templates is a leaf, so even a
        # TYPE_CHECKING import would couple it to a layer it must not know.
        banned = [
            ("loongforge", layer)
            for layer in ("data", "engines", "models", "training", "evaluation", "extensions")
        ]
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
        banned = [("loongforge", layer) for layer in ("training", "evaluation", "extensions")]
        for path in _py_files(DATA):
            # ``TYPE_CHECKING`` imports count too: data must not depend on the engine schema.
            for target, _, _ in _imports(path):
                for prefix in banned:
                    self.assertFalse(_starts_with(target, prefix), f"{_rel(path)}: {'.'.join(target)}")
                if not _starts_with(target, ("loongforge", "engines")):
                    continue
                allowed = any(
                    _starts_with(target, prefix) and (files is None or path.name in files)
                    for prefix, files in DATA_ENGINE_ALLOWED.items()
                )
                self.assertTrue(allowed, f"{_rel(path)} imports {'.'.join(target)}")

    def test_data_do_not_import_model_implementations(self):
        for path in _py_files(DATA):
            for target in _runtime_imports(path):
                if not _starts_with(target, ("loongforge", "models")):
                    continue
                offenders = [part for part in target if _is_model_implementation(part)]
                self.assertFalse(
                    offenders,
                    f"{_rel(path)} imports model implementation {'.'.join(target)}",
                )

    def test_data_reads_no_engine_arguments(self):
        """Data components get execution facts from ``DataRuntime``, not from backend args."""
        engine_names = _torch_engine_argument_names()
        found = set()
        for path in _py_files(DATA):
            for node in ast.walk(_parse(path)):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in engine_names
                    and _is_training_args(node.value)
                ):
                    found.add((_rel(path), node.lineno, node.attr))
        self.assertEqual(found, set(), f"data reads Torch engine arguments: {sorted(found)}")

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

    def test_only_training_imports_training(self):
        for layer in ("engines", "models", "evaluation", "chat_templates", "extensions", "data"):
            for path in _py_files(PKG / layer):
                for target in _runtime_imports(path):
                    self.assertFalse(
                        _starts_with(target, ("loongforge", "training")),
                        f"{_rel(path)} imports {'.'.join(target)}",
                    )

    def test_models_do_not_import_engine_global_readers(self):
        for path in _py_files(PKG / "models"):
            for target in _runtime_imports(path):
                from_engine = _starts_with(target, ("loongforge", "engines")) or _starts_with(
                    target, ("megatron", "training")
                )
                self.assertFalse(
                    from_engine and target[-1] in GLOBAL_READERS, f"{_rel(path)} imports {'.'.join(target)}"
                )

    def test_models_do_not_import_data(self):
        for path in _py_files(PKG / "models"):
            for target in _runtime_imports(path):
                self.assertFalse(
                    _starts_with(target, ("loongforge", "data")), f"{_rel(path)} imports {'.'.join(target)}"
                )

    def test_models_do_not_import_other_layers(self):
        banned = [("loongforge", layer) for layer in ("training", "evaluation", "extensions", "chat_templates")]
        for path in _py_files(PKG / "models"):
            for target in _runtime_imports(path):
                for prefix in banned:
                    self.assertFalse(_starts_with(target, prefix), f"{_rel(path)} imports {'.'.join(target)}")

    def test_models_do_not_import_engines(self):
        found = set()
        for path in _py_files(PKG / "models"):
            rel = str(path.relative_to(PKG / "models"))
            # ``TYPE_CHECKING`` imports count too: models receive engine state as explicit arguments.
            for target, _, _ in _imports(path):
                if _starts_with(target, ("loongforge", "engines")):
                    found.add((rel, ".".join(target)))
        self.assertEqual(found, set(), "models must not import engines")

    def test_type_checking_imports_stay_type_only(self):
        # The three files below carry model/data type contracts under TYPE_CHECKING.
        # Guard the visitor so they never get reported as runtime imports.
        contracts = {
            "models/embodied/groot_n1_6/modeling_groot_n1_6.py",
            "models/embodied/groot_n1_7/modeling_groot_n1_7.py",
            "models/embodied/cosmos3/modeling_cosmos3.py",
        }
        for rel in contracts:
            type_only = [
                target
                for target, type_checking, _ in _imports(PKG / rel)
                if type_checking and _starts_with(target, ("loongforge", "data"))
            ]
            self.assertTrue(type_only, f"{rel}: expected a TYPE_CHECKING data contract")

    def test_engines_import_no_data_at_run_time(self):
        for path in _py_files(PKG / "engines"):
            for target in _runtime_imports(path):
                self.assertFalse(
                    _starts_with(target, ("loongforge", "data")), f"{_rel(path)} imports {'.'.join(target)}"
                )

    def test_engines_import_only_generic_model_packages(self):
        for path in _py_files(PKG / "engines"):
            for target in _runtime_imports(path):
                if not _starts_with(target, ("loongforge", "models")):
                    continue
                allowed = any(_starts_with(target, prefix) for prefix in ENGINE_MODEL_ALLOWED)
                self.assertTrue(allowed, f"{_rel(path)} imports {'.'.join(target)}")

    def test_training_backends_are_isolated(self):
        for backend, other in (("megatron", "torch"), ("torch", "megatron")):
            for path in _py_files(PKG / "training" / backend):
                for target in _runtime_imports(path):
                    self.assertFalse(
                        _starts_with(target, ("loongforge", "training", other))
                        or _starts_with(target, ("loongforge", "engines", other)),
                        f"{_rel(path)} imports {'.'.join(target)}",
                    )

    def test_models_and_engines_do_not_import_extensions(self):
        for layer in ("models", "engines"):
            for path in _py_files(PKG / layer):
                for target in _runtime_imports(path):
                    self.assertFalse(
                        _starts_with(target, ("loongforge", "extensions")),
                        f"{_rel(path)} imports {'.'.join(target)}",
                    )

    def test_extensions_do_not_import_training_or_data(self):
        for path in _py_files(PKG / "extensions"):
            for target in _runtime_imports(path):
                for layer in ("training", "data"):
                    self.assertFalse(
                        _starts_with(target, ("loongforge", layer)),
                        f"{_rel(path)} imports {'.'.join(target)}",
                    )

    def test_evaluation_does_not_import_training_or_extensions(self):
        for path in _py_files(PKG / "evaluation"):
            for target in _runtime_imports(path):
                self.assertFalse(
                    _starts_with(target, ("loongforge", "training"))
                    or _starts_with(target, ("loongforge", "extensions")),
                    f"{_rel(path)} imports {'.'.join(target)}",
                )


class RegistryStringsTest(unittest.TestCase):
    def test_embodied_model_type_keys_match_between_layers(self):
        models = set(_literal_assignment(PKG / "models" / "embodied" / "registry.py", "MODEL_MODULES"))
        data = set(_literal_assignment(EMBODIED / "registry.py", "MODEL_MODULES"))
        self.assertEqual(models, data - PLAIN_MODULE_MODEL_TYPES)

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

    def test_embodied_dataset_formats(self):
        formats = _literal_assignment(EMBODIED / "registry.py", "DATASET_FORMATS")
        for dataset_format, entry in formats.items():
            module, func = entry.split(":")
            target = EMBODIED.joinpath(*module.split(".")).with_suffix(".py")
            self.assertTrue(target.is_file(), f"DATASET_FORMATS[{dataset_format!r}]: {module}")
            self.assertIn(func, _top_level_names(target), f"DATASET_FORMATS[{dataset_format!r}]: {func}")

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
