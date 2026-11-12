# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for embodied data module discovery."""

import ast
import importlib
import pkgutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "loongforge.data.embodied"


def _load_registry(name, relative_path, functions, **state):
    path = ROOT / "loongforge/data/embodied" / relative_path
    nodes = [
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in functions
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    module = ModuleType(name)
    module.__dict__.update(importlib=importlib, pkgutil=pkgutil, Path=Path, **state)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


class TestEmbodiedDataDiscovery(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data_dir = Path(temporary.name) / "loongforge/data/embodied"
        modules = {}
        for name in ("loongforge", "loongforge.data", PACKAGE, f"{PACKAGE}.transforms"):
            directory = Path(temporary.name) / name.replace(".", "/")
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "__init__.py").touch()
            module = ModuleType(name)
            module.__path__ = [str(directory)]
            module.__file__ = str(directory / "__init__.py")
            modules[name] = module

        self.transform = _load_registry(
            f"{PACKAGE}.transforms.registry", "transforms/registry.py",
            {"register_transform_builder", "get_transform_builder", "discover_transform_builders"},
            __file__=str(self.data_dir / "transforms/registry.py"),
            _TRANSFORM_BUILDER_REGISTRY={}, _DISCOVERED_TRANSFORM_BUILDERS=False,
        )
        self.collator = _load_registry(
            f"{PACKAGE}.transforms.collator", "transforms/collator.py",
            {"register_preprocessor", "get_preprocessor", "discover_preprocessors"},
            __file__=str(self.data_dir / "transforms/collator.py"),
            _PREPROCESSOR_REGISTRY={}, _DISCOVERED_PREPROCESSORS=False,
        )
        self.default_sampler = object()
        self.sampler = _load_registry(
            f"{PACKAGE}.sampler_builder", "sampler_builder.py",
            {"register_sampler_builder", "get_sampler_builder", "discover_sampler_builders"},
            __file__=str(self.data_dir / "sampler_builder.py"),
            _SAMPLER_BUILDER_REGISTRY={}, _DISCOVERED_SAMPLER_BUILDERS=False,
            default_sampler_builder=self.default_sampler,
        )
        for module in (self.transform, self.collator, self.sampler):
            modules[module.__name__] = module
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)

        model = self.data_dir / "discovery_fixture"
        (model / "transforms").mkdir(parents=True)
        (model / "__init__.py").touch()
        (model / "transforms/__init__.py").write_text(
            f"from {PACKAGE}.transforms.registry import register_transform_builder\n"
            f"from {PACKAGE}.transforms.collator import register_preprocessor\n"
            '@register_transform_builder("discovery_fixture")\n'
            "def build_transform(context):\n"
            "    return []\n"
            '@register_preprocessor("discovery_fixture")\n'
            "class FixturePreprocessor:\n"
            "    pass\n"
        )
        sampler_source = (
            f"from {PACKAGE}.sampler_builder import register_sampler_builder\n"
            '@register_sampler_builder("{model_type}")\n'
            "def build_sampler(context):\n"
            "    return context\n"
        )
        (model / "samplers.py").write_text(sampler_source.format(model_type="discovery_fixture"))
        inline = self.data_dir / "inline_fixture"
        inline.mkdir()
        (inline / "__init__.py").write_text(sampler_source.format(model_type="inline_fixture"))
        ignored = self.data_dir / "_ignored"
        ignored.mkdir()
        (ignored / "__init__.py").write_text("raise AssertionError('private package imported')\n")
        (self.data_dir / "plain_module.py").write_text("raise AssertionError('plain module imported')\n")
        importlib.invalidate_caches()

    def test_unlisted_packages_register_without_entry_imports(self):
        builder = self.transform.get_transform_builder("discovery_fixture")
        self.assertEqual(builder(object()), [])
        self.assertEqual(
            self.collator.get_preprocessor("discovery_fixture").__name__,
            "FixturePreprocessor",
        )
        context = object()
        for model_type in ("discovery_fixture", "inline_fixture"):
            with self.subTest(model_type=model_type):
                self.assertIs(self.sampler.get_sampler_builder(model_type)(context), context)
        self.assertIs(self.sampler.get_sampler_builder("unknown"), self.default_sampler)

    def test_each_registry_discovers_once(self):
        for discover in (
            self.transform.discover_transform_builders,
            self.collator.discover_preprocessors,
            self.sampler.discover_sampler_builders,
        ):
            with self.subTest(discover=discover.__name__):
                with patch.object(importlib, "import_module", wraps=importlib.import_module) as load:
                    discover()
                    first_count = load.call_count
                    discover()
                    self.assertGreater(first_count, 0)
                    self.assertEqual(load.call_count, first_count)

    def test_discovery_preserves_import_errors(self):
        broken = self.data_dir / "broken_fixture"
        (broken / "transforms").mkdir(parents=True)
        (broken / "__init__.py").write_text("raise ImportError('fixture dependency missing')\n")
        (broken / "transforms/__init__.py").touch()
        importlib.invalidate_caches()
        for lookup in (
            self.transform.get_transform_builder,
            self.collator.get_preprocessor,
            self.sampler.get_sampler_builder,
        ):
            with self.subTest(lookup=lookup.__name__):
                with self.assertRaisesRegex(ImportError, "fixture dependency missing"):
                    lookup("discovery_fixture")


if __name__ == "__main__":
    unittest.main()
