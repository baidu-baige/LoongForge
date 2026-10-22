# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.test_vlm_dataset_inputs import REPO_ROOT, _load_functions


class LayoutPathsTest(unittest.TestCase):
    def test_dreamzero_vae_loader_uses_existing_file(self):
        loader = Mock()
        spec = SimpleNamespace(loader=loader)
        module = object()
        util = SimpleNamespace(
            spec_from_file_location=Mock(return_value=spec),
            module_from_spec=Mock(return_value=module),
        )
        namespace = _load_functions(
            "tools/dreamzero_data_preprocess/cache_precompute/features.py",
            {"_load_vae_module"},
            {"_REPO_ROOT": REPO_ROOT, "importlib": SimpleNamespace(util=util)},
        )
        self.assertIs(namespace["_load_vae_module"](), module)
        path = util.spec_from_file_location.call_args.args[1]
        self.assertEqual(path.name, "wan_video_vae.py")
        self.assertTrue(path.is_file(), path)
        loader.exec_module.assert_called_once_with(module)

    def test_evaluation_default_roots_match_repo_layout(self):
        for relative in (
            "loongforge/evaluation/embodied/orchestrator/server_manager.py",
            "loongforge/evaluation/embodied/orchestrator/runners/robotwin_runner.py",
        ):
            with self.subTest(path=relative), patch.dict(os.environ, {}, clear=True):
                namespace = _load_functions(
                    relative,
                    {"build_argparser"},
                    {"argparse": argparse, "os": os, "sys": sys},
                )
                parser = namespace["build_argparser"]()
                args = parser.parse_args([])
                path = Path(args.eval_root).relative_to(args.loongforge_root)
                self.assertEqual(path, Path("loongforge/evaluation/embodied"))
                self.assertTrue((REPO_ROOT / path).is_dir())
                self.assertEqual(parser.parse_args(["--eval-root", "/custom"]).eval_root, "/custom")


class ExamplesLayoutTest(unittest.TestCase):
    """Guard the `examples*/<model>/<phase>/` convention so it cannot drift."""

    PHASES = {"pretrain", "finetuning", "checkpoint_convert", "eval", "preprocess"}
    MODEL_ROOT_FILES = {"README.md"}

    def _model_dirs(self):
        for root in ("examples", "examples_xpu"):
            for model in sorted((REPO_ROOT / root).iterdir()):
                if model.is_dir():
                    yield root, model

    def test_examples_directories_are_model_names(self):
        """Every examples dir names a model in configs/models, or a version of one."""
        config_models = {p.name for p in (REPO_ROOT / "configs/models").iterdir() if p.is_dir()}
        for root, model in self._model_dirs():
            with self.subTest(root=root, model=model.name):
                # `llama3.1` is a version of the `llama3` family directory; a
                # wildcard grouping directory such as `kimi_k2.x` is not.
                is_family = model.name in config_models
                is_version = any(
                    model.name.startswith(name) and model.name != name
                    for name in config_models
                )
                self.assertTrue(
                    is_family or is_version,
                    f"{root}/{model.name} is not a model under configs/models/",
                )

    def test_examples_model_dirs_use_phase_directories(self):
        """A model directory holds only phase subdirectories, assets/, and its README."""
        for root, model in self._model_dirs():
            with self.subTest(root=root, model=model.name):
                stray_files = sorted(
                    child.name for child in model.iterdir()
                    if child.is_file() and child.name not in self.MODEL_ROOT_FILES
                )
                self.assertEqual(stray_files, [], f"{root}/{model.name} has loose files")

                unknown_dirs = sorted(
                    child.name for child in model.iterdir()
                    if child.is_dir() and child.name not in self.PHASES and child.name != "assets"
                )
                self.assertEqual(unknown_dirs, [], f"{root}/{model.name} has non-phase dirs")

    def test_phase_scripts_resolve_repo_root_three_levels_up(self):
        """A script in `examples/<model>/<phase>/` needs `../../..` for the repo root."""
        two_levels_up = re.compile(r'(?:SCRIPT_DIR\}?/|BASH_SOURCE\[0\]\}"\))/\.\./\.\.["\']')
        for root, model in self._model_dirs():
            for phase in model.iterdir():
                if not phase.is_dir() or phase.name not in self.PHASES:
                    continue
                for script in sorted(phase.glob("*.sh")):
                    text = script.read_text(encoding="utf-8")
                    with self.subTest(script=str(script.relative_to(REPO_ROOT))):
                        self.assertEqual(
                            two_levels_up.findall(text), [],
                            "two levels up from a phase directory is examples/, not the repo root",
                        )


if __name__ == "__main__":
    unittest.main()
