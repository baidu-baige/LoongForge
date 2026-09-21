# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
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
            "tools/data_preprocess/embodied/dreamzero/cache_precompute/features.py",
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


if __name__ == "__main__":
    unittest.main()
