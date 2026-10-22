# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import runpy
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from tests.test_vlm_dataset_inputs import REPO_ROOT


@pytest.mark.parametrize(("operation", "module_suffix"), [("load", "loader"), ("save", "saver")])
def test_bridge_is_lazy_and_forwards_arguments(operation, module_suffix):
    module_name = f"mcore_bridge.checkpoint.hf_checkpoint_{module_suffix}"
    function_name = f"{operation}_hf_checkpoint_online"
    module = ModuleType(module_name)
    target = Mock(return_value=object())
    setattr(module, function_name, target)
    tools = str(REPO_ROOT / "tools")
    clean_path = [p for p in sys.path if p not in (str(REPO_ROOT), tools)]
    with patch.object(sys, "path", clean_path.copy()):
        before = set(sys.modules)
        bridge = runpy.run_path(str(REPO_ROOT / "loongforge/engines/megatron/hf_bridge.py"))
        assert sys.path == clean_path
        assert not any(name.startswith("mcore_bridge") for name in set(sys.modules) - before)
        with patch.dict(sys.modules, {module_name: module}):
            model, args = object(), object()
            assert bridge[function_name](model, args=args) is target.return_value
            target.assert_called_once_with(model, args=args)
            bridge[function_name](model, args=args)
        assert sys.path.count(tools) == 1
        assert sys.path.count(str(REPO_ROOT)) == 1
