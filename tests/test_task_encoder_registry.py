# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import ast
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tests.test_vlm_dataset_inputs import REPO_ROOT, _load_functions


SOURCE = "loongforge/data/vlm/__init__.py"
# Read the real table without importing optional training dependencies.
REGISTRY = next(
    ast.literal_eval(node.value)
    for node in ast.parse((REPO_ROOT / SOURCE).read_text(encoding="utf-8")).body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "TASK_ENCODER_REGISTRY" for target in node.targets)
)


@pytest.fixture
def resolver():
    load = Mock()
    namespace = _load_functions(
        SOURCE,
        {"resolve_task_encoder"},
        {"TASK_ENCODER_REGISTRY": REGISTRY, "importlib": SimpleNamespace(import_module=load)},
    )
    return namespace["resolve_task_encoder"], load


@pytest.mark.parametrize("name,target", REGISTRY.items())
def test_resolver_selects_registered_module_and_class(resolver, name, target):
    resolve, load = resolver
    module_path, class_name = target.rsplit(".", 1)
    encoder = type(class_name, (), {})
    load.return_value = SimpleNamespace(**{class_name: encoder})

    assert resolve(name) is encoder
    assert resolve(name.upper()) is encoder
    assert load.call_args_list == [call(module_path), call(module_path)]


def test_unknown_encoder_keeps_value_error_and_available_names(resolver):
    resolve, load = resolver
    name = "missing-task-encoder"

    with pytest.raises(ValueError) as raised:
        resolve(name)

    assert str(raised.value) == f"Unknown task encoder '{name}'. Available: {sorted(REGISTRY)}"
    load.assert_not_called()


def test_resolver_preserves_import_error(resolver):
    resolve, load = resolver
    name = next(iter(REGISTRY))
    error = ModuleNotFoundError("encoder dependency missing")
    load.side_effect = error

    with pytest.raises(ModuleNotFoundError) as raised:
        resolve(name)

    assert raised.value is error
    load.assert_called_once_with(REGISTRY[name].rsplit(".", 1)[0])


def test_resolver_preserves_missing_class_error(resolver):
    resolve, load = resolver
    name, target = next(iter(REGISTRY.items()))
    module_path, class_name = target.rsplit(".", 1)
    load.return_value = SimpleNamespace()

    with pytest.raises(AttributeError, match=class_name):
        resolve(name)

    load.assert_called_once_with(module_path)
