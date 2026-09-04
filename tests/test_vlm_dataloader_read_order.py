# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_functions(relative_path, names, namespace=None):
    """Load selected pure-Python functions without importing the GPU stack."""
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {function.name for function in functions} == set(names)

    namespace = {} if namespace is None else namespace
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            SimpleNamespace(),
            {"shuffle_buffer_size": None, "max_samples_per_sequence": None},
        ),
        (
            SimpleNamespace(
                data_shuffle_buffer_size=1,
                data_max_samples_per_sequence=0,
            ),
            {"shuffle_buffer_size": None, "max_samples_per_sequence": None},
        ),
        (
            SimpleNamespace(
                data_shuffle_buffer_size=4096,
                data_max_samples_per_sequence=256,
            ),
            {"shuffle_buffer_size": 4096, "max_samples_per_sequence": 256},
        ),
    ],
)
def test_energon_read_order_kwargs(args, expected):
    namespace = _load_functions(
        "loongforge/data/multimodal/dataloader_provider.py",
        {"_energon_read_order_kwargs"},
    )
    assert namespace["_energon_read_order_kwargs"](args) == expected


@pytest.mark.parametrize(
    ("data_path", "expected_path"),
    [
        (["dataset"], "dataset"),
        (["dataset-a", "dataset-b"], "/tmp/metadataset.yaml"),
    ],
)
def test_get_train_dataset_forwards_read_order_kwargs(data_path, expected_path):
    get_train_dataset = Mock(return_value="train-dataset")
    worker_config = object()
    energon = SimpleNamespace(
        WorkerConfig=Mock(return_value=worker_config),
        get_train_dataset=get_train_dataset,
    )
    args = SimpleNamespace(
        data_path=data_path,
        micro_batch_size=2,
        num_workers=4,
        packing_buffer_size=10000,
        data_shuffle_buffer_size=4096,
        data_max_samples_per_sequence=256,
        rank=0,
    )
    namespace = {
        "energon": energon,
        "parallel_state": SimpleNamespace(
            get_data_parallel_rank=lambda: 2,
            get_data_parallel_world_size=lambda: 8,
            get_data_parallel_group=lambda: "dp-group",
        ),
        "get_args": lambda: args,
        "get_blend_from_list": Mock(
            return_value=(["dataset-a", "dataset-b"], [0.25, 0.75])
        ),
        "create_metadataset_yaml": Mock(return_value="/tmp/metadataset.yaml"),
        "print_error_handler": object(),
        "print_rank_0": Mock(),
    }
    namespace = _load_functions(
        "loongforge/data/multimodal/dataloader_provider.py",
        {
            "_energon_read_order_kwargs",
            "_validate_energon_data_paths",
            "get_train_dataset",
        },
        namespace,
    )

    assert namespace["get_train_dataset"]("task-encoder") == "train-dataset"
    get_train_dataset.assert_called_once()
    path, = get_train_dataset.call_args.args
    kwargs = get_train_dataset.call_args.kwargs
    assert path == expected_path
    assert kwargs["shuffle_buffer_size"] == 4096
    assert kwargs["max_samples_per_sequence"] == 256
    assert kwargs["task_encoder"] == "task-encoder"
    assert kwargs["worker_config"] is worker_config


def test_multimodal_argument_defaults_and_overrides():
    class LanguageModelFamilies:
        @staticmethod
        def names():
            return ["llama"]

    namespace = _load_functions(
        "loongforge/train/arguments.py",
        {"_add_extra_multimodal_args"},
        {
            "get_support_model_archs": lambda values: values,
            "constants": SimpleNamespace(LanguageModelFamilies=LanguageModelFamilies),
        },
    )
    parser = namespace["_add_extra_multimodal_args"](argparse.ArgumentParser())

    defaults = parser.parse_args([])
    assert defaults.data_shuffle_buffer_size == 0
    assert defaults.data_max_samples_per_sequence == 0

    configured = parser.parse_args(
        [
            "--data-shuffle-buffer-size",
            "4096",
            "--data-max-samples-per-sequence",
            "256",
        ]
    )
    assert configured.data_shuffle_buffer_size == 4096
    assert configured.data_max_samples_per_sequence == 256
