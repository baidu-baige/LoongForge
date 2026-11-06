# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Reordering keeps tokens, labels, images, and videos in sample order."""

import ast
import sys
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rebalance(monkeypatch):
    args = NS(sequence_parallel=False, force_image_size=2, dp_balance_max_len_ratio_vlm=1.0)
    training = ModuleType("megatron.training")
    training.get_args = lambda: args
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    namespace = dict(
        torch=torch, partial=partial, get_args=training.get_args, IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        get_tokenizer=lambda: NS(tokenizer=NS(convert_tokens_to_ids=lambda token: 99)),
    )
    definitions = {
        "pack.py": {
            "VLMDataSample", "InternVLDataSample", "pack_data_for_vlm", "depack_data_for_vlm",
            "pack_data_for_intern_vl", "depack_data_for_intern_vl",
        },
        "reconstruct.py": {
            "reconstruct_llm_for_vlm", "reconstruct_visual_for_vlm", "reconstruct_visual_grid_thw_for_vlm",
            "reconstruct_llm_for_internvl", "reconstruct_image_flags_for_internvl",
            "reconstruct_pixel_values_for_internvl",
        },
        "balance.py": {"reorder_data_for_vlm", "reorder_data_for_internvl"},
    }
    for filename, names in definitions.items():
        path = ROOT / "loongforge/data/dp_balance/rebalance" / filename
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        assert {node.name for node in nodes} == names
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)

    def configure(skip):
        order = None if skip else [1, 0]
        gathered = (object(), object(), object())

        def redistribute(tensors, plan, *metadata):
            assert plan is order
            reconstruct = metadata[-1]
            return [reconstruct(tensors[index].flatten()) for index in order]

        namespace.update(
            gather_sample_info_across_dp=Mock(return_value=gathered),
            get_dp_group_by_device=lambda tensor: NS(size=lambda: 1),
            solve_sample_dp_reorder_plan=Mock(return_value=(order, [1.0])),
            _load_tracker=NS(get_historical_costs=Mock(return_value=None), update=Mock()),
            load_estimate_per_sample=object(),
            redistribute_tensor_helper=Mock(side_effect=redistribute),
            redistribute_tensors=Mock(side_effect=redistribute),
        )
        return namespace

    return configure


@pytest.mark.parametrize("visuals", [("video",), ("image", "video")])
@pytest.mark.parametrize("skip", [False, True])
def test_vlm_reorder_keeps_tokens_and_visuals_together(rebalance, visuals, skip):
    functions = rebalance(skip)
    samples = []
    for index, length in enumerate((2, 3)):
        tokens = torch.full((1, length), index + 1)
        image = torch.full((2, 4), index + 10.0) if "image" in visuals else None
        samples.append(functions["VLMDataSample"](
            tokens, tokens + 100, tokens + 200,
            image, torch.tensor([1, 1, 2]) if image is not None else None,
            torch.full((4, 4), index + 20.0), torch.tensor([2, 1, 2]),
        ))
    data = functions["pack_data_for_vlm"](samples)
    actual = functions["reorder_data_for_vlm"](data)
    gather = functions["gather_sample_info_across_dp"]
    torch.testing.assert_close(gather.call_args_list[0].args[0], torch.tensor([2, 3], dtype=torch.int))
    functions["_load_tracker"].update.assert_called_once_with([1.0])
    if skip:
        assert actual is data
        functions["redistribute_tensor_helper"].assert_not_called()
    else:
        expected = functions["pack_data_for_vlm"](samples[::-1])
        for key, value in expected.items():
            if value is None:
                assert actual[key] is None
            else:
                torch.testing.assert_close(actual[key], value)
        assert functions["redistribute_tensor_helper"].call_count == 3 + 2 * len(visuals)


@pytest.mark.parametrize("skip", [False, True])
def test_internvl_reorder_keeps_tokens_and_images_together(rebalance, skip):
    functions = rebalance(skip)
    samples = [
        functions["InternVLDataSample"](
            torch.full((1, 3, 2, 2), float(index)),
            torch.tensor([[99] + [index] * (index + 1)]),
            torch.ones(1, dtype=torch.int64),
            torch.full((1, index + 2), 0.5),
            torch.full((1, index + 2), index + 10),
        ) for index in range(2)
    ]
    data = functions["pack_data_for_intern_vl"](samples)
    actual = functions["reorder_data_for_internvl"](data)
    if skip:
        assert actual is data
        functions["redistribute_tensor_helper"].assert_not_called()
        functions["redistribute_tensors"].assert_not_called()
    else:
        expected = functions["pack_data_for_intern_vl"](samples[::-1])
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        assert functions["redistribute_tensor_helper"].call_count == 3
        assert functions["redistribute_tensors"].call_count == 2
