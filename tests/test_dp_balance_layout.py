# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from datetime import timedelta
import runpy
import sys
from types import SimpleNamespace
from typing import Any, Dict, get_type_hints
from unittest.mock import Mock, patch

import pytest

from tests.test_vlm_dataset_inputs import REPO_ROOT, _load_functions

torch = pytest.importorskip("torch")
REBALANCE = "loongforge/engines/megatron/distributed/dp_balance/rebalance/"
INTERNVL = "loongforge/training/megatron/methods/utils/internvl_dp_balance.py"


class ReverseContext:
    """Exercise layout callbacks without requiring a Megatron process group."""

    def __init__(self, skip=False):
        self.order = None if skip else [1, 0]
        self.plan = Mock(return_value=self.order)
        self.gather = Mock(return_value=object())
        self.grouped_calls = 0

    def redistribute(self, tensors, plan, gathered, reconstruct):
        assert plan is self.order
        assert gathered is self.gather.return_value
        return [reconstruct(tensors[i].flatten()) for i in plan]

    def redistribute_grouped(self, tensors, plan, reconstruct):
        assert plan is self.order
        self.grouped_calls += 1
        return [reconstruct(tensors[i].flatten()) for i in plan]


def vlm_functions():
    namespace = runpy.run_path(str(REPO_ROOT / REBALANCE / "pack.py"))
    namespace.update(runpy.run_path(str(REPO_ROOT / REBALANCE / "reconstruct.py")))
    namespace["partial"] = partial
    return _load_functions(
        REBALANCE + "balance.py", {"reorder_data_for_vlm"}, namespace,
    )


@pytest.mark.parametrize("visuals", [(), ("image",), ("video",), ("image", "video")])
@pytest.mark.parametrize("skip", [False, True])
def test_vlm_reorder_keeps_tokens_and_visuals_together(visuals, skip):
    functions = vlm_functions()
    samples = []
    for i, length in enumerate((2, 3)):
        tokens = torch.full((1, length), i + 1)
        image = torch.full((2, 4), i + 10.0) if "image" in visuals else None
        video = torch.full((4, 4), i + 20.0) if "video" in visuals else None
        samples.append(functions["VLMDataSample"](
            tokens, tokens + 100, tokens + 200, image,
            torch.tensor([1, 1, 2]) if image is not None else None,
            video, torch.tensor([2, 1, 2]) if video is not None else None,
        ))
    data = functions["pack_data_for_vlm"](samples)
    ctx = ReverseContext(skip)
    actual = functions["reorder_data_for_vlm"](data, ctx)
    torch.testing.assert_close(ctx.plan.call_args.args[0], torch.tensor([2, 3], dtype=torch.int))
    if skip:
        assert actual is data
        ctx.gather.assert_not_called()
        assert ctx.grouped_calls == 0
    else:
        expected = functions["pack_data_for_vlm"](samples[::-1])
        for key, value in expected.items():
            if value is None:
                assert actual[key] is None
            else:
                torch.testing.assert_close(actual[key], value)
        ctx.gather.assert_called_once()
        assert ctx.grouped_calls == 2 * len(visuals)
    for sample in samples:
        single = functions["pack_data_for_vlm"]([sample])
        unpacked = functions["depack_data_for_vlm"](single)
        assert len(unpacked) == 1
        torch.testing.assert_close(unpacked[0].tokens, sample.tokens)
        for key in ("image_grid_thw", "video_grid_thw"):
            if single[key] is not None:
                assert single[key].shape == (1, 3)


@pytest.mark.parametrize("skip", [False, True])
def test_internvl_reorder_and_layout_registration(skip):
    layout = runpy.run_path(str(REPO_ROOT / REBALANCE / "layout.py"))
    args = SimpleNamespace(force_image_size=2)
    tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 99))
    modules = {
        "loongforge.engines.megatron": SimpleNamespace(get_args=lambda: args, get_tokenizer=lambda: tokenizer),
        "loongforge.engines.megatron.distributed.dp_balance.rebalance.layout": SimpleNamespace(**layout),
        "loongforge.models.vision.internvl.internvl_config": SimpleNamespace(IMG_CONTEXT_TOKEN="<IMG_CONTEXT>"),
    }
    with patch.dict(sys.modules, modules):
        functions = runpy.run_path(str(REPO_ROOT / INTERNVL))
    assert get_type_hints(functions["pack_data_for_intern_vl"])["return"] == Dict[str, Any]
    samples = [
        functions["InternVLDataSample"](
            torch.full((1, 3, 2, 2), float(i)),
            torch.tensor([[99] + [i] * (i + 1)]),
            torch.ones(1, dtype=torch.int64),
            torch.full((1, i + 2), 0.5),
            torch.full((1, i + 2), i + 10),
        ) for i in range(2)
    ]
    data = functions["pack_data_for_intern_vl"](samples)
    ctx = ReverseContext(skip)
    actual = functions["reorder_data_for_internvl"](data, ctx)
    if skip:
        assert actual is data
        ctx.gather.assert_not_called()
        assert actual["attn_mask"].shape == (1, 3)
    else:
        expected = functions["pack_data_for_intern_vl"](samples[::-1])
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        assert ctx.grouped_calls == 2
    layout["install_batch_layout"]("intern_vl", functions["INTERNVL_BATCH_LAYOUT"])
    registered = layout["get_batch_layout"]("intern_vl")
    assert registered is functions["INTERNVL_BATCH_LAYOUT"]
    assert registered.caller == "InternVL"
    assert registered.cumulative_lengths(data) is data["attn_mask"]


def test_unsupported_family():
    layout = runpy.run_path(str(REPO_ROOT / REBALANCE / "layout.py"))
    with pytest.raises(ValueError, match="No batch layout registered"):
        layout["get_batch_layout"]("missing")


def _video_worker(rank, rendezvous):
    import torch.distributed as dist
    from loongforge.engines.megatron.distributed.dp_balance.rebalance import balance, layout as layouts, pack

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60),
    )
    try:
        layouts.install_batch_layout("qwen3_vl", balance.VLM_BATCH_LAYOUT)
        layout = layouts.get_batch_layout("qwen3_vl")
        assert layout is balance.VLM_BATCH_LAYOUT
        assert layout.reorder is balance.reorder_data_for_vlm
        assert layout.caller == "VLM"
        # A forced swap exercises the actual gather/all-to-all path on CPU batches.
        ctx = balance.RebalanceContext(1.0, "VLM")
        ctx.plan = lambda _: [[(0, 1)], [(0, 0)]]
        tokens = torch.full((1, rank + 2), rank + 1)
        data = pack.pack_data_for_vlm([pack.VLMDataSample(
            tokens, tokens + 10, torch.ones_like(tokens),
            pixel_values_videos=torch.full((2, 4), rank + 1.0),
            vid_thw=torch.tensor([2, 1, 1]),
        )])
        assert layout.cumulative_lengths(data) is data["cu_lengths"]
        with patch.object(balance, "get_dp_group_by_device", return_value=dist.group.WORLD):
            result = layout.reorder(data, ctx)
        other = 1 - rank
        torch.testing.assert_close(result["tokens"], torch.full((1, other + 2), other + 1))
        torch.testing.assert_close(result["labels"], torch.full((1, other + 2), other + 11))
        torch.testing.assert_close(result["pixel_values_videos"], torch.full((2, 4), other + 1.0))
        torch.testing.assert_close(result["video_grid_thw"], torch.tensor([[2, 1, 1]]))
    finally:
        dist.destroy_process_group()


def test_default_vlm_layout_cross_rank_redistribution(tmp_path):
    pytest.importorskip("megatron.core")
    import torch.multiprocessing as mp

    mp.spawn(_video_worker, args=(f"file://{tmp_path / 'video_rdzv'}",), nprocs=2, join=True)
