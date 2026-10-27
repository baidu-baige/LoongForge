# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Padding rules live in the engine; data only applies the result."""

import pickle
from types import SimpleNamespace

import pytest
import torch

from loongforge.data.vlm.dataloader import seq_padding_for_cp
from loongforge.engines.megatron.data_padding import (
    VlmPackedPadding,
    llm_pad_to_multiple_of,
    vlm_packed_padding,
)


def _args(**overrides):
    values = dict(
        enable_chunkpipe=False,
        sft_chunkpipe_mode=False,
        chunksize=256,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        fp8=None,
        fp8_recipe=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_llm_padding_factors():
    assert llm_pad_to_multiple_of(_args()) == 1
    assert llm_pad_to_multiple_of(_args(tensor_model_parallel_size=8, sequence_parallel=True)) == 8
    assert llm_pad_to_multiple_of(_args(context_parallel_size=4)) == 8
    assert llm_pad_to_multiple_of(
        _args(tensor_model_parallel_size=2, sequence_parallel=True, context_parallel_size=2)
    ) == 8
    assert llm_pad_to_multiple_of(_args(fp8="e4m3")) == 128
    assert llm_pad_to_multiple_of(
        _args(tensor_model_parallel_size=2, sequence_parallel=True, fp8="e4m3")
    ) == 256


def test_chunkpipe_padding_is_the_chunk_size():
    assert llm_pad_to_multiple_of(
        _args(enable_chunkpipe=True, sft_chunkpipe_mode=True, chunksize=512)
    ) == 512


def test_vlm_packed_padding_only_when_alignment_is_needed():
    assert vlm_packed_padding(_args()) is None
    assert vlm_packed_padding(_args(tensor_model_parallel_size=4)) is None
    # Non-blockwise FP8 alone does not require packed alignment.
    assert vlm_packed_padding(_args(fp8="e4m3", fp8_recipe="delayed")) is None
    assert vlm_packed_padding(_args(context_parallel_size=2)) is not None
    assert vlm_packed_padding(_args(tensor_model_parallel_size=2, sequence_parallel=True)) is not None
    assert vlm_packed_padding(_args(fp8="e4m3", fp8_recipe="blockwise")) is not None


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(tp_size=4, cp_size=1, sequence_parallel=True, fp8_enabled=False, fp8_recipe=None), 4),
        (dict(tp_size=1, cp_size=2, sequence_parallel=False, fp8_enabled=False, fp8_recipe=None), 4),
        (dict(tp_size=2, cp_size=2, sequence_parallel=True, fp8_enabled=False, fp8_recipe=None), 8),
        # lcm(1, 128) for blockwise FP8 without SP.
        (dict(tp_size=2, cp_size=1, sequence_parallel=False, fp8_enabled=True, fp8_recipe="blockwise"), 128),
        # lcm(2, 128 * 2) for blockwise FP8 with SP.
        (dict(tp_size=2, cp_size=1, sequence_parallel=True, fp8_enabled=True, fp8_recipe="blockwise"), 256),
        # lcm(4, 32) for mxfp8 under CP.
        (dict(tp_size=1, cp_size=2, sequence_parallel=False, fp8_enabled=True, fp8_recipe="mxfp8"), 32),
        # lcm(12, 16 * 3) for the default FP8 recipe under SP + CP.
        (dict(tp_size=3, cp_size=2, sequence_parallel=True, fp8_enabled=True, fp8_recipe="delayed"), 48),
    ],
)
def test_vlm_total_multiple(kwargs, expected):
    assert VlmPackedPadding(**kwargs).total_multiple == expected


def test_vlm_packed_padding_is_picklable_and_frozen():
    padding = vlm_packed_padding(_args(tensor_model_parallel_size=2, fp8="e4m3", fp8_recipe="blockwise"))
    assert padding == VlmPackedPadding(2, 1, False, True, "blockwise")
    assert pickle.loads(pickle.dumps(padding)) == padding
    with pytest.raises(Exception):
        padding.tp_size = 4


def test_seq_padding_applies_segment_and_total_padding():
    padding = VlmPackedPadding(tp_size=1, cp_size=2, sequence_parallel=False, fp8_enabled=False, fp8_recipe=None)
    lengths = [3, 5]
    total = sum(lengths)
    data = {
        "tokens": torch.arange(total).unsqueeze(0),
        "labels": torch.arange(total).unsqueeze(0),
        "attn_mask": torch.zeros(1, total, dtype=torch.bool),
        "cu_lengths": torch.tensor([[0, 3, 8]], dtype=torch.int32),
        "max_lengths": torch.tensor([5], dtype=torch.int32),
    }
    out = seq_padding_for_cp(data, padding, pad_token_id=-1)
    padded = [n + padding.segment_padding(n) for n in lengths]
    assert out["cu_lengths"].tolist() == [[0, padded[0], padded[0] + padded[1]]]
    assert out["tokens"].shape[1] % padding.total_multiple == 0
    assert out["tokens"][0, :3].tolist() == [0, 1, 2]
    assert out["tokens"][0, 3:padded[0]].tolist() == [-1] * (padded[0] - 3)
