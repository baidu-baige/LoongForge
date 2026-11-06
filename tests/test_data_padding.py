# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Padding factors and padded token, label, and mask values."""

from math import gcd
from types import SimpleNamespace as NS

import pytest

from tests.test_engine_migration import _load_functions


def _args(**overrides):
    values = dict(
        enable_chunkpipe=False, sft_chunkpipe_mode=False, chunksize=256,
        tensor_model_parallel_size=1, sequence_parallel=False, context_parallel_size=1,
        fp8=None, fp8_recipe=None, packing_sft_data=True, variable_seq_lengths=True,
        seq_length=1024, mtp_num_layers=0,
    )
    values.update(overrides)
    return NS(**values)


@pytest.mark.parametrize(("overrides", "expected"), [
    ({}, 1),
    (dict(tensor_model_parallel_size=8, sequence_parallel=True), 8),
    (dict(context_parallel_size=4), 8),
    (dict(tensor_model_parallel_size=2, sequence_parallel=True, context_parallel_size=2), 8),
    (dict(fp8="e4m3"), 128),
    (dict(tensor_model_parallel_size=2, sequence_parallel=True, fp8="e4m3"), 256),
    (dict(enable_chunkpipe=True, sft_chunkpipe_mode=True, chunksize=512), 512),
])
def test_llm_padding_factors(overrides, expected):
    args = _args(**overrides)
    tokenizer = NS(hf_tokenizer=lambda: object())
    namespace = _load_functions(
        "loongforge/data/utils.py", {"build_sft_data_collator"},
        get_args=lambda: args, get_tokenizer=lambda: tokenizer, AutoTokenizerFromHF=NS,
        constants=NS(IGNORE_INDEX=-100), PaddingStrategy=NS(LONGEST="longest", MAX_LENGTH="max_length"),
    )
    collator = namespace["build_sft_data_collator"](NS)
    assert collator.pad_to_multiple_of == expected
    if args.enable_chunkpipe:
        assert collator.max_length == 512
        assert collator.chunkpipe_base_length == 512


@pytest.mark.parametrize(("overrides", "expected"), [
    ({}, False),
    (dict(tensor_model_parallel_size=4), False),
    (dict(fp8="e4m3", fp8_recipe="delayed"), False),
    (dict(context_parallel_size=2), True),
    (dict(tensor_model_parallel_size=2, sequence_parallel=True), True),
    (dict(fp8="e4m3", fp8_recipe="blockwise"), True),
])
def test_vlm_packed_padding_only_when_alignment_is_needed(overrides, expected):
    namespace = _load_functions(
        "loongforge/data/vlm/dataloader_provider.py", {"_needs_packed_alignment"},
    )
    assert namespace["_needs_packed_alignment"](_args(**overrides)) is expected


@pytest.fixture
def padding_api():
    torch = pytest.importorskip("torch")
    upstream = _load_functions(
        "third_party/Loong-Megatron/megatron/core/models/multimodal/context_parallel.py", {"get_padding"},
    )
    namespace = _load_functions(
        "loongforge/data/vlm/dataloader_provider.py",
        {"_lcm", "_sequence_padding_factor", "_fp8_padding_factor", "seq_padding_for_cp"},
        torch=torch, F=torch.nn.functional, gcd=gcd,
        context_parallel=NS(get_padding=upstream["get_padding"]), PAD_TOKEN_ID=151643, IGNORE_INDEX=-100,
    )
    return torch, namespace["seq_padding_for_cp"]


@pytest.mark.parametrize(("kwargs", "expected"), [
    (dict(tp_size=4, cp_size=1, has_sp=True), 4),
    (dict(tp_size=1, cp_size=2, has_sp=False), 4),
    (dict(tp_size=2, cp_size=2, has_sp=True), 8),
    (dict(tp_size=2, cp_size=1, has_sp=False, fp8_enabled=True, fp8_recipe="blockwise"), 128),
    (dict(tp_size=2, cp_size=1, has_sp=True, fp8_enabled=True, fp8_recipe="blockwise"), 256),
    (dict(tp_size=1, cp_size=2, has_sp=False, fp8_enabled=True, fp8_recipe="mxfp8"), 32),
    (dict(tp_size=3, cp_size=2, has_sp=True, fp8_enabled=True, fp8_recipe="delayed"), 48),
])
def test_vlm_total_multiple(padding_api, kwargs, expected):
    torch, pad = padding_api
    data = {
        "tokens": torch.arange(8).unsqueeze(0),
        "labels": torch.arange(8).unsqueeze(0),
        "attn_mask": torch.zeros(1, 8, dtype=torch.bool),
        "cu_lengths": torch.tensor([[0, 3, 8]], dtype=torch.int32),
        "max_lengths": torch.tensor([5], dtype=torch.int32),
    }
    out = pad(data, pad_token_id=-1, **kwargs)
    assert out["tokens"].shape[1] % expected == 0
    assert out["cu_lengths"][0, -1] == out["tokens"].shape[1]
    assert out["tokens"][0, :3].tolist() == [0, 1, 2]


def test_seq_padding_applies_segment_and_total_padding(padding_api):
    torch, pad = padding_api
    data = {
        "tokens": torch.arange(8).unsqueeze(0),
        "labels": torch.arange(8).unsqueeze(0),
        "attn_mask": torch.zeros(1, 8, dtype=torch.bool),
        "cu_lengths": torch.tensor([[0, 3, 8]], dtype=torch.int32),
        "max_lengths": torch.tensor([5], dtype=torch.int32),
    }
    out = pad(data, cp_size=2, pad_token_id=-1)
    assert out["cu_lengths"].tolist() == [[0, 4, 12]]
    assert out["max_lengths"].tolist() == [8]
    assert out["tokens"].tolist() == [[0, 1, 2, -1, 3, 4, 5, 6, 7, -1, -1, -1]]
    assert out["labels"].tolist() == [[0, 1, 2, -100, 3, 4, 5, 6, 7, -100, -100, -100]]
    assert out["attn_mask"].tolist() == [[False, False, False, True, False, False, False, False, False, True, True, True]]
