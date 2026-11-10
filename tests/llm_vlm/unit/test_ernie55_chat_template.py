# Copyright 2026 The LoongForge Authors.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT text and supervision against the Bridge ERNIE template fixtures."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from loongforge.data.chat_template import MAPPING_NAME_TO_TEMPLATE
from loongforge.tokenizer import build_tokenizer


CASES = json.loads((Path(__file__).parent / "fixtures/ernie55_reference_template_cases.json").read_text())["cases"]


@pytest.fixture
def tokenizer():
    vocab = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|ernie:im_end|>",
        pad_token="<|ernie:im_end|>",
    )


def test_fast_tokenizer_loads_without_a_transformers_ernie_model(tmp_path, tokenizer):
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "ernie5_5_moe",
                "max_position_embeddings": 1048576,
                "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536},
            }
        )
    )
    args = SimpleNamespace(
        tokenizer_type="HFTokenizer",
        model_family="ernie5_5",
        rank=0,
        hf_tokenizer_path=str(tmp_path),
        use_fast_tokenizer=True,
        padding_side="right",
        seq_length=512,
        split_special_tokens=False,
        additional_special_tokens=None,
        training_phase="pretrain",
        padded_vocab_size=len(tokenizer),
    )
    loaded = build_tokenizer(args)
    text = "你好，计算 1+1。"
    assert loaded.tokenize(text) == tokenizer.encode(text)
    assert loaded.vocab_size == len(tokenizer)


@pytest.mark.parametrize("case", [case for case in CASES if not case.get("inference")], ids=lambda case: case["name"])
def test_reference_sft_text_and_assistant_mask(case, tokenizer):
    template = deepcopy(MAPPING_NAME_TO_TEMPLATE["ernie5.5-hf"])
    template.chat_template_kwargs = case.get("kwargs", {})
    original = deepcopy(case["messages"])
    ids, labels, mask, length = template.encode_openai(
        SimpleNamespace(hf_tokenizer=lambda: tokenizer),
        case["messages"],
        tools=case.get("tools"),
    )
    assert tokenizer.decode(ids) == case["text"]
    assert tokenizer.decode([label for label in labels if label != -100]) == case["supervised_text"]
    assert len(ids) == len(labels) == len(mask) == length
    assert mask == [int(label != -100) for label in labels]
    assert case["messages"] == original


def test_string_tool_arguments_preserve_text_and_supervision(tokenizer):
    template = deepcopy(MAPPING_NAME_TO_TEMPLATE["ernie5.5-hf"])
    case = deepcopy(next(case for case in CASES if case["name"] == "tools_xml"))
    call = case["messages"][1]["tool_calls"][0]["function"]
    call["arguments"] = json.dumps(call["arguments"])
    original = deepcopy(case["messages"])
    ids, labels, _, _ = template.encode_openai(
        SimpleNamespace(hf_tokenizer=lambda: tokenizer),
        case["messages"],
        tools=case["tools"],
    )
    assert tokenizer.decode(ids) == case["text"]
    assert tokenizer.decode([label for label in labels if label != -100]) == case["supervised_text"]
    assert case["messages"] == original
