# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SFT preprocessing behavior with small token fixtures and no Torch/HF imports."""

import ast
import bisect
import json
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loongforge.data import constants


class FakeLegacyTemplate:
    mm_plugin = None
    efficient_eos = False

    def encode_multiturn(self, tokenizer, messages, system):
        return [
            (
                list(map(int, (system or "") if i == 0 else "")) + list(map(int, messages[i]["content"])),
                list(map(int, messages[i + 1]["content"])),
            )
            for i in range(0, len(messages) - 1, 2)
        ]


class FakeHFChatTemplate:
    def __init__(self):
        self.encode_openai = Mock(return_value=([1, 2, 3, 4], [-100, -100, 3, 4], [0, 0, 1, 1], 4))


@pytest.fixture(scope="module")
def preprocess():
    source = Path(__file__).resolve().parents[1] / "loongforge/data/llm/sft_supervised_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(
        bisect=bisect,
        json=json,
        defaultdict=defaultdict,
        constants=constants,
        logger=logging.getLogger("sft_preprocess_test"),
        HFChatTemplate=FakeHFChatTemplate,
    )
    # Keep production function bodies; defer annotations to avoid heavy imports.
    body = ast.parse("from __future__ import annotations").body + functions
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_preprocess_supervised_dataset"]


def _config(**overrides):
    values = dict(
        chat_template=FakeLegacyTemplate(),
        tokenizer=SimpleNamespace(pad=0, eos=9, padding_side="right"),
        processor=None,
        sequence_length=16,
        history_mask_loss=False,
        train_on_prompt=False,
        ignore_index=constants.IGNORE_INDEX,
        eod_mask_loss=False,
        enable_discard_sample=False,
        packing=False,
        sort_batch=False,
        context_parallel_size=1,
        enable_chunkpipe=False,
        chunksize=4,
        mtp_num_layers=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _samples(*pairs):
    return {
        "prompt": [[{"role": "user", "content": p}] for p, _ in pairs],
        "response": [[{"role": "assistant", "content": a}] for _, a in pairs],
        "system": [None for _ in pairs],
        "images": [[] for _ in pairs],
        "videos": [[] for _ in pairs],
    }


@pytest.mark.parametrize("eod_mask_loss", [False, True])
@pytest.mark.parametrize(
    "efficient_eos,history_mask_loss,labels,mask",
    [
        (False, False, [-100, -100, 2, -100, 4], [0, 0, 1, 0, 1]),
        (False, True, [-100, -100, -100, -100, 4], [0, 0, 0, 0, 1]),
        (True, False, [-100, -100, 2, 9, 4, 9], [0, 0, 1, 1, 1, 1]),
        (True, True, [9, -100, -100, -100, 4, 9], [1, 0, 0, 0, 1, 1]),
    ],
)
def test_plain_masks_and_media(preprocess, eod_mask_loss, efficient_eos, history_mask_loss, labels, mask):
    samples = _samples(("1", "2"), ("", ""))
    samples["prompt"][0] += samples["response"][0] + [{"role": "user", "content": "3"}]
    samples["response"][0] = [{"role": "assistant", "content": "4"}]
    samples["system"][0] = "8"
    samples["images"][0], samples["videos"][0] = ["a.png"], ["b.mp4"]
    config = _config(eod_mask_loss=eod_mask_loss, history_mask_loss=history_mask_loss)
    config.chat_template.efficient_eos = efficient_eos
    out = preprocess(samples, config)
    assert out["input_ids"][0] == [8, 1, 2, 3, 4] + ([9] if efficient_eos else [])
    assert out["labels"][0] == labels
    assert out["attention_mask"][0] == [1] * len(labels)
    assert out["images"][0] == ["a.png"]
    assert out["videos"][0] == ["b.mp4"]
    assert ("loss_mask" in out) is not eod_mask_loss
    if not eod_mask_loss:
        assert out["loss_mask"][0] == mask
    if efficient_eos:
        assert out["input_ids"][1] == out["labels"][1] == [9]
    else:
        assert len(out["input_ids"]) == 1


def test_train_on_prompt(preprocess):
    out = preprocess(_samples(("12", "34")), _config(train_on_prompt=True))
    assert out["input_ids"] == out["labels"] == [[1, 2, 3, 4]]
    assert out["loss_mask"] == [[1, 1, 1, 1]]


@pytest.mark.parametrize("discard", [False, True])
def test_truncation_and_discard_use_original_length(preprocess, discard):
    out = preprocess(
        _samples(("1", "234"), ("11", "2345")),
        _config(sequence_length=4, enable_discard_sample=discard),
    )
    ids = [[1, 2, 3, 4], [1, 1, 2, 3]]
    labels = [[-100, 2, 3, 4], [-100, -100, 2, 3]]
    masks = [[0, 1, 1, 1], [0, 0, 1, 1]]
    assert out["input_ids"] == (ids[:1] if discard else ids)
    assert out["labels"] == (labels[:1] if discard else labels)
    assert out["loss_mask"] == (masks[:1] if discard else masks)
    assert out["attention_mask"] == [[1, 1, 1, 1]] * (1 if discard else 2)


def test_skips_empty_and_invalid_turn_counts(preprocess, caplog):
    samples = _samples(("1", "2"), ("", ""), ("3", "4"), ("5", "6"), ("7", "8"))
    samples["prompt"][2] *= 2
    samples["response"][3] = []
    samples["response"][4] *= 2
    out = preprocess(samples, _config())
    assert out["input_ids"] == [[1, 2]]
    assert out["labels"] == [[-100, 2]]
    assert out["loss_mask"] == [[0, 1]]
    for message in ("no tokens", "2 prompt msgs", "0 response msgs", "2 response msgs"):
        assert message in caplog.text


@pytest.mark.parametrize("sort_batch", [False, True])
@pytest.mark.parametrize("eod_mask_loss", [False, True])
def test_packing_order_masks_and_load_sort(preprocess, sort_batch, eod_mask_loss):
    out = preprocess(
        _samples(("1", "23456"), ("2", "345"), ("6", "781"), ("7", "8")),
        _config(
            packing=True,
            sequence_length=8,
            sort_batch=sort_batch,
            eod_mask_loss=eod_mask_loss,
        ),
    )
    expected = [
        ([1, 2, 3, 4, 5, 6, 7, 8], [-100, 2, 3, 4, 5, 6, -100, 8], [1] * 6 + [2] * 2, [0, 1, 1, 1, 1, 1, 0, 1]),
        ([6, 7, 8, 1, 2, 3, 4, 5], [-100, 7, 8, 1, -100, 3, 4, 5], [1] * 4 + [2] * 4, [0, 1, 1, 1, 0, 1, 1, 1]),
    ]
    if sort_batch:
        expected.reverse()  # Equal row lengths, but costs 32 and 40.
    for key, column in zip(("input_ids", "labels", "attention_mask"), zip(*expected)):
        assert out[key] == list(column)
    assert out["images"] == out["videos"] == [[], []]
    assert ("loss_mask" in out) is not eod_mask_loss
    if not eod_mask_loss:
        assert out["loss_mask"] == [row[3] for row in expected]


@pytest.mark.parametrize(
    "side,ids,labels,mask",
    [
        ("right", [4, 5, 0, 0, 1, 2, 3, 0], [-100, 5, -100, -100, -100, 2, 3, -100], [0, 1, 0, 0, 0, 1, 1, 0]),
        ("left", [0, 0, 4, 5, 0, 1, 2, 3], [-100, -100, -100, 5, -100, -100, 2, 3], [0, 0, 0, 1, 0, 0, 1, 1]),
    ],
)
def test_packing_cp_padding(preprocess, side, ids, labels, mask):
    config = _config(packing=True, sequence_length=8, context_parallel_size=2)
    config.tokenizer.padding_side = side
    out = preprocess(_samples(("1", "23"), ("4", "5")), config)
    assert out["input_ids"] == [ids]
    assert out["labels"] == [labels]
    assert out["loss_mask"] == [mask]
    assert out["attention_mask"] == [[1, 1, 1, 1, 2, 2, 2, 2]]


def test_packing_discards_sample_that_exceeds_limit_after_cp_padding(preprocess):
    out = preprocess(
        _samples(("123", "4567"), ("1", "2")),
        _config(packing=True, sequence_length=7, context_parallel_size=2),
    )
    assert out["input_ids"] == [[1, 2, 0, 0]]
    assert out["labels"] == [[-100, 2, -100, -100]]
    assert out["loss_mask"] == [[0, 1, 0, 0]]
    assert out["attention_mask"] == [[1, 1, 1, 1]]


@pytest.mark.parametrize("column", ["images", "videos"])
def test_packing_rejects_media(preprocess, column):
    samples = _samples(("1", "2"))
    samples[column][0] = ["media"]
    with pytest.raises(AssertionError, match="packing is not supported for images/videos"):
        preprocess(samples, _config(packing=True))


@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize("eod_mask_loss", [False, True])
@pytest.mark.parametrize("packing", [False, True])
def test_chunkpipe_shifts_groups_and_bridges(preprocess, mtp, eod_mask_loss, packing):
    out = preprocess(
        _samples(("3", "45"), ("12", "34567"), ("", "8")),
        _config(
            enable_chunkpipe=True,
            mtp_num_layers=mtp,
            eod_mask_loss=eod_mask_loss,
            packing=packing,
            sort_batch=True,
        ),
    )
    ids = [[1, 2, 3, 4], [5, 6, 7, 0], [3, 4, 5, 8]]
    labels = [[-100, 3, 4, 5], [6, 7, -100, -100], [4, 5, -100, -100]]
    masks = [[0, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]
    if mtp:
        ids = [ids[0] + [5, 6], ids[1] + [0, 0], ids[2] + [0, 0]]
        labels = [labels[0] + [6, 7], labels[1] + [-100, -100], labels[2] + [-100, -100]]
        masks = [masks[0] + [1, 1], masks[1] + [0, 0], masks[2] + [0, 0]]
    assert out["input_ids"] == ids
    assert out["labels"] == labels
    assert out["attention_mask"] == [
        [1, 1, 1, 1] + [0] * mtp,
        [1, 1, 1, 1] + [0] * mtp,
        [1, 1, 1, 2] + [0] * mtp,
    ]
    assert out["chunk_group_size"] == [2, 2, 1]
    assert out["group_total_tokens"] == [5, 5, 2]
    assert out["images"] == out["videos"] == [[], [], []]
    assert ("loss_mask" in out) is not eod_mask_loss
    if not eod_mask_loss:
        assert out["loss_mask"] == masks


def test_chunkpipe_pads_short_bin_without_cross_sample_loss(preprocess):
    out = preprocess(
        _samples(("1", "2")),
        _config(enable_chunkpipe=True, mtp_num_layers=1),
    )
    assert out["input_ids"] == [[1, 2, 0, 0, 0]]
    assert out["labels"] == [[2, -100, -100, -100, -100]]
    assert out["loss_mask"] == [[1, 0, 0, 0, 0]]
    assert out["attention_mask"] == [[1, 1, 0, 0, 0]]
    assert out["chunk_group_size"] == out["group_total_tokens"] == [1]


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("mode", ["plain", "discard", "packing", "chunkpipe"])
def test_openai_modes_and_json_decoding(preprocess, encoded, mode):
    template = FakeHFChatTemplate()
    options = {
        "plain": {},
        "discard": dict(sequence_length=3, enable_discard_sample=True),
        "packing": dict(packing=True, sort_batch=True),
        "chunkpipe": dict(enable_chunkpipe=True, chunksize=3, mtp_num_layers=1),
    }[mode]
    config = _config(chat_template=template, history_mask_loss=True, **options)
    messages = [{"role": "user", "content": "12"}, {"role": "assistant", "content": "34"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    samples = dict(
        messages=[json.dumps(messages) if encoded else messages],
        tools=[json.dumps(tools) if encoded else tools],
        images=[[]],
        videos=[[]],
    )
    if mode == "discard":
        template.encode_openai.return_value = ([1, 2, 3], [-100, -100, 3], [0, 0, 1], 4)
    out = preprocess(samples, config)
    template.encode_openai.assert_called_once_with(
        tokenizer=config.tokenizer,
        messages=messages,
        tools=tools,
        train_on_prompt=False,
        history_mask_loss=True,
        ignore_index=-100,
        max_length=config.sequence_length,
    )
    if mode == "discard":
        assert all(value == [] for value in out.values())
    elif mode == "chunkpipe":
        assert out["input_ids"] == [[1, 2, 3, 4], [4, 0, 0, 0]]
        assert out["labels"] == [[-100, 3, 4, -100], [-100, -100, -100, -100]]
        assert out["loss_mask"] == [[0, 1, 1, 0], [0, 0, 0, 0]]
        assert out["attention_mask"] == [[1, 1, 1, 0], [1, 1, 1, 0]]
        assert out["chunk_group_size"] == out["group_total_tokens"] == [2, 2]
        assert out["images"] == out["videos"] == [[], []]
    else:
        assert out["input_ids"] == [[1, 2, 3, 4]]
        assert out["labels"] == [[-100, -100, 3, 4]]
        assert out["loss_mask"] == [[0, 0, 1, 1]]
        assert out["attention_mask"] == [[1, 1, 1, 1]]
        assert out["images"] == out["videos"] == [[]]


@pytest.mark.parametrize("tools", [None, "", "missing"])
def test_openai_absent_tools(preprocess, tools):
    template = FakeHFChatTemplate()
    template.encode_openai.return_value = ([1, 2, 3, 4], [1, 2, 3, 4], [1, 1, 1, 1], 4)
    messages = [{"role": "user", "content": "12"}, {"role": "assistant", "content": "34"}]
    samples = dict(messages=[messages], images=[[]], videos=[[]])
    if tools != "missing":
        samples["tools"] = [tools]
    out = preprocess(samples, _config(chat_template=template, train_on_prompt=True))
    assert template.encode_openai.call_args.kwargs["tools"] is None
    assert template.encode_openai.call_args.kwargs["train_on_prompt"] is True
    assert out["input_ids"] == out["labels"] == [[1, 2, 3, 4]]
    assert out["loss_mask"] == [[1, 1, 1, 1]]


@pytest.mark.parametrize(
    "template,samples,message",
    [
        (FakeHFChatTemplate, {"prompt": []}, "HFChatTemplate requires"),
        (FakeLegacyTemplate, {"messages": []}, "Legacy ChatTemplate"),
        (FakeLegacyTemplate, {"prompt": []}, "Legacy ChatTemplate"),
        (FakeLegacyTemplate, {"response": []}, "Legacy ChatTemplate"),
    ],
)
def test_missing_columns_are_rejected(preprocess, template, samples, message):
    with pytest.raises(ValueError, match=message):
        preprocess(samples, _config(chat_template=template()))


@pytest.mark.parametrize("messages", [{}, "{}"])
def test_openai_rejects_non_list_messages(preprocess, messages):
    template = FakeHFChatTemplate()
    with pytest.raises(ValueError, match="messages must be a list"):
        preprocess({"messages": [messages]}, _config(chat_template=template))
    template.encode_openai.assert_not_called()
