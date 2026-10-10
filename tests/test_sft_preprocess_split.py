# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Differential test: split ``_preprocess_supervised_dataset`` must not change its output.

``_frozen_preprocess`` is a verbatim copy of the function before it was split into
``_encode_one`` / ``_append_row`` / ``_concat_knapsack`` / ``_emit_chunkpipe`` / ``_emit_packing``.
Both versions run on synthetic samples and the outputs are compared element by element.

Runs on CPU without torch, datasets or transformers: the functions are extracted from
the source file (same approach as tests/test_vlm_dataset_inputs.py) and the chat templates
are small fakes. Do not edit the frozen copy.
"""

import ast
import bisect
import json
import logging
import unittest
from collections import defaultdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from loongforge import constants

SOURCE = Path(__file__).resolve().parents[1] / "loongforge/data/llm/sft_tokenize_utils.py"


class FakeHFChatTemplate:
    """Stands in for ``loongforge.chat_templates.hf.HFChatTemplate`` (needs numpy)."""


def _load_namespace():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {
        "logging": logging, "bisect": bisect, "json": json, "partial": partial,
        "defaultdict": defaultdict, "constants": constants, "logger": logging.getLogger("sft_split_test"),
        "HFChatTemplate": FakeHFChatTemplate, "TYPE_CHECKING": False,
        "Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "Sequence": Sequence,
        "Tuple": Tuple, "Union": Union,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def _frozen_preprocess(ns):
    """Return the pre-split function, bound to the helpers loaded from the source file."""
    _encode_openai_example = ns["_encode_openai_example"]
    _encode_supervised_example = ns["_encode_supervised_example"]
    _build_knapsacks = ns["_build_knapsacks"]
    _pad_sequence_to_multiple = ns["_pad_sequence_to_multiple"]
    _split_long_sequence = ns["_split_long_sequence"]
    logger = ns["logger"]
    HFChatTemplate = FakeHFChatTemplate

    def _preprocess_supervised_dataset(samples, config):
        model_inputs = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "images": [],
            "videos": [],
        }

        if not config.eod_mask_loss:
            # pad may be equal to eos, in order to avoid the wrong execution of mask,
            # the loss mask is generated here separately
            model_inputs["loss_mask"] = []

        if config.enable_chunkpipe:
            # Track how many consecutive chunks belong to the same source sequence,
            # so that the sampler can keep them together and in order.
            model_inputs["chunk_group_size"] = []
            # Per-chunk N_g: total response tokens of the source sequence this chunk
            # belongs to. All chunks from the same source sequence carry the same
            # value; bin-packed short sequences store the bin's total response
            # tokens. Consumed by the SFT chunkpipe per-sample loss path.
            model_inputs["group_total_tokens"] = []

        pad_to_multiple_of = 1
        if config.packing:
            all_input_ids, all_labels, all_loss_mask = [], [], []
            all_sampel_lens = []
            len_to_sample_indexs = defaultdict(list)
            index = 0
            # When using context parallel, sequence is split by CP size
            pad_to_multiple_of *= (
                (2 * config.context_parallel_size)
                if (config.context_parallel_size and config.context_parallel_size > 1)
                else 1
            )

        if config.enable_chunkpipe:
            chunksize = config.chunksize
            mtp_num_layers = getattr(config, "mtp_num_layers", 0) or 0
            # Buffers for long sequences (len > chunksize): will be split into chunks
            long_input_ids, long_labels, long_loss_mask = [], [], []
            # Buffers for short sequences (len <= chunksize): will be binpacked
            short_input_ids, short_labels, short_loss_mask = [], [], []
            short_sample_lens = []
            short_len_to_sample_indexs = defaultdict(list)
            short_index = 0

        use_hf_chat_template = isinstance(config.chat_template, HFChatTemplate)
        if use_hf_chat_template:
            if "messages" not in samples:
                raise ValueError(
                    "HFChatTemplate requires OpenAI Chat Completions-style "
                    "`messages` samples. Use dataset format "
                    "`openai` with a registered `*-hf` chat template."
                )
            sample_count = len(samples["messages"])
        else:
            if "prompt" not in samples or "response" not in samples:
                raise ValueError(
                    "Legacy ChatTemplate preprocessing requires `prompt` and "
                    "`response` samples. Use a registered `*-hf` chat template for "
                    "OpenAI Chat Completions-style `messages` samples."
                )
            sample_count = len(samples["prompt"])

        for i in range(sample_count):
            if use_hf_chat_template:
                input_ids, labels, loss_mask, ori_total_len = _encode_openai_example(
                    messages_json=samples["messages"][i],
                    tools_json=samples["tools"][i] if "tools" in samples else None,
                    config=config,
                )
            else:
                if (
                    len(samples["prompt"][i]) % 2 != 1
                    or len(samples["response"][i]) != 1
                ):
                    # Compact form: dumping full payloads made >100MB log files.
                    logger.warning(
                        "Ignore invalid sample: %d prompt msgs, %d response msgs.",
                        len(samples["prompt"][i]),
                        len(samples["response"][i]),
                    )
                    continue

                input_ids, labels, loss_mask, ori_total_len = _encode_supervised_example(
                    prompt=samples["prompt"][i],
                    response=samples["response"][i],
                    system=samples["system"][i],
                    images=samples["images"][i] or [],
                    videos=samples["videos"][i] or [],
                    config=config,
                )

            if not input_ids:
                logger.warning("Ignore sample with no tokens after preprocessing.")
                continue

            if config.enable_discard_sample:
                if ori_total_len > config.sequence_length:
                    continue

            if config.enable_chunkpipe:
                _sample_len = len(input_ids)
                if _sample_len > chunksize:
                    # Long sequence: collect for later splitting
                    long_input_ids.append(input_ids)
                    long_labels.append(labels)
                    long_loss_mask.append(loss_mask)
                else:
                    # Short sequence: collect for later binpacking
                    short_input_ids.append(input_ids)
                    short_labels.append(labels)
                    short_loss_mask.append(loss_mask)
                    short_sample_lens.append(_sample_len)
                    short_len_to_sample_indexs[_sample_len].append(short_index)
                    short_index += 1

            else:
                if not config.packing:
                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["labels"].append(labels)
                    model_inputs["attention_mask"].append([1] * len(input_ids))
                    model_inputs["images"].append(samples["images"][i])
                    model_inputs["videos"].append(samples["videos"][i])
                    if not config.eod_mask_loss:
                        model_inputs["loss_mask"].append(loss_mask)

                else:
                    # TODO: support packing for images/videos
                    assert samples["images"][i] in [None, []] and samples["videos"][i] in [
                        None,
                        [],
                    ], "packing is not supported for images/videos yet."

                    if pad_to_multiple_of > 1:
                        input_ids = _pad_sequence_to_multiple(
                            config, input_ids, pad_to_multiple_of, config.tokenizer.pad
                        )
                        labels = _pad_sequence_to_multiple(
                            config, labels, pad_to_multiple_of, constants.IGNORE_INDEX
                        )
                        loss_mask = _pad_sequence_to_multiple(
                            config, loss_mask, pad_to_multiple_of, 0
                        )

                    # prepare for packing
                    _sample_len = len(input_ids)
                    if _sample_len > config.sequence_length:
                        logger.warning(
                            f"Ignore too long sample with length {_sample_len} > {config.sequence_length}."
                        )
                        continue

                    all_input_ids.append(input_ids)
                    all_labels.append(labels)
                    all_loss_mask.append(loss_mask)
                    all_sampel_lens.append(_sample_len)
                    len_to_sample_indexs[_sample_len].append(index)
                    index += 1

        if not config.packing and not config.enable_chunkpipe:
            return model_inputs

        if config.enable_chunkpipe:
            pad_token_id = config.tokenizer.pad
            ignore_index = config.ignore_index

            # (c) Long sequence splitting: split each long sequence into chunks of chunksize
            for idx in range(len(long_input_ids)):
                chunks = _split_long_sequence(
                    long_input_ids[idx],
                    long_labels[idx],
                    long_loss_mask[idx],
                    chunksize,
                    pad_token_id,
                    ignore_index,
                    mtp_num_layers,
                )
                num_chunks = len(chunks)
                # N_g = total response tokens across all chunks of this source
                # sequence (sum of per-chunk loss masks). Shared by every chunk.
                group_total_tokens = sum(
                    sum(chunk_loss_mask[:chunksize]) for _, _, chunk_loss_mask in chunks
                )
                for chunk_input_ids, chunk_labels, chunk_loss_mask in chunks:
                    model_inputs["input_ids"].append(chunk_input_ids)
                    model_inputs["labels"].append(chunk_labels)
                    model_inputs["attention_mask"].append(
                        [1] * chunksize + [0] * mtp_num_layers
                    )
                    model_inputs["images"].append([])
                    model_inputs["videos"].append([])
                    model_inputs["chunk_group_size"].append(num_chunks)
                    model_inputs["group_total_tokens"].append(group_total_tokens)
                    if not config.eod_mask_loss:
                        model_inputs["loss_mask"].append(chunk_loss_mask)

            # (d) Short sequence binpacking: pack short sequences into bins of chunksize
            knapsacks = _build_knapsacks(short_sample_lens, chunksize)
            for knapsack in knapsacks:
                packed_input_ids, packed_labels, packed_loss_mask, packed_attention_mask = (
                    [], [], [], [],
                )
                for i, length in enumerate(knapsack):
                    idx = short_len_to_sample_indexs[length].pop()
                    packed_input_ids += short_input_ids[idx]
                    # Pre-shift labels and loss_mask per sequence for next-token prediction:
                    # shift left by 1, last position set to IGNORE/0 (end of sequence)
                    packed_labels += short_labels[idx][1:] + [ignore_index]
                    packed_loss_mask += short_loss_mask[idx][1:] + [0]
                    packed_attention_mask += [i + 1] * len(short_input_ids[idx])  # start from 1

                # Pad to chunksize
                padding_len = chunksize - len(packed_input_ids)
                if padding_len > 0:
                    packed_input_ids += [pad_token_id] * padding_len
                    packed_labels += [ignore_index] * padding_len
                    packed_loss_mask += [0] * padding_len
                    packed_attention_mask += [0] * padding_len

                if mtp_num_layers > 0:
                    packed_input_ids += [pad_token_id] * mtp_num_layers
                    packed_labels += [ignore_index] * mtp_num_layers
                    packed_loss_mask += [0] * mtp_num_layers
                    packed_attention_mask += [0] * mtp_num_layers

                model_inputs["input_ids"].append(packed_input_ids)
                model_inputs["labels"].append(packed_labels)
                model_inputs["attention_mask"].append(packed_attention_mask)
                model_inputs["images"].append([])
                model_inputs["videos"].append([])
                model_inputs["chunk_group_size"].append(1)
                # Bin-packed chunk is treated as a single sample; N_g = total
                # response tokens of the base chunk, excluding MTP bridge padding.
                model_inputs["group_total_tokens"].append(
                    sum(packed_loss_mask[:chunksize])
                )
                if not config.eod_mask_loss:
                    model_inputs["loss_mask"].append(packed_loss_mask)

            return model_inputs

        # build packing
        knapsacks = _build_knapsacks(all_sampel_lens, config.sequence_length)
        estimated_computational_load_list = []
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_labels, packed_loss_masks = (
                [],
                [],
                [],
                [],
            )
            # for language model, we use the estimated computational load to sort the batch
            estimated_computational_load = 0

            for i, length in enumerate(knapsack):
                index = len_to_sample_indexs[length].pop()
                # packing
                packed_input_ids += all_input_ids[index]
                estimated_computational_load += len(all_input_ids[index]) ** 2
                packed_labels += all_labels[index]
                packed_loss_masks += all_loss_mask[index]
                packed_attention_masks += [i + 1] * len(
                    all_input_ids[index]
                )  # start from 1

            estimated_computational_load_list.append(estimated_computational_load)
            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["attention_mask"].append(packed_attention_masks)
            # TODO: support images/videos, just placeholder for now
            model_inputs["images"].append([])
            model_inputs["videos"].append([])

            if not config.eod_mask_loss:
                model_inputs["loss_mask"].append(packed_loss_masks)

        if config.sort_batch:
            sorted_indices = sorted(
                range(len(model_inputs["input_ids"])),
                key=lambda i: estimated_computational_load_list[i],
            )
            model_inputs["input_ids"] = [
                model_inputs["input_ids"][i] for i in sorted_indices
            ]
            model_inputs["labels"] = [model_inputs["labels"][i] for i in sorted_indices]
            model_inputs["attention_mask"] = [
                model_inputs["attention_mask"][i] for i in sorted_indices
            ]
            # TODO: add images pixels

            if not config.eod_mask_loss:
                model_inputs["loss_mask"] = [
                    model_inputs["loss_mask"][i] for i in sorted_indices
                ]

        return model_inputs

    return _preprocess_supervised_dataset


def _enc(text):
    """Fake tokenizer: one id (1..49) per character; 0 is the pad id, 1 the eos id."""
    return [ord(ch) % 48 + 2 for ch in text]


class FakeLegacyTemplate:
    mm_plugin = None

    def __init__(self, efficient_eos=False):
        self.efficient_eos = efficient_eos

    def encode_multiturn(self, tokenizer, messages, system):
        pairs = []
        for k in range(0, len(messages) - 1, 2):
            source = _enc(messages[k]["content"])
            if k == 0 and system:
                source = _enc(system) + source
            pairs.append((source, _enc(messages[k + 1]["content"])))
        return pairs


class FakeOpenAITemplate(FakeHFChatTemplate):
    def encode_openai(self, tokenizer, messages, tools, train_on_prompt, history_mask_loss, ignore_index, max_length):
        ids, labels, mask = [], [], []
        for message in messages:
            part = _enc(message["content"])
            train = train_on_prompt or message["role"] == "assistant"
            ids += part
            labels += part if train else [ignore_index] * len(part)
            mask += [1 if train else 0] * len(part)
        if tools:
            ids = ids[:-1]
            labels = labels[:-1]
            mask = mask[:-1]
        return ids[:max_length], labels[:max_length], mask[:max_length], len(ids)


def _config(template, **overrides):
    values = dict(
        chat_template=template,
        tokenizer=SimpleNamespace(pad=0, eos=1, padding_side="right"),
        processor=None,
        sequence_length=24,
        history_mask_loss=False,
        train_on_prompt=False,
        ignore_index=constants.IGNORE_INDEX,
        eod_mask_loss=False,
        enable_discard_sample=False,
        packing=False,
        sort_batch=False,
        context_parallel_size=1,
        enable_chunkpipe=False,
        chunksize=8,
        mtp_num_layers=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# (prompt text, answer text) pairs: lengths 0 (dropped), short, medium, longer than the chunk size.
PAIRS = [("q" * 3, "a" * 2), ("", ""), ("x" * 6, "y" * 7), ("p" * 9, "r" * 11), ("m" * 2, "n" * 1),
         ("u" * 5, "v" * 5), ("s" * 20, "t" * 18), ("k" * 4, "l" * 3), ("w" * 7, "z" * 6)]


def _legacy_samples(images=None):
    count = len(PAIRS) + 1
    prompts = [[{"role": "user", "content": p}] for p, _ in PAIRS]
    responses = [[{"role": "assistant", "content": a}] for _, a in PAIRS]
    prompts.append([{"role": "user", "content": "bad"}] * 2)  # even prompt count: invalid, skipped
    responses.append([{"role": "assistant", "content": "bad"}])
    return {
        "prompt": prompts,
        "response": responses,
        "system": ["sys" if k % 3 == 0 else None for k in range(count)],
        "images": images if images is not None else [None, [], None, [], None, [], None, [], None, []],
        "videos": [None, [], None, [], None, [], None, [], None, []],
    }


def _openai_samples():
    count = len(PAIRS)
    return {
        "messages": [
            json.dumps([{"role": "user", "content": p}, {"role": "assistant", "content": a}])
            if k % 2 else [{"role": "user", "content": p}, {"role": "assistant", "content": a}]
            for k, (p, a) in enumerate(PAIRS)
        ],
        "tools": [None, "", None, json.dumps([{"name": "f"}]), None, None, None, None, None],
        "images": [[] for _ in range(count)],
        "videos": [[] for _ in range(count)],
    }


class SftPreprocessSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_namespace()
        cls.old = staticmethod(_frozen_preprocess(cls.ns))
        cls.new = staticmethod(cls.ns["_preprocess_supervised_dataset"])

    def assert_same(self, samples, config, min_rows=1):
        expected = self.old(samples, config)
        actual = self.new(samples, config)
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            self.assertEqual(expected[key], actual[key], key)
        self.assertGreaterEqual(len(actual["input_ids"]), min_rows)
        return actual

    def test_plain(self):
        images = [["a.png"], [], None, [], None, ["b.png", "c.png"], None, [], None, []]
        for efficient_eos in (False, True):
            for eod_mask_loss in (False, True):
                for history_mask_loss in (False, True):
                    with self.subTest(efficient_eos=efficient_eos, eod=eod_mask_loss, hist=history_mask_loss):
                        config = _config(
                            FakeLegacyTemplate(efficient_eos),
                            eod_mask_loss=eod_mask_loss,
                            history_mask_loss=history_mask_loss,
                        )
                        out = self.assert_same(_legacy_samples(images), config, min_rows=7)
                        self.assertEqual(out["images"][0], ["a.png"])

    def test_plain_discard_overlong(self):
        config = _config(FakeLegacyTemplate(), sequence_length=20, enable_discard_sample=True)
        out = self.assert_same(_legacy_samples(), config)
        self.assertEqual(len(out["input_ids"]), 6)

    def test_packing(self):
        for padding_side in ("right", "left"):
            for cp, seq_len in ((1, 24), (2, 24), (2, 15)):  # (2, 15): padded sample exceeds seq_len
                for sort_batch in (False, True):
                    for eod_mask_loss in (False, True):
                        with self.subTest(side=padding_side, cp=cp, seq=seq_len, sort=sort_batch, eod=eod_mask_loss):
                            config = _config(
                                FakeLegacyTemplate(), packing=True, sort_batch=sort_batch, eod_mask_loss=eod_mask_loss,
                                context_parallel_size=cp, sequence_length=seq_len,
                                tokenizer=SimpleNamespace(pad=0, eos=1, padding_side=padding_side),
                            )
                            self.assert_same(_legacy_samples(), config)

    def test_packing_rejects_images(self):
        config = _config(FakeLegacyTemplate(), packing=True)
        samples = _legacy_samples(images=[["a.png"]] + [None] * 9)
        for run in (self.old, self.new):
            with self.assertRaisesRegex(AssertionError, "packing is not supported"):
                run(samples, config)

    def test_chunkpipe(self):
        for mtp in (0, 2):
            for eod_mask_loss in (False, True):
                for packing in (False, True):  # chunkpipe wins over packing
                    with self.subTest(mtp=mtp, eod=eod_mask_loss, packing=packing):
                        config = _config(
                            FakeLegacyTemplate(), enable_chunkpipe=True, packing=packing, mtp_num_layers=mtp,
                            eod_mask_loss=eod_mask_loss, sequence_length=64, sort_batch=True,
                        )
                        out = self.assert_same(_legacy_samples(), config, min_rows=4)
                        self.assertIn(1, out["chunk_group_size"])
                        self.assertTrue(any(n > 1 for n in out["chunk_group_size"]))
                        self.assertTrue(all(x == [] for x in out["images"] + out["videos"]))

    def test_openai_plain_packing_chunkpipe(self):
        template = FakeOpenAITemplate()
        variants = {
            "plain": dict(),
            "plain_discard": dict(enable_discard_sample=True, sequence_length=20),
            "packing": dict(packing=True, sort_batch=True),
            "chunkpipe": dict(enable_chunkpipe=True, mtp_num_layers=1, sequence_length=64),
        }
        for name, overrides in variants.items():
            with self.subTest(name):
                self.assert_same(_openai_samples(), _config(template, **overrides))

    def test_missing_columns_are_rejected(self):
        for run in (self.old, self.new):
            with self.assertRaisesRegex(ValueError, "HFChatTemplate requires"):
                run({"prompt": []}, _config(FakeOpenAITemplate()))
            with self.assertRaisesRegex(ValueError, "Legacy ChatTemplate"):
                run({"messages": []}, _config(FakeLegacyTemplate()))


if __name__ == "__main__":
    unittest.main()
