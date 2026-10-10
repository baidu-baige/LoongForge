# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory).
# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocess the sft dataset."""

import logging
import bisect
import json
from functools import partial
from collections import defaultdict

from typing import TYPE_CHECKING, Union, Dict, List, Any, Sequence, Optional, Tuple
import datasets
from datasets import Dataset, IterableDataset

from loongforge import constants
from loongforge.chat_templates.hf import HFChatTemplate

if TYPE_CHECKING:
    from .sft_dataset_config import SFTDatasetConfig


logger = logging.getLogger(__name__)


def _infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    """
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def _encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    images: Sequence[str],
    videos: Sequence[str],
    config: "SFTDatasetConfig",
) -> Tuple[List[int], List[int], List[int], List[int], List[int], int]:
    """Preprocess single sample"""

    if config.chat_template.mm_plugin is not None:
        messages, _ = config.chat_template.mm_plugin.process_messages(
            prompt + response, images, videos, config.processor
        )
    else:
        messages = prompt + response
    input_ids, labels, loss_mask = [], [], []

    encode_pairs = config.chat_template.encode_multiturn(
        tokenizer=config.tokenizer,
        messages=messages,
        system=system,
    )

    total_len = 1 if config.chat_template.efficient_eos else 0

    ori_total_len = total_len
    for turn_idx, (source_ids, target_ids) in enumerate(encode_pairs):
        ori_total_len = len(source_ids) + len(target_ids) + ori_total_len

    if config.history_mask_loss:
        encode_pairs = encode_pairs[::-1]  # high priority for last turns

    for turn_idx, (source_ids, target_ids) in enumerate(encode_pairs):
        if total_len >= config.sequence_length:
            break

        source_len, target_len = _infer_seqlen(
            len(source_ids), len(target_ids), config.sequence_length - total_len
        )
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_len += source_len + target_len

        if config.train_on_prompt:
            source_label = source_ids
        elif turn_idx != 0 and config.chat_template.efficient_eos:
            # refer to https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/fine-tune.py#L81
            source_label = [config.tokenizer.eos] + [config.ignore_index] * (
                source_len - 1
            )
        else:
            source_label = [config.ignore_index] * source_len

        if config.history_mask_loss and turn_idx != 0:
            # train on the last turn only
            target_label = [config.ignore_index] * target_len
        else:
            # turn_idx == 0 is the last turn
            target_label = target_ids

        if config.history_mask_loss:
            # reversed order
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
            loss_mask = [
                0 if t == config.ignore_index else 1
                for t in (source_label + target_label)
            ] + loss_mask
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label
            loss_mask += [
                0 if t == config.ignore_index else 1
                for t in (source_label + target_label)
            ]

    if config.chat_template.efficient_eos:
        # for efficient_eos, we need to add eos token to the end of the last turn
        input_ids += [config.tokenizer.eos]
        labels += [config.tokenizer.eos]
        loss_mask += [1]

    return input_ids, labels, loss_mask, ori_total_len


def _encode_openai_example(
    messages_json: str,
    tools_json: Optional[str],
    config: "SFTDatasetConfig",
) -> Tuple[List[int], List[int], List[int], int]:
    """Preprocess a single OpenAI-style messages/tools sample."""
    messages = (
        json.loads(messages_json)
        if isinstance(messages_json, str)
        else messages_json
    )
    if tools_json in (None, ""):
        tools = None
    else:
        tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
    if not isinstance(messages, list):
        raise ValueError(
            f"OpenAI-style sample messages must be a list, got {type(messages)}"
        )

    input_ids, labels, loss_mask, ori_total_len = config.chat_template.encode_openai(
        tokenizer=config.tokenizer,
        messages=messages,
        tools=tools,
        train_on_prompt=config.train_on_prompt,
        history_mask_loss=config.history_mask_loss,
        ignore_index=config.ignore_index,
        max_length=config.sequence_length,
    )
    return input_ids, labels, loss_mask, ori_total_len


def _build_knapsacks(numbers: List[int], capacity: int) -> List[List[int]]:
    """
    An efficient greedy algorithm with binary search for the knapsack problem.
    """
    numbers.sort()
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        if numbers[0] > capacity:
            # no more numbers can be added
            break

        while remaining_capacity > 0:
            index = bisect.bisect_right(numbers, remaining_capacity)
            if index == 0:
                break

            remaining_capacity -= numbers[index - 1]
            current_knapsack.append(numbers.pop(index - 1))

        knapsacks.append(current_knapsack)

    return knapsacks


def _pad_sequence_to_multiple(config, sequence, multiple_of, pad_token_id):
    padding_length = (multiple_of - len(sequence) % multiple_of) % multiple_of
    if config.tokenizer.padding_side == "right":
        return sequence + [pad_token_id] * padding_length
    return [pad_token_id] * padding_length + sequence


def _split_long_sequence(
    input_ids: List[int],
    labels: List[int],
    loss_mask: List[int],
    chunksize: int,
    pad_token_id: int,
    ignore_index: int,
    mtp_num_layers: int = 0,
) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Split a long sequence (len > chunksize) into chunks with a base length of chunksize.
    Labels and loss_mask are pre-shifted to next-token prediction format:
      - Non-final chunks: labels[i] = original_labels[start+i+1], covering the chunk boundary.
      - Final chunk: last label is IGNORE (no next token to predict).
    When MTP is enabled, append mtp_num_layers bridge tokens after the base chunk.
    """
    chunks = []
    seq_len = len(input_ids)
    num_chunks = (seq_len + chunksize - 1) // chunksize

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunksize
        end = min(start + chunksize, seq_len)
        is_final_chunk = (chunk_idx == num_chunks - 1)

        chunk_input_ids = input_ids[start:end]

        # Pre-shift labels and loss_mask for next-token prediction
        if not is_final_chunk:
            # Non-final chunk: shift by 1, end+1 naturally includes the boundary token
            chunk_labels = labels[start + 1 : end + 1]
            chunk_loss_mask = loss_mask[start + 1 : end + 1]
        else:
            # Final chunk: shift by 1, append IGNORE/0 at the end
            chunk_labels = labels[start + 1 : end] + [ignore_index]
            chunk_loss_mask = loss_mask[start + 1 : end] + [0]

        # Pad the base chunk if needed
        padding_len = chunksize - len(chunk_input_ids)
        if padding_len > 0:
            chunk_input_ids = chunk_input_ids + [pad_token_id] * padding_len
            chunk_labels = chunk_labels + [ignore_index] * padding_len
            chunk_loss_mask = chunk_loss_mask + [0] * padding_len

        if mtp_num_layers > 0:
            if not is_final_chunk:
                bridge_end = min(end + mtp_num_layers, seq_len)
                bridge_input_ids = input_ids[end:bridge_end]

                bridge_label_end = min(end + 1 + mtp_num_layers, seq_len)
                bridge_labels = labels[end + 1:bridge_label_end]
                bridge_loss_mask = loss_mask[end + 1:bridge_label_end]

                bridge_padding_len = mtp_num_layers - len(bridge_input_ids)
                if bridge_padding_len > 0:
                    bridge_input_ids += [pad_token_id] * bridge_padding_len

                bridge_label_padding_len = mtp_num_layers - len(bridge_labels)
                if bridge_label_padding_len > 0:
                    bridge_labels += [ignore_index] * bridge_label_padding_len
                    bridge_loss_mask += [0] * bridge_label_padding_len
            else:
                bridge_input_ids = [pad_token_id] * mtp_num_layers
                bridge_labels = [ignore_index] * mtp_num_layers
                bridge_loss_mask = [0] * mtp_num_layers

            chunk_input_ids = chunk_input_ids + bridge_input_ids
            chunk_labels = chunk_labels + bridge_labels
            chunk_loss_mask = chunk_loss_mask + bridge_loss_mask

        chunks.append((chunk_input_ids, chunk_labels, chunk_loss_mask))

    return chunks


def _encode_one(
    samples: Dict[str, List[Any]],
    i: int,
    config: "SFTDatasetConfig",
    use_hf_chat_template: bool,
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    """Encode sample ``i``. Return (input_ids, labels, loss_mask), or None if it is skipped."""
    if use_hf_chat_template:
        input_ids, labels, loss_mask, ori_total_len = _encode_openai_example(
            messages_json=samples["messages"][i],
            tools_json=samples["tools"][i] if "tools" in samples else None,
            config=config,
        )
    else:
        if len(samples["prompt"][i]) % 2 != 1 or len(samples["response"][i]) != 1:
            # Compact form: dumping full payloads made >100MB log files.
            logger.warning(
                "Ignore invalid sample: %d prompt msgs, %d response msgs.",
                len(samples["prompt"][i]),
                len(samples["response"][i]),
            )
            return None

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
        return None

    if config.enable_discard_sample and ori_total_len > config.sequence_length:
        return None

    return input_ids, labels, loss_mask


def _append_row(
    model_inputs: Dict[str, List[Any]],
    config: "SFTDatasetConfig",
    input_ids: List[int],
    labels: List[int],
    attention_mask: List[int],
    loss_mask: List[int],
    images: Any,
    videos: Any,
    **extra: Any,
) -> None:
    """Append one output row. ``extra`` holds the chunkpipe-only columns."""
    model_inputs["input_ids"].append(input_ids)
    model_inputs["labels"].append(labels)
    model_inputs["attention_mask"].append(attention_mask)
    model_inputs["images"].append(images)
    model_inputs["videos"].append(videos)
    if not config.eod_mask_loss:
        model_inputs["loss_mask"].append(loss_mask)
    for key, value in extra.items():
        model_inputs[key].append(value)


def _concat_knapsack(
    knapsack: List[int],
    seqs: Tuple[List[List[int]], List[List[int]], List[List[int]]],
    len_to_indexs: Dict[int, List[int]],
    *,
    shift: bool,
    ignore_index: Optional[int] = None,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Concatenate the sequences of one knapsack.

    Return (input_ids, labels, loss_mask, attention_mask); attention_mask numbers the
    sequences from 1. With ``shift`` the labels and loss_mask are moved left by one
    position per sequence (next-token prediction format) and ``ignore_index`` fills the last one.
    """
    all_input_ids, all_labels, all_loss_mask = seqs
    packed_input_ids, packed_labels, packed_loss_mask, packed_attention_mask = [], [], [], []
    for i, length in enumerate(knapsack):
        idx = len_to_indexs[length].pop()
        packed_input_ids += all_input_ids[idx]
        if shift:
            packed_labels += all_labels[idx][1:] + [ignore_index]
            packed_loss_mask += all_loss_mask[idx][1:] + [0]
        else:
            packed_labels += all_labels[idx]
            packed_loss_mask += all_loss_mask[idx]
        packed_attention_mask += [i + 1] * len(all_input_ids[idx])  # start from 1
    return packed_input_ids, packed_labels, packed_loss_mask, packed_attention_mask


def _emit_chunkpipe(
    model_inputs: Dict[str, List[Any]],
    config: "SFTDatasetConfig",
    long_seqs: Tuple[List[List[int]], List[List[int]], List[List[int]]],
    short_seqs: Tuple[List[List[int]], List[List[int]], List[List[int]]],
) -> None:
    """Write chunkpipe rows: split long sequences into chunks, binpack short ones."""
    chunksize = config.chunksize
    mtp_num_layers = getattr(config, "mtp_num_layers", 0) or 0
    pad_token_id = config.tokenizer.pad
    ignore_index = config.ignore_index

    # (c) Long sequence splitting: split each long sequence into chunks of chunksize
    for input_ids, labels, loss_mask in zip(*long_seqs):
        chunks = _split_long_sequence(
            input_ids, labels, loss_mask, chunksize, pad_token_id, ignore_index, mtp_num_layers
        )
        # N_g = total response tokens across all chunks of this source
        # sequence (sum of per-chunk loss masks). Shared by every chunk.
        group_total_tokens = sum(
            sum(chunk_loss_mask[:chunksize]) for _, _, chunk_loss_mask in chunks
        )
        for chunk_input_ids, chunk_labels, chunk_loss_mask in chunks:
            _append_row(
                model_inputs, config, chunk_input_ids, chunk_labels,
                [1] * chunksize + [0] * mtp_num_layers, chunk_loss_mask, [], [],
                chunk_group_size=len(chunks), group_total_tokens=group_total_tokens,
            )

    # (d) Short sequence binpacking: pack short sequences into bins of chunksize
    short_sample_lens = [len(input_ids) for input_ids in short_seqs[0]]
    short_len_to_sample_indexs = defaultdict(list)
    for index, length in enumerate(short_sample_lens):
        short_len_to_sample_indexs[length].append(index)
    for knapsack in _build_knapsacks(short_sample_lens, chunksize):
        packed_input_ids, packed_labels, packed_loss_mask, packed_attention_mask = _concat_knapsack(
            knapsack, short_seqs, short_len_to_sample_indexs, shift=True, ignore_index=ignore_index
        )

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

        # Bin-packed chunk is treated as a single sample; N_g = total
        # response tokens of the base chunk, excluding MTP bridge padding.
        _append_row(
            model_inputs, config, packed_input_ids, packed_labels, packed_attention_mask,
            packed_loss_mask, [], [],
            chunk_group_size=1, group_total_tokens=sum(packed_loss_mask[:chunksize]),
        )


def _emit_packing(
    model_inputs: Dict[str, List[Any]],
    config: "SFTDatasetConfig",
    seqs: Tuple[List[List[int]], List[List[int]], List[List[int]]],
) -> None:
    """Write packing rows: binpack the sequences into bins of sequence_length."""
    all_sample_lens = [len(input_ids) for input_ids in seqs[0]]
    len_to_sample_indexs = defaultdict(list)
    for index, length in enumerate(all_sample_lens):
        len_to_sample_indexs[length].append(index)
    estimated_computational_load_list = []
    for knapsack in _build_knapsacks(all_sample_lens, config.sequence_length):
        packed_input_ids, packed_labels, packed_loss_mask, packed_attention_mask = _concat_knapsack(
            knapsack, seqs, len_to_sample_indexs, shift=False
        )
        # for language model, we use the estimated computational load to sort the batch
        estimated_computational_load_list.append(sum(length**2 for length in knapsack))
        # TODO: support images/videos, just placeholder for now
        _append_row(
            model_inputs, config, packed_input_ids, packed_labels, packed_attention_mask,
            packed_loss_mask, [], [],
        )

    if config.sort_batch:
        sorted_indices = sorted(
            range(len(model_inputs["input_ids"])),
            key=lambda i: estimated_computational_load_list[i],
        )
        # TODO: add images pixels
        for key in ("input_ids", "labels", "attention_mask", "loss_mask"):
            if key in model_inputs:
                model_inputs[key] = [model_inputs[key][i] for i in sorted_indices]


def _preprocess_supervised_dataset(
    samples: Dict[str, List[Any]],
    config: "SFTDatasetConfig",
) -> Dict[str, List[List[int]]]:
    """
    Preprocess supervised dataset.
    """
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

    # (input_ids, labels, loss_mask) lists collected for chunkpipe or packing.
    # Chunkpipe: long sequences (len > chunksize) are split into chunks,
    # short ones are binpacked. Packing only uses ``short_seqs``.
    long_seqs = ([], [], [])
    short_seqs = ([], [], [])

    # When using context parallel, sequence is split by CP size
    pad_to_multiple_of = (
        2 * config.context_parallel_size
        if (config.context_parallel_size and config.context_parallel_size > 1)
        else 1
    )

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
        encoded = _encode_one(samples, i, config, use_hf_chat_template)
        if encoded is None:
            continue
        input_ids, labels, loss_mask = encoded

        if config.enable_chunkpipe:
            seqs = long_seqs if len(input_ids) > config.chunksize else short_seqs
        elif not config.packing:
            _append_row(
                model_inputs, config, input_ids, labels, [1] * len(input_ids), loss_mask,
                samples["images"][i], samples["videos"][i],
            )
            continue
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

            if len(input_ids) > config.sequence_length:
                logger.warning(
                    f"Ignore too long sample with length {len(input_ids)} > {config.sequence_length}."
                )
                continue
            seqs = short_seqs

        for buffer, seq in zip(seqs, (input_ids, labels, loss_mask)):
            buffer.append(seq)

    if config.enable_chunkpipe:
        _emit_chunkpipe(model_inputs, config, long_seqs, short_seqs)
    elif config.packing:
        _emit_packing(model_inputs, config, short_seqs)

    return model_inputs


def _chunked_sort(dataset: List[Dict], chunk_size: int) -> List[Dict]:
    """Sort the dataset in chunks and merge them."""
    import heapq

    chunks = [dataset[i : i + chunk_size] for i in range(0, len(dataset), chunk_size)]
    sorted_chunks = [sorted(chunk, key=lambda x: x["d_len"]) for chunk in chunks]
    return list(heapq.merge(*sorted_chunks, key=lambda x: x["d_len"]))


def convert_to_tokenized_data(
    dataset: Union["Dataset", "IterableDataset"],
    config: "SFTDatasetConfig",
    load_from_cache_file: bool = False,
) -> Union["Dataset", "IterableDataset"]:
    """Convert the dataset to the tokenized form."""
    columns = [
        col for col in next(iter(dataset)).keys() if col not in ["images", "videos"]
    ]

    kwargs = {}
    if not config.streaming:
        kwargs = dict(
            num_proc=config.num_preprocess_workers,
            load_from_cache_file=load_from_cache_file,
            desc="Converting dataset to tokenized data",
        )
        if config.sort_batch and not config.packing and not config.enable_chunkpipe:
            dataset_list = list(dataset)
            # Sort the dataset by length of samples
            sorted_dataset = _chunked_sort(dataset_list, chunk_size=100000)
            dataset = Dataset.from_list(sorted_dataset)
    # The data in the dataset varies in length,
    # which may lead to inconsistent types being inferred (such as int8, int32),
    # resulting in the error "The features can't be aligned." ,
    # Therefore, it is necessary to specify the output type through features to avoid automatic type inference.
    features = datasets.Features()
    features["input_ids"] = datasets.Sequence(
        feature=datasets.Value(dtype="int64", id=None), length=-1, id=None
    )
    features["labels"] = datasets.Sequence(
        feature=datasets.Value(dtype="int64", id=None), length=-1, id=None
    )
    features["attention_mask"] = datasets.Sequence(
        feature=datasets.Value(dtype="int64", id=None), length=-1, id=None
    )
    if not config.eod_mask_loss:
        features["loss_mask"] = datasets.Sequence(
            feature=datasets.Value(dtype="int64", id=None), length=-1, id=None
        )
    features["images"] = datasets.Sequence(
        datasets.Value(dtype="string", id=None), length=-1, id=None
    )
    features["videos"] = datasets.Sequence(
        datasets.Value(dtype="string", id=None), length=-1, id=None
    )
    if config.enable_chunkpipe:
        features["chunk_group_size"] = datasets.Value(dtype="int64", id=None)
        features["group_total_tokens"] = datasets.Value(dtype="int64", id=None)

    dataset = dataset.map(
        partial(_preprocess_supervised_dataset, config=config),
        batched=True,
        remove_columns=columns,
        features=features,
        batch_size=config.packing_buffer_size,
        **kwargs,
    )

    return dataset
