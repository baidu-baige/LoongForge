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
from functools import partial
from collections import defaultdict

from typing import TYPE_CHECKING, Union, Dict, List, Any, Sequence, Optional, Tuple
import datasets
from datasets import Dataset, IterableDataset

from loongforge.utils import constants

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from .sft_dataset import SFTDatasetConfig


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

    for i in range(len(samples["prompt"])):
        if len(samples["prompt"][i]) % 2 != 1 or len(samples["response"][i]) != 1:
            logger.warning(
                f"Ignore invalid sample, prompt: {samples['prompt'][i]}, response: {samples['response'][i]}"
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

        if config.enable_discard_sample:
            if ori_total_len > config.sequence_length:
                continue

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

    if not config.packing:
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
        if config.sort_batch and not config.packing:
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

    dataset = dataset.map(
        partial(_preprocess_supervised_dataset, config=config),
        batched=True,
        remove_columns=columns,
        features=features,
        batch_size=config.packing_buffer_size,
        **kwargs,
    )

    return dataset
