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


"""Preprocess data to unified format"""

import os
import logging
from functools import partial

from typing import TYPE_CHECKING, Union, Dict, List, Any, Sequence

from datasets import Features, Value

from loongforge.utils.constants import SFTDataFormats, DataRoles


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from .sft_dataset import (
        SFTDataFormat,
        SFTDatasetConfig,
        AlpacaColumns,
        ShareGPTColumns,
        ShareGPTTags,
    )


logger = logging.getLogger(__name__)


def _convert_path(images: Sequence[str], dataset_dir: str):
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    """
    if len(images) == 0:
        return []

    images = images[:]
    for i in range(len(images)):
        fullpath = os.path.join(dataset_dir, images[i])
        if os.path.isfile(fullpath):
            images[i] = fullpath

    return images


def _convert_alpaca(samples: Dict[str, List[Any]], alpaca_columns: "AlpacaColumns"):
    outputs = {
        "prompt": [],
        "response": [],
        "system": [],
        "d_len": [],
        "videos": [],
        "images": [],
    }
    for i in range(len(samples[alpaca_columns.prompt])):
        prompt = []
        d_len = 0
        # preprocess history
        if alpaca_columns.history and isinstance(
            samples[alpaca_columns.history][i], list
        ):
            for history_prompt, history_response in samples[alpaca_columns.history][i]:
                prompt.extend(
                    [
                        {"role": DataRoles.USER, "content": history_prompt},
                        {"role": DataRoles.ASSISTANT, "content": history_response},
                    ]
                )
                d_len += len(history_prompt) + len(history_response)
        # preprocess prompt & query
        _content = []
        for col in [alpaca_columns.prompt, alpaca_columns.query]:
            if col and samples[col][i]:
                _content.append(samples[col][i])
                d_len += len(samples[col][i])

        prompt.append({"role": DataRoles.USER, "content": "\n".join(_content)})

        # preprocess response
        response = []
        if alpaca_columns.response:
            resp = samples[alpaca_columns.response][i]
            if isinstance(resp, list):
                response = [
                    {"role": DataRoles.ASSISTANT, "content": content}
                    for content in resp
                ]
                for content in resp:
                    d_len += len(content)
            elif isinstance(resp, str):
                response = [{"role": DataRoles.ASSISTANT, "content": resp}]
                d_len += len(resp)

        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append(
            samples[alpaca_columns.system][i] if alpaca_columns.system else ""
        )
        outputs["d_len"].append(d_len)
        outputs["videos"].append([])  # TODO: support videos
        outputs["images"].append([])  # TODO: support images

    return outputs


def _convert_sharegpt(
    samples: Dict[str, Any],
    sharegpt_columns: "ShareGPTColumns",
    sharegpt_tags: "ShareGPTTags",
    dataset_dir: str,
):
    """
    Converts sharegpt format dataset to the standard format.
    """
    outputs = {
        "prompt": [],
        "response": [],
        "system": [],
        "d_len": [],
        "videos": [],
        "images": [],
    }
    convert_path = partial(_convert_path, dataset_dir=dataset_dir)
    # custom role name => standard role name
    tag_mapping = {
        sharegpt_tags.user_tag: DataRoles.USER,
        sharegpt_tags.assistant_tag: DataRoles.ASSISTANT,
        sharegpt_tags.observation_tag: DataRoles.OBSERVATION,
        sharegpt_tags.function_tag: DataRoles.FUNCTION,
        sharegpt_tags.system_tag: DataRoles.SYSTEM,
    }

    accept_tags = (
        (sharegpt_tags.user_tag, sharegpt_tags.observation_tag),
        (sharegpt_tags.assistant_tag, sharegpt_tags.function_tag),
    )

    for i, messages in enumerate(samples[sharegpt_columns.messages]):

        system_message = next(
            (
                msg
                for msg in messages
                if msg[sharegpt_tags.role_tag] == sharegpt_tags.system_tag
            ),
            None,
        )

        if sharegpt_tags.system_tag is not None and system_message is not None:
            # if message contain system
            system = system_message[sharegpt_tags.content_tag]
            # remove system from messages
            messages.remove(system_message)
        else:
            # try use system from global column
            system = (
                samples[sharegpt_columns.system][i] if sharegpt_columns.system else ""
            )

        aligned_messages = []
        invalid_data = False

        for turn_idx, message in enumerate(messages):
            if message[sharegpt_tags.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning(f"Invalid role tag in: {messages}, skipping.")
                invalid_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[sharegpt_tags.role_tag]],
                    "content": message[sharegpt_tags.content_tag],
                }
            )

        if len(aligned_messages) % 2 != 0:
            logger.warning(f"Invalid number of turns in: {messages}, skipping.")
            invalid_data = True

        if invalid_data:
            aligned_messages = []

        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]
        d_len = sum(len(msg["content"]) for msg in aligned_messages)
        videos = samples[sharegpt_columns.videos][i] if sharegpt_columns.videos else []
        images = samples[sharegpt_columns.images][i] if sharegpt_columns.images else []

        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append(system)
        outputs["d_len"].append(d_len)
        outputs["videos"].append(convert_path(videos))
        outputs["images"].append(convert_path(images))
    return outputs


def convert_to_unified_format(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_path: str,
    data_format: "SFTDataFormat",
    config: "SFTDatasetConfig",
    load_from_cache_file: bool = False,
) -> Union["Dataset", "IterableDataset"]:
    """Convert the dataset to the unified format."""
    if data_format.format == SFTDataFormats.ALPACA:
        convert_func = partial(_convert_alpaca, alpaca_columns=data_format.columns)
    elif data_format.format == SFTDataFormats.SHAREGPT:
        convert_func = partial(
            _convert_sharegpt,
            sharegpt_columns=data_format.columns,
            sharegpt_tags=data_format.tags,
            dataset_dir=os.path.dirname(dataset_path),
        )
    else:
        raise NotImplementedError()

    column_names = [
        col for col in next(iter(dataset)).keys() if col not in ["images", "videos"]
    ]

    features = Features(
        {
            "prompt": [
                {"role": Value(dtype="string"), "content": Value(dtype="string")}
            ],
            "response": [
                {"role": Value(dtype="string"), "content": Value(dtype="string")}
            ],
            "system": Value(dtype="string"),
            "d_len": Value(dtype="int64"),
            "videos": [Value(dtype="string")],
            "images": [Value(dtype="string")],
        }
    )

    kwargs = {}
    if not config.streaming:
        kwargs = dict(
            num_proc=config.num_preprocess_workers,
            load_from_cache_file=load_from_cache_file,
            desc="Converting dataset to unified format",
        )

    return dataset.map(
        convert_func,
        batched=True,
        remove_columns=column_names,
        features=features,
        **kwargs,
    )
