# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from ERNIE (https://github.com/PaddlePaddle/ERNIE/)
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
utils for data processor
"""

import base64
import math
import torch
from io import BytesIO
import numpy as np

import xxhash
from PIL import Image
from transformers import AutoProcessor


def get_text_token_num(tokenizer, text: str):
    """text tokenize and count"""
    return len(tokenizer.encode(text)["input_ids"])


def get_uniq_id(text):
    """text hash"""
    return xxhash.xxh32_intdigest(text)

def merge_list(lists):
    """merge multi list to one list

    Args:
        lists (list[list]): [[], [], ...]

    Returns:
        list: one list
    """
    new_list = lists[0]
    for one in lists[1:]:
        new_list.extend(one)
    return new_list


class ErnieTensorDataset(torch.utils.data.Dataset):
    """
    Simple tensor dataset for ERNIE-VL that loads pre-processed .npy files.

    Reads tensor data from numpy files listed in metadata_path and converts
    them to PyTorch tensors. Uses random sampling with fixed seed for reproducibility.

    Args:
        args: Config with seed and hf_tokenizer_path
        metadata_path: Path to file listing all .npy file paths
        steps_per_epoch: Total number of training steps
    """
    def __init__(self, args, metadata_path, steps_per_epoch=0):
        self.manual_seed = args.seed
        self.steps_per_epoch = steps_per_epoch
        self.processor = AutoProcessor.from_pretrained(args.hf_tokenizer_path,  trust_remote_code=True)
        self.file_names = []

        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.file_names.append(line.strip())

    def __getitem__(self, index):
        seed = (self.manual_seed + index) % 2**32
        numpy_random_state = np.random.RandomState(seed=seed)
        data_id = numpy_random_state.randint(0, self.steps_per_epoch)
        data_id = data_id % len(self.file_names)
        data_name = self.file_names[data_id]
        data = np.load(data_name)
        data_item = {
            "images": torch.from_numpy(data["images"]),
            "input_ids": torch.from_numpy(data["input_ids"]),
            "token_type_ids": torch.from_numpy(data["token_type_ids"])[:, :-1],
            "position_ids": torch.from_numpy(data["position_ids"]),
            "grid_thw": torch.from_numpy(data["grid_thw"]),
            "image_type_ids": torch.from_numpy(data["image_type_ids"]),
            "labels": torch.from_numpy(data["labels"])
        }
        return data_item

    def __len__(self):
        return self.steps_per_epoch