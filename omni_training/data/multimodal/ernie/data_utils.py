# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
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
from io import BytesIO

import xxhash
from PIL import Image


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
