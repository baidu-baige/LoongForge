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
Ernie45VLTokenizer — HuggingFace transformers implementation.
"""

import random
import os
import re
import logging
from shutil import copyfile
from typing import Dict, Optional, Tuple, List

import numpy as np
import sentencepiece as spm

# ── transformers replacements ─────────────────────────────────────────────────
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TensorType

TextInput = str
logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────────────────────────────────────

coor_num = 1001
NOT_FOUND_TOKEN_ID = -101
NUM_IMAGE_SPECIAL_TOKEN = 2048
NUM_AUDIO_SPECIAL_TOKEN = 1024
SFT_IMAGE_START_TOKEN = "<|IMAGE_START|>"
SFT_IMAGE_END_TOKEN = "<|IMAGE_END|>"
SFT_VIDEO_START_TOKEN = "<|VIDEO_START|>"
SFT_VIDEO_END_TOKEN = "<|VIDEO_END|>"
SFT_ASR_START_TOKEN = "<|ASR_START|>"
SFT_ASR_END_TOKEN = "<|ASR_END|>"

special_tokens_info = {
    "image_placeholder": "<|IMAGE_PLACEHOLDER|>",
    "loc_coor": [f"<|LOC_{i}|>" for i in range(coor_num)],
    "loc_begin_end": ["<|LOC_BEGIN|>", "<|LOC_END|>", "<|LOC_SEP|>"],
    "image_begin_end": ["<|BOI|>", "<|EOI|>"],
    "video_begin_end": ["<|BOV|>", "<|EOV|>"],
    "sft_video_begin_end": [SFT_VIDEO_START_TOKEN, SFT_VIDEO_END_TOKEN],
}


class Ernie45VLTokenizer(PreTrainedTokenizer):
    """
    Ernie45VLTokenizer — pure transformers implementation.
    """

    resource_files_names = {
        "vocab_file": "tokenizer.model",
    }
    pretrained_resource_files_map = {"vocab_file": {"ernie-bot-10b": None}}
    pretrained_init_configuration = {"ernie-bot-10b": {}}
    model_input_names = ["input_ids", "position_ids", "attention_mask", "labels"]
    padding_side = "right"

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        cls_token="<cls>",
        eos_token="</s>",
        mask_token="<mask:0>",
        pad_token="<pad>",
        sep_token="<sep>",
        unk_token="<unk>",
        additional_special_tokens=None,
        **kwargs,
    ):
        if additional_special_tokens is None:
            additional_special_tokens = ["<mask:1>", "<mask:7>"]
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        super().__init__(
            bos_token=bos_token,
            cls_token=cls_token,
            eos_token=eos_token,
            mask_token=mask_token,
            pad_token=pad_token,
            sep_token=sep_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            verbose=False,
            **kwargs,
        )

    @property
    def space_token(self):
        """space_token"""
        return "<mask:1>"

    @property
    def space_token_id(self):
        """space_token_id"""
        return self.sp_model.piece_to_id("<mask:1>")

    @property
    def gend_token(self):
        """gend_token"""
        return "<mask:7>"

    @property
    def gend_token_id(self):
        """gend_token_id"""
        return self.sp_model.piece_to_id("<mask:7>")

    @property
    def im_start_id(self):
        """im_start_id"""
        return self.sp_model.piece_to_id("<|im_start|>")

    @property
    def im_end_id(self):
        """im_end_id"""
        return self.sp_model.piece_to_id("<|im_end|>")

    @property
    def vocab_size(self):
        """vocab_size"""
        return self.sp_model.vocab_size()

    def get_vocab(self):
        """get_vocab"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """_tokenize"""
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """_convert_token_to_id"""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, id):
        """_convert_id_to_token"""
        return self.sp_model.id_to_piece(id)

    def convert_tokens_to_string(self, tokens):
        """convert_tokens_to_string"""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def prepare_for_model(self, *args, **kwargs):
        """prepare_for_model"""
        kwargs.pop("add_special_tokens", None)
        return super().prepare_for_model(*args, **kwargs)

    def save_vocabulary(
        self, save_directory, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """save_vocabulary"""
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + self.resource_files_names["vocab_file"],
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                fi.write(self.sp_model.serialized_model_proto())
        return (out_vocab_file,)

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """tokenize"""
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)
        _no_split_tokens = list(self.added_tokens_encoder.keys())
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            escaped_special_toks = [
                re.escape(s_tok)
                for s_tok in (_no_split_tokens + self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(
                pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text
            )
        no_split_token = set(_no_split_tokens)
        tokens = self.tokens_trie.split(text)
        tokenized_text = []
        for token in tokens:
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        return tokenized_text

    def _decode(self, *args, **kwargs):
        """_decode"""
        kwargs.pop("clean_up_tokenization_spaces", None)
        kwargs.pop("spaces_between_special_tokens", None)
        return super()._decode(
            *args,
            **kwargs,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )

    def _pad(
        self,
        encoded_inputs: Dict,
        max_length: Optional[int] = None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        """_pad"""
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        if return_attention_mask:
            required_input = encoded_inputs[self.model_input_names[0]]
            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = len(required_input)
            if (
                max_length is not None
                and pad_to_multiple_of is not None
                and (max_length % pad_to_multiple_of != 0)
            ):
                max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
            needs_to_be_padded = (
                padding_strategy != PaddingStrategy.DO_NOT_PAD
                and len(required_input) != max_length
            )
            if (
                "attention_mask" in encoded_inputs
                and encoded_inputs["attention_mask"] is not None
            ):
                attention_mask = encoded_inputs.pop("attention_mask")
                # Convert list or numpy array to tensor
                if isinstance(attention_mask, list):
                    attention_mask = np.array(attention_mask)
                elif not isinstance(attention_mask, np.ndarray):
                    raise ValueError(
                        f"Unexpected type {type(attention_mask)} of attention_mask"
                    )
            else:
                attention_mask = np.tril(
                    np.ones((len(required_input), len(required_input)), dtype=np.int64)
                )
                attention_mask = np.expand_dims(attention_mask, axis=0)
            if needs_to_be_padded:
                difference = max_length - len(required_input)
                if self.padding_side == "right":
                    pad_width = (
                        [(0, difference)] if attention_mask.ndim == 1
                        else [(0, 0), (0, difference), (0, difference)]
                    )
                elif self.padding_side == "left":
                    pad_width = (
                        [(difference, 0)] if attention_mask.ndim == 1
                        else [(0, 0), (difference, 0), (difference, 0)]
                    )
                else:
                    raise ValueError("Invalid padding strategy: " + str(self.padding_side))
                attention_mask = np.pad(
                    attention_mask,
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=0,
                )
        encoded_inputs = super()._pad(
            encoded_inputs,
            max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=False,
        )
        if return_attention_mask:
            encoded_inputs["attention_mask"] = attention_mask.tolist()
        return encoded_inputs


_tokenizer_cache = {}
