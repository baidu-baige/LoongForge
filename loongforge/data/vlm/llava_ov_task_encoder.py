# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LLaVA-OneVision-1.5 under the Apache-2.0 License.

"""LLavA-OneVision TaskEncoder class."""

import re

import numpy as np
import torch
from megatron.energon import VQASample
from transformers import AutoProcessor
from typing_extensions import override

from loongforge.data.vlm.length_sort_dataset import LengthPoolSortDataset
from loongforge import constants

from .base_task_encoder import BaseTaskSample
from .vlm_task_encoder import (
    IMAGE_TOKEN_WITH_TAGS,
    VLMTaskEncoder,
    VLMTaskSample,
)


class LLavaOv15TaskEncoder(VLMTaskEncoder):
    """A task encoder for LLava OV 1.5 that extends VLMTaskEncoder."""

    _pool_sort = False  # True only while build_train_datasets runs.

    def __init__(self, args, tokenizer, chat_template=None):
        super().__init__(args, tokenizer, chat_template)
        # The parent loads from hf_processor_path when set; LLaVA-OV always uses the tokenizer path.
        if getattr(args, "hf_processor_path", None):
            self.processor = AutoProcessor.from_pretrained(
                args.hf_tokenizer_path, trust_remote_code=True
            )
            if args.image_resolution:
                setattr(self.processor, "image_resolution", args.image_resolution)

    def encode_vqa(self, sample: VQASample) -> BaseTaskSample:
        """Encode pretrain sample in Qwen2VL style."""
        if self.args.training_phase == constants.TrainingPhase.PRETRAIN:
            if self.args.add_question_in_pretrain:
                text = (sample.context + sample.answers).replace(
                    "<image>", IMAGE_TOKEN_WITH_TAGS
                )
            else:
                text = IMAGE_TOKEN_WITH_TAGS + sample.answers
            text = text + self.tokenizer.tokenizer.eos_token
            input_ids, target, imgs, image_grid_thw, attn_mask = self._process(
                sample.image, text
            )
        elif self.args.training_phase == constants.TrainingPhase.SFT:

            if len(sample.answers) < 1:
                raise ValueError("sample.answers < 1!")

            # Add image resize check for PIL.Image
            if sample.image is not None:

                img_arr = np.array(sample.image)
                if np.sum(img_arr) == 0:
                    raise ValueError("Image pixels are all zero!")

            # Truncate answer to the last full sentence if it exceeds the max length.
            max_answer_length = self.args.training_rice_vl_max_answer_length
            if len(sample.answers) > max_answer_length:
                original_length = len(sample.answers)

                # Perform a preliminary cut at the maximum allowed length.
                preliminary_cut = sample.answers[:max_answer_length]

                # Clean up trailing punctuation and whitespace from the preliminary cut
                cleaned_cut = preliminary_cut.rstrip(".。 \t\n")

                # Find the last occurrence of a sentence-ending punctuation mark
                # followed by a space or the end of the string.
                # This pattern looks for sentence enders (. or 。)
                sentence_enders_pattern = r"[.。]"

                # Find all matches and get the end position of the last match
                matches = list(re.finditer(sentence_enders_pattern, cleaned_cut))

                if matches:
                    # Get the end position of the last match
                    last_end_index = matches[-1].end()
                    # Truncate at the end of the last full sentence.
                    sample.answers = cleaned_cut[:last_end_index]
                else:
                    # Fallback to a hard cut of the original preliminary string if no sentence ender is found.
                    sample.answers = preliminary_cut

                print(
                    f"Answer truncated to a full sentence. "
                    f"Original length: {original_length}, New length: {len(sample.answers)}"
                )

            text = self.processor.apply_chat_template(
                [
                    {"role": "user", "content": sample.context},
                    {"role": "assistant", "content": sample.answers},
                ],
                tokenize=False,
            ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)
            if text[-1] == "\n":
                text = text[:-1]
            input_ids, _, imgs, image_grid_thw, attn_mask = self._process(
                sample.image, text, add_special_tokens=False
            )
            target = torch.ones_like(input_ids) * constants.IGNORE_INDEX
            answers = self.tokenizer.tokenize(sample.answers, add_special_tokens=False)
            target[-len(answers) - 1 : -1] = torch.tensor(answers)
            target[-1] = input_ids[-1]
            # print(target[-1])
        else:
            raise NotImplementedError(
                f"Unknown training phase {self.args.training_phase}"
            )

        num_tiles = [len(image_grid_thw)]

        if self.args.enable_discard_sample:
            assert (
                len(input_ids) <= self.args.seq_length
            ), f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert (
                image_grid_thw.prod() / 4 <= self.args.seq_length
            ), f"{sample.__key__} grid_thw: {image_grid_thw}"

        return VLMTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    @override
    def build_train_datasets(self, **kwargs):
        """Upstream, but pool-sort before batching (train only; val shares build_batch)."""
        self._pool_sort = True
        try:
            return super().build_train_datasets(**kwargs)
        finally:
            self._pool_sort = False

    @override
    def build_batch(self, dataset, **kwargs):
        """Insert pool sorting before batching when enabled."""
        pool_size = getattr(self.args, "length_sort_pool_size", 0)
        if self._pool_sort and pool_size and pool_size > 0:
            dataset = LengthPoolSortDataset(
                dataset,
                pool_size=pool_size,
                key_fn=lambda s: getattr(s, "total_len", len(getattr(s, "tokens"))),
                ascending=not getattr(self.args, "length_sort_desc", False),
                worker_config=kwargs["worker_config"],
            )
        return super().build_batch(dataset, **kwargs)
