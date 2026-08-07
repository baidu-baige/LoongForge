# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V-4.6 task encoder for raw multimodal datasets."""

import torch

from loongforge.data.multimodal.vlm_task_encoder import (
    IMAGE_TOKEN,
    IMAGE_TOKEN_WITH_TAGS,
    IGNORE_INDEX,
    VLMTaskEncoder,
)
from loongforge.utils.constants import Placeholder

from .mm_plugin import MiniCPMV46Plugin


class MiniCPMV46TaskEncoder(VLMTaskEncoder):
    """Use MiniCPM-native image sizing and placeholders in the Energon path."""

    def __init__(self, args):
        super().__init__(args)
        self.minicpm_plugin = MiniCPMV46Plugin()

    def _resize_image(self, image, size_factor=28):
        del size_factor
        return image

    def _resize_video(self, vision, image_factor=28, frame_factor=2):
        del vision, image_factor, frame_factor
        raise ValueError("MiniCPM-V-4.6 video input is not supported.")

    def _process(self, image, text):
        """Build the pretraining tensor contract with MiniCPM visual metadata."""
        images = [] if image is None else [image]
        text = text.replace(IMAGE_TOKEN_WITH_TAGS, Placeholder.IMAGE)
        text = text.replace(IMAGE_TOKEN, Placeholder.IMAGE)

        messages, mm_inputs = self.minicpm_plugin.process_messages(
            [{"role": "user", "content": text}],
            images,
            [],
            self.processor,
        )
        input_ids = torch.tensor(
            self.tokenizer.tokenize(messages[0]["content"], add_special_tokens=False)
        )
        target = input_ids.clone()
        image_processor = getattr(self.processor, "image_processor", None)
        use_image_id = getattr(
            self.processor,
            "default_use_image_id",
            getattr(image_processor, "use_image_id", True),
        )
        placeholders = self.minicpm_plugin._build_image_placeholders(
            mm_inputs,
            self.processor,
            use_image_id=use_image_id,
        )
        input_id_list = input_ids.tolist()
        search_start = 0
        for placeholder in placeholders:
            placeholder_ids = self.tokenizer.tokenize(
                placeholder,
                add_special_tokens=False,
            )
            placeholder_length = len(placeholder_ids)
            for start in range(search_start, len(input_id_list) - placeholder_length + 1):
                if input_id_list[start : start + placeholder_length] == placeholder_ids:
                    target[start : start + placeholder_length] = IGNORE_INDEX
                    search_start = start + placeholder_length
                    break
            else:
                raise ValueError("MiniCPM visual placeholder was not found after tokenization.")

        attn_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        image_grid_thw = mm_inputs.get("image_grid_thw")
        pixel_values = mm_inputs.get("pixel_values")
        images = [pixel_values] if pixel_values is not None else []
        return input_ids, target, images, image_grid_thw, attn_mask
