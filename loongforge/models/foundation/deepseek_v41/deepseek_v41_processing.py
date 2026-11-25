# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Native DeepSeek-V4.1 image sizing and batch preparation."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import torch

from .deepseek_v41_vision import DeepseekV41VisionConfig

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


def _vision_config(config):
    value = config.vision_config
    if value is None:
        raise ValueError("DeepSeek-V4.1 image processing requires vision_config")
    return DeepseekV41VisionConfig.from_mapping(value)


def plan_image_grid(width: int, height: int, config):
    """Return resized dimensions and ViT/LLM grid sizes for one image."""
    if width <= 0 or height <= 0 or config.max_image_tokens < 4:
        raise ValueError("Images require positive dimensions and max_image_tokens >= 4")
    logical_width = float(width)
    logical_height = float(height)
    if config.max_wh_ratio is not None and width > height * config.max_wh_ratio:
        logical_width = height * config.max_wh_ratio
    if logical_width * logical_height < config.min_pixels:
        scale = math.sqrt(config.min_pixels / (logical_width * logical_height))
        logical_width = int(logical_width * scale)
        logical_height = int(logical_height * scale)

    patch = config.patch_size
    ratio = config.downsample_ratio
    resized_width = math.ceil(logical_width / patch) * patch
    resized_height = math.ceil(logical_height / patch) * patch
    llm_height = math.ceil((resized_height // patch) / ratio)
    llm_width = math.ceil((resized_width // patch) / ratio)
    if llm_height * (llm_width + 1) + 2 > config.max_image_tokens:
        aspect = logical_height / logical_width
        max_width = math.sqrt((config.max_image_tokens - 2) / aspect + 0.25) - 0.5
        max_height = max_width * aspect
        cell = patch * ratio
        if max_width < 1:
            resized_height, resized_width = (
                (config.max_image_tokens - 2) // 2 * cell,
                cell,
            )
        elif max_height < 1:
            resized_height, resized_width = cell, (config.max_image_tokens - 3) * cell
        else:
            scale = min(
                math.floor(max_width) * cell / logical_width,
                math.floor(max_height) * cell / logical_height,
            )
            resized_height = math.floor(logical_height * scale / patch) * patch
            resized_width = math.floor(logical_width * scale / patch) * patch
        llm_height = math.ceil((resized_height // patch) / ratio)
        llm_width = math.ceil((resized_width // patch) / ratio)
    if min(resized_height, resized_width) <= 0:
        raise ValueError("The image cannot fit in the configured visual-token budget")
    return (
        resized_width,
        resized_height,
        resized_height // patch,
        resized_width // patch,
        llm_height,
        llm_width,
    )


def _load_image(value):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        for key in ("image", "bytes", "path"):
            if value.get(key) is not None:
                return _load_image(value[key])
        raise ValueError("Image records must contain image, bytes, or path")
    if isinstance(value, bytes):
        with Image.open(io.BytesIO(value)) as image:
            return image.convert("RGB")
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    raise TypeError(f"Unsupported image source: {type(value).__name__}")


def preprocess_image(value, config):
    """Resize, normalize, and split one RGB image into row-major patches."""
    from PIL import ImageOps

    image = _load_image(value)
    width, height, vit_height, vit_width, llm_height, llm_width = plan_image_grid(image.width, image.height, config)
    if config.max_wh_ratio is not None and image.width >= config.max_wh_ratio * image.height:
        image = image.resize((width, height))
    else:
        image = ImageOps.pad(image, (width, height), color=(127, 127, 127))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
    pixels = pixels.permute(2, 0, 1).div_(255).sub_(0.5).div_(0.5)
    patch = config.patch_size
    patches = (
        pixels.reshape(3, vit_height, patch, vit_width, patch)
        .permute(1, 3, 0, 2, 4)
        .reshape(vit_height * vit_width, 3, patch, patch)
        .to(torch.bfloat16)
    )
    return patches, (vit_height, vit_width, llm_height, llm_width)


class DeepseekV41ImageProcessor:
    """Prepare the native image span and tensors consumed by ``DeepseekV41Model``."""

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.vision_config = _vision_config(config)

    def __call__(self, text, images=None, *, padding=False):
        texts = [text] if isinstance(text, str) else list(text)
        if not texts:
            raise ValueError("DeepSeek-V4.1 image processing requires at least one prompt")
        if images is None:
            image_groups = [[] for _ in texts]
        elif isinstance(images, list) and images and isinstance(images[0], list):
            image_groups = images
        elif len(texts) == 1:
            image_groups = [images if isinstance(images, list) else [images]]
        elif isinstance(images, list) and len(images) == len(texts):
            image_groups = [[image] for image in images]
        else:
            raise ValueError("Images must be grouped per prompt")
        if len(image_groups) != len(texts):
            raise ValueError("The number of image groups must match the number of prompts")

        tokenized = self.tokenizer(texts, add_special_tokens=False, padding=False)
        rows, type_rows, all_patches, all_grids = [], [], [], []
        for raw_ids, group in zip(tokenized["input_ids"], image_groups):
            if raw_ids.count(self.config.image_token_id) != len(group):
                raise ValueError("Image placeholders and supplied images do not match")
            ids, types = [], []
            image_iter = iter(group)
            for token_id in raw_ids:
                if token_id != self.config.image_token_id:
                    ids.append(token_id)
                    types.append(TEXT)
                    continue
                patches, grid = preprocess_image(next(image_iter), self.vision_config)
                vit_height, vit_width, llm_height, llm_width = grid
                span_types = [IMAGE_START] + ([IMAGE] * llm_width + [IMAGE_NEW_LINE]) * llm_height + [IMAGE_END]
                ids.extend([self.config.image_token_id] * len(span_types))
                types.extend(span_types)
                all_patches.append(patches)
                all_grids.append((vit_height, vit_width))
            rows.append(ids)
            type_rows.append(types)

        target_length = max(map(len, rows))
        if padding:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.config.pad_token_id
            if pad_id is None:
                raise ValueError("Padding requires a tokenizer pad token or pad_token_id")
            for ids, types in zip(rows, type_rows):
                count = target_length - len(ids)
                ids.extend([pad_id] * count)
                types.extend([TEXT] * count)

        attention_mask = [
            [1] * len(row) if not padding else [int(token != self.tokenizer.pad_token_id) for token in row]
            for row in rows
        ]
        if padding:
            for row, mask in zip(rows, attention_mask):
                if self.tokenizer.pad_token_id is None:
                    mask[:] = [1 if token != self.config.pad_token_id else 0 for token in row]
                else:
                    mask[:] = [1 if token != self.tokenizer.pad_token_id else 0 for token in row]
        result = {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "vision_token_types": torch.tensor(type_rows, dtype=torch.int32),
        }
        if all_patches:
            result["pixel_values"] = torch.cat(all_patches)
            result["image_grid_hws"] = torch.tensor(all_grids, dtype=torch.long)
        return result


__all__ = [
    "DeepseekV41ImageProcessor",
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_START",
    "TEXT",
    "plan_image_grid",
    "preprocess_image",
]
