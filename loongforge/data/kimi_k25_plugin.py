# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Kimi K2.5 Multimodal Plugin for LoongForge

This plugin handles Kimi K2.5 specific image/video processing and token formatting.
Kimi K2.5 uses a different token format than Qwen2-VL:
- Image: <|media_begin|>image<|media_content|><|media_pad|><|media_end|>
- Video: timestamp<|media_begin|>video<|media_content|><|media_pad|><|media_end|>
"""

import logging
import math
from copy import deepcopy
from io import BytesIO
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from typing_extensions import override
from PIL import Image

from .mm_plugin import MMPlugin

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch
    from transformers.image_processing_utils import BaseImageProcessor
    from PIL.Image import Image as ImageObject

    ImageInput = Union[str, Dict, "ImageObject"]
    VideoInput = str


# Kimi K2.5 special tokens
MEDIA_BEGIN = "<|media_begin|>"
MEDIA_END = "<|media_end|>"
MEDIA_CONTENT = "<|media_content|>"
MEDIA_PAD = "<|media_pad|>"

# Image format: <|media_begin|>image<|media_content|><|media_pad|><|media_end|>
IMAGE_PLACEHOLDER_TEMPLATE = f"{MEDIA_BEGIN}image{MEDIA_CONTENT}{{tokens}}{MEDIA_END}"
# Video chunk format: timestamp<|media_begin|>video<|media_content|><|media_pad|><|media_end|>
VIDEO_CHUNK_TEMPLATE = "{{timestamp}}" + f"{MEDIA_BEGIN}video{MEDIA_CONTENT}{{tokens}}{MEDIA_END}"


class KimiK25Plugin(MMPlugin):
    """Kimi K2.5 multimodal plugin.

    Handles Kimi K2.5 specific:
    - Image preprocessing (NaViT-style dynamic resolution)
    - Video chunking with timestamps
    - Token placeholder expansion based on grid_thw
    """

    def __init__(
        self,
        image_token: Optional[str] = MEDIA_CONTENT,
        video_token: Optional[str] = MEDIA_CONTENT,
        merge_kernel_size: Tuple[int, int] = (2, 2),
        temporal_merge_kernel_size: int = 4,
    ) -> None:
        """Initialize KimiK25Plugin.

        Args:
            image_token: Token used for image placeholders (default: <|media_content|>)
            video_token: Token used for video placeholders (default: <|media_content|>)
            merge_kernel_size: Spatial merge kernel size [h, w] (default: [2, 2])
            temporal_merge_kernel_size: Temporal merge kernel size (default: 4)
        """
        super().__init__(image_token=image_token, video_token=video_token)
        self.merge_kernel_size = merge_kernel_size
        self.temporal_merge_kernel_size = temporal_merge_kernel_size
        # Kimi uses <|media_content|> as the placeholder token
        self.media_placeholder_token_id = 163605

    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        """Preprocess image for Kimi K2.5.

        Kimi K2.5 uses NaViT-style dynamic resolution, but we apply basic constraints here.
        The actual resizing is done by the Kimi processor.
        """
        image = super()._preprocess_image(image, **kwargs)

        # Ensure minimum size (Kimi uses patch_size=14, so min should be at least 14)
        min_size = 28
        if min(image.width, image.height) < min_size:
            width = max(image.width, min_size)
            height = max(image.height, min_size)
            image = image.resize((width, height), resample=Image.NEAREST)

        # Limit extreme aspect ratios
        max_aspect_ratio = 200
        if image.width / image.height > max_aspect_ratio:
            width = int(image.height * (max_aspect_ratio - 20))
            image = image.resize((width, image.height), resample=Image.NEAREST)
        elif image.height / image.width > max_aspect_ratio:
            height = int(image.width * (max_aspect_ratio - 20))
            image = image.resize((image.width, height), resample=Image.NEAREST)

        return image

    def _compute_num_tokens_from_grid_thw(self, grid_thw) -> int:
        """Compute number of tokens after spatial merge and temporal pooling.

        For Kimi K2.5:
        - grid_thw = [T, H, W] where H, W are in patch units
        - After spatial merge: new_h = H // merge_h, new_w = W // merge_w
        - After temporal pooling: T dimension is pooled away
        - Final tokens = new_h * new_w

        Args:
            grid_thw: Tensor or list [T, H, W]

        Returns:
            Number of tokens for this image/video chunk
        """
        if hasattr(grid_thw, 'tolist'):
            t, h, w = grid_thw.tolist()
        else:
            t, h, w = grid_thw

        merge_h, merge_w = self.merge_kernel_size
        new_height = h // merge_h
        new_width = w // merge_w
        # Temporal dimension is pooled, only spatial dimensions matter
        return new_height * new_width

    @override
    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        """Process visual inputs for Kimi K2.5.

        Returns:
            pixel_values: tensor with shape (num_patches, patch_dim)
            grid_thws: tensor with shape (num_images, 3) for [T, H, W]
        """
        import torch

        # Use Kimi's media processor
        media_processor = getattr(processor, 'media_processor', None)
        if media_processor is None:
            media_processor = getattr(processor, 'image_processor', None)

        if media_processor is None:
            raise ValueError("Processor must have media_processor or image_processor")

        mm_inputs = {}

        if len(images) != 0:
            # Regularize images first
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 2048),
            )
            # Build medias list for Kimi processor
            medias = [{'type': 'image', 'image': img} for img in images]

            # Process through Kimi processor
            processed = processor(
                text="",  # Empty text, we only need pixel values
                medias=medias,
                return_tensors="pt",
            )

            if "pixel_values" in processed:
                mm_inputs["pixel_values"] = processed["pixel_values"]
            if "grid_thws" in processed:
                mm_inputs["image_grid_thw"] = processed["grid_thws"]

        if len(videos) != 0:
            # For videos, Kimi splits them into chunks
            # This is typically handled by the processor
            medias = [{'type': 'video', 'video': vid} for vid in videos]

            processed = processor(
                text="",
                medias=medias,
                return_tensors="pt",
            )

            if "pixel_values" in processed:
                if "pixel_values" in mm_inputs:
                    # Concatenate with image pixel values
                    mm_inputs["pixel_values_videos"] = processed["pixel_values"]
                else:
                    mm_inputs["pixel_values_videos"] = processed["pixel_values"]
            if "grid_thws" in processed:
                mm_inputs["video_grid_thw"] = processed["grid_thws"]

        return mm_inputs

    def _build_image_placeholder(self, num_tokens: int) -> str:
        """Build image placeholder string with correct number of tokens.

        Format: <|media_begin|>image<|media_content|>...<|media_content|><|media_end|>
        """
        # Kimi uses multiple <|media_content|> tokens as placeholders
        tokens_str = self.image_token * num_tokens
        return f"{MEDIA_BEGIN}image{tokens_str}{MEDIA_END}"

    def _build_video_chunk_placeholder(self, num_tokens: int, timestamp: str = "") -> str:
        """Build video chunk placeholder string with timestamp.

        Format: timestamp<|media_begin|>video<|media_content|>...<|media_content|><|media_end|>
        """
        tokens_str = self.video_token * num_tokens
        return f"{timestamp}{MEDIA_BEGIN}video{tokens_str}{MEDIA_END}"

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[Dict[str, str]], Dict]:
        """Process messages and replace placeholders with Kimi K2.5 format.

        Replaces generic <image>/<video> placeholders with Kimi-specific format:
        - <image> -> <|media_begin|>image<|media_content|>...<|media_end|>
        - <video> -> chunks with timestamps

        Args:
            messages: List of message dicts with 'role' and 'content'
            images: List of image inputs
            videos: List of video inputs
            processor: Kimi processor with media_processor

        Returns:
            Tuple of (processed_messages, mm_inputs)
        """
        from loongforge.utils.constants import Placeholder

        self._validate_input(images, videos)

        # Get multimodal inputs (pixel values and grid_thws)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)

        for message in messages:
            content = message["content"]

            # Replace image placeholders
            while Placeholder.IMAGE in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(
                        f"`len(images)` ({len(images)}) is less than the number of "
                        f"{Placeholder.IMAGE} tokens in messages."
                    )

                # Compute number of tokens for this image
                num_tokens = self._compute_num_tokens_from_grid_thw(
                    image_grid_thw[num_image_tokens]
                )

                # Build placeholder string
                placeholder = self._build_image_placeholder(num_tokens)

                content = content.replace(Placeholder.IMAGE, placeholder, 1)
                num_image_tokens += 1

            # Replace video placeholders
            while Placeholder.VIDEO in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(
                        f"`len(videos)` ({len(videos)}) is less than the number of "
                        f"{Placeholder.VIDEO} tokens in messages."
                    )

                # For video, Kimi splits into chunks with timestamps
                # Here we handle as single chunk for simplicity
                num_tokens = self._compute_num_tokens_from_grid_thw(
                    video_grid_thw[num_video_tokens]
                )

                # Build video chunk placeholder (without timestamp for now)
                placeholder = self._build_video_chunk_placeholder(num_tokens)

                content = content.replace(Placeholder.VIDEO, placeholder, 1)
                num_video_tokens += 1

            message["content"] = content

        # Validate counts
        if len(images) != num_image_tokens:
            raise ValueError(
                f"The number of images ({len(images)}) does not match "
                f"the number of {Placeholder.IMAGE} tokens ({num_image_tokens})"
            )

        if len(videos) != num_video_tokens:
            raise ValueError(
                f"The number of videos ({len(videos)}) does not match "
                f"the number of {Placeholder.VIDEO} tokens ({num_video_tokens})"
            )

        return messages, mm_inputs

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        """Get multimodal inputs for batched processing.

        Args:
            images: Sequence of image inputs
            videos: Sequence of video inputs
            imglens: Number of images per sample
            vidlens: Number of videos per sample
            seqlens: Sequence lengths
            processor: Kimi processor

        Returns:
            Dictionary with pixel_values, grid_thws, etc.
        """
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


# Plugin registry entry
def get_kimi_k25_plugin(**kwargs) -> KimiK25Plugin:
    """Factory function to create KimiK25Plugin with default settings."""
    return KimiK25Plugin(
        image_token=MEDIA_CONTENT,
        video_token=MEDIA_CONTENT,
        merge_kernel_size=(2, 2),
        temporal_merge_kernel_size=4,
        **kwargs,
    )
