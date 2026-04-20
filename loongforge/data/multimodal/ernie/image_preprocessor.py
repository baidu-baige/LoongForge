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

"""image preprocessor adaptive — transformers implementation"""

import math
from typing import List, Optional, Union

import numpy as np
import PIL
from PIL import Image

# ── transformers image processing utilities ───────────────────────────────────
from transformers import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import (
    convert_to_rgb as _tf_convert_to_rgb,
    normalize as _tf_normalize,
    rescale as _tf_rescale,
    resize as _tf_resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
try:
    from transformers.image_utils import ImageInput
except ImportError:
    ImageInput = Union[PIL.Image.Image, np.ndarray, List]

from transformers.tokenization_utils_base import TensorType
import logging

logger = logging.getLogger(__name__)

# ── PIL-aware wrappers (transformers image funcs expect numpy arrays) ─────────

def convert_to_rgb(image):
    """convert_to_rgb"""
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    return _tf_convert_to_rgb(image)


def normalize(image, mean, std, data_format=None, **kwargs):
    """normalize"""
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    return _tf_normalize(image, mean, std, data_format=data_format, **kwargs)


def rescale(image, scale, data_format=None, **kwargs):
    """rescale"""
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    return _tf_rescale(image, scale, data_format=data_format, **kwargs)


def resize(image, size, resample=None, data_format=None, **kwargs):
    """resize"""
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    return _tf_resize(image, size, resample=resample, data_format=data_format, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VideoInput = Union[
    List["PIL.Image.Image"],
    "np.ndarray",
    List["np.ndarray"],
    List[List["PIL.Image.Image"]],
    List[List["np.ndarray"]],
]

__all__ = ["AdaptiveImageProcessor"]


def is_scaled_image(image: np.ndarray) -> bool:
    """is_scaled_image"""
    if image.dtype == np.uint8:
        return False
    return np.min(image) >= 0 and np.max(image) <= 1


def make_batched_images(images) -> List[List[ImageInput]]:
    """make_batched_images"""
    if (
        isinstance(images, (list, tuple))
        and isinstance(images[0], (list, tuple))
        and is_valid_image(images[0][0])
    ):
        return [img for img_list in images for img in img_list]
    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images
    elif is_valid_image(images):
        return [images]
    raise ValueError(f"Could not make batched images from {images}")


def make_batched_videos(videos) -> List[VideoInput]:
    """make_batched_videos"""
    if (
        isinstance(videos, (list, tuple))
        and isinstance(videos[0], (list, tuple))
        and is_valid_image(videos[0][0])
    ):
        return videos
    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        if isinstance(videos[0], Image.Image):
            return [videos]
        elif len(videos[0].shape) == 4:
            return [list(video) for video in videos]
    elif is_valid_image(videos) and len(videos.shape) == 4:
        return [list(videos)]
    raise ValueError(f"Could not make batched video from {videos}")


class AdaptiveImageProcessor(BaseImageProcessor):
    r"""
    Image processor for adaptive resolution inputs.
    Identical public interface and numerical behaviour to the original implementation.
    """

    model_input_names = [
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ]

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[float, List[float]] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_conv_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_conv_size = temporal_conv_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}
        self.do_convert_rgb = do_convert_rgb

    def set_pixels(self, min_pixels=None, max_pixels=None, msg=""):
        """set_pixels"""
        if min_pixels is not None:
            assert isinstance(min_pixels, int) and min_pixels >= 0
            logger.info(f"{msg} AdaptiveImageProcessor set min_pixels = {min_pixels}")
            self.min_pixels = min_pixels
            self.size["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            assert isinstance(max_pixels, int) and max_pixels > 0
            logger.info(f"{msg} AdaptiveImageProcessor set max_pixels = {max_pixels}")
            self.max_pixels = max_pixels
            self.size["max_pixels"] = int(max_pixels)

    def get_smarted_resize(self, height, width, min_pixels=None, max_pixels=None):
        """get_smarted_resize"""
        actual_min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        actual_max_pixels = max_pixels if max_pixels is not None else self.max_pixels
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=actual_min_pixels,
            max_pixels=actual_max_pixels,
        )
        return (resized_height, resized_width), (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )

    def to_dict(self):
        """to_dict"""
        encoder_dict = super().to_dict()
        encoder_dict.pop("image_processor_type", None)
        return encoder_dict

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: bool = True,
        resample: PILImageResampling = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = False,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        predetermined_grid_thw=None,
    ):
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning(
                "It looks like you are trying to rescale already rescaled images. "
                "If the input images have pixel values between 0 and 1, "
                "set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []

        if predetermined_grid_thw is not None:
            assert len(predetermined_grid_thw) == len(images), (
                f"len(predetermined_grid_thw) {len(predetermined_grid_thw)} "
                f"== len(images) {len(images)}"
            )

        for img_idx, image in enumerate(images):
            if do_resize:
                if predetermined_grid_thw is not None:
                    (resized_height, resized_width) = predetermined_grid_thw[img_idx]
                    resized_height *= self.patch_size
                    resized_width *= self.patch_size
                else:
                    resized_height, resized_width = smart_resize(
                        height,
                        width,
                        factor=self.patch_size * self.merge_size,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    )
                image = image.astype("uint8")
                # Convert numpy array back to PIL Image
                image = Image.fromarray(image)
                image = resize(
                    image,
                    size=(resized_height, resized_width),
                    resample=resample,
                    data_format=input_data_format,
                )
            if do_rescale:
                image = rescale(image, scale=rescale_factor, data_format=input_data_format)

            if do_normalize:
                image = normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    data_format=input_data_format,
                )

            image = to_channel_dimension_format(
                image, data_format, input_channel_dim=input_data_format
            )

            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose([0, 3, 1, 2])

        channel = patches.shape[1]
        grid_t = patches.shape[0]
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )
        patches = patches.reshape(
            [
                grid_t,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            ]
        )
        patches = patches.transpose([0, 2, 5, 3, 6, 1, 4, 7])
        flatten_patches = patches.reshape(
            [grid_t * grid_h * grid_w, channel * self.patch_size * self.patch_size]
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: bool = True,
        size: Optional[Union[int, List[int]]] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        predetermined_grid_thw=None,
    ):
        """preprocess"""
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = make_batched_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray."
            )

        data = {}

        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for img_idx, image in enumerate(images):
                predetermined_grid_thw_one = (
                    [predetermined_grid_thw[img_idx]]
                    if predetermined_grid_thw is not None
                    else None
                )
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    predetermined_grid_thw=predetermined_grid_thw_one,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}

        if videos is not None:
            pixel_values, vision_grid_thws = [], []
            for imgs in videos:
                patches, video_grid_thw = self._preprocess(
                    imgs,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    predetermined_grid_thw=predetermined_grid_thw,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(video_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {
                "pixel_values_videos": pixel_values,
                "video_grid_thw": vision_grid_thws,
            }

        return BatchFeature(data=data, tensor_type=return_tensors)


# ── math helpers ──────────────────────────────────────────────────────────────

def round_by_factor(number: int, factor: int) -> int:
    """round_by_factor"""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """ceil_by_factor"""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """floor_by_factor"""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
):
    """smart_resize"""
    if max(height, width) / min(height, width) > MAX_RATIO:
        if height > width:
            new_width = max(factor, round_by_factor(width, factor))
            new_height = floor_by_factor(new_width * MAX_RATIO, factor)
        else:
            new_height = max(factor, round_by_factor(height, factor))
            new_width = floor_by_factor(new_height * MAX_RATIO, factor)

        logger.info(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}, "
            f"resize to {max(new_height, new_width) / min(new_height, new_width)}"
        )
        height = new_height
        width = new_width

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if min_pixels > h_bar * w_bar or h_bar * w_bar > max_pixels:
        raise ValueError(f"encounter invalid h_bar: {h_bar}, w_bar: {w_bar}")

    return h_bar, w_bar
