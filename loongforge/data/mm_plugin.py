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

"""multimodal plugin"""

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
    TypedDict,
    Union,
    Type,
)

import numpy as np
from transformers.image_utils import get_image_size, to_numpy_array
from typing_extensions import override

from loongforge.utils.constants import Placeholder

from PIL import Image

logger = logging.getLogger(__name__)
from PIL.Image import Image as ImageObject


if TYPE_CHECKING:
    import torch
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        """Encoded image type."""

        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, EncodedImage, ImageObject]
    VideoInput = str


class MMPlugin:
    """MM Plugin"""

    def __init__(self, image_token: Optional[str], video_token: Optional[str]) -> None:
        self.image_token = image_token
        self.video_token = video_token

    def _validate_input(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")

        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")

    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        image_resolution: int = kwargs.get("image_resolution")
        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(
                image.height * resize_factor
            )
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        r"""
        Computes video sample frames according to fps.
        """
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        total_frames = video_stream.frames
        sample_frames = (
            float(video_stream.duration * video_stream.time_base) * video_fps
        )
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return math.floor(sample_frames)

    def _regularize_images(
        self, images: Sequence["ImageInput"], **kwargs
    ) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(
                    "Expect input is a list of Images, but got {}.".format(type(image))
                )

            results.append(self._preprocess_image(image, **kwargs))

        return results

    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(
            processor, "video_processor", image_processor
        )
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(
                    image_processor(input_dict["images"], return_tensors="pt")
                )
            if input_dict.get("videos") is not None:
                mm_inputs.update(
                    video_processor(input_dict["videos"], return_tensors="pt")
                )
        elif (
            input_dict.get("images") is not None or input_dict.get("videos") is not None
        ):  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return messages

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""
        Builds batched multimodal inputs for VLMs.
        """
        self._validate_input(images, videos)
        return {}

    def _calculate_timestamps(
        self,
        indices: Union[list[int], np.ndarray],
        video_fps: float,
        merge_size: int = 2):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        # @JJJYmmm frames are merged by self.merge_size, \
        # so we need to average the timestamps between the first/last frame within the temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps


class Qwen2VLPlugin(MMPlugin):
    """Qwen2VL plugin"""

    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        sample_frames = super()._get_video_sample_frames(video_stream, **kwargs)
        sample_frames = sample_frames // 2 * 2
        return sample_frames

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while Placeholder.IMAGE in content:
                if num_image_tokens > len(image_grid_thw):
                    raise ValueError(
                        "`len(images)` is less than the number of {} tokens.".format(
                            Placeholder.IMAGE
                        )
                    )

                content = content.replace(
                    Placeholder.IMAGE,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token
                        * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            while Placeholder.VIDEO in content:
                if num_video_tokens > len(video_grid_thw):
                    raise ValueError(
                        "`len(videos)` is less than the number of {} tokens.".format(
                            Placeholder.VIDEO
                        )
                    )

                content = content.replace(
                    Placeholder.VIDEO,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.video_token
                        * (video_grid_thw[num_video_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(
                "The number of images does not match the number of {} tokens".format(
                    Placeholder.IMAGE
                )
            )

        if len(videos) != num_video_tokens:
            raise ValueError(
                "The number of videos does not match the number of {} tokens".format(
                    Placeholder.VIDEO
                )
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
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)

class Qwen3VLPlugin(MMPlugin):
    """ Qwen3VL plugin """
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 32:
            width, height = max(image.width, 32), max(image.height, 32)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        sample_frames = super()._get_video_sample_frames(video_stream, **kwargs)
        sample_frames = sample_frames // 2 * 2
        return sample_frames

    @override
    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                videos_data = input_dict["videos"]
                if isinstance(videos_data, dict):
                    videos_list = videos_data.get("videos", [])
                    durations = videos_data.get("durations", [None] * len(videos_list))
                else:
                    # Compatible with plain list or tensor
                    videos_list = videos_data
                    durations = [getattr(v, "duration", None) for v in videos_list]
                video_metadata = [
                    {"fps": getattr(processor, "video_fps", 24.0), "duration": duration, "total_num_frames": len(video)}
                    for video, duration in zip(videos_list, durations)
                ]
                mm_inputs.update(
                    video_processor(input_dict["videos"],
                    video_metadata=video_metadata, return_metadata=True)
                )
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_size: int = getattr(image_processor, "merge_size")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])
        video_metadata = mm_inputs.get("video_metadata", None)

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while Placeholder.IMAGE in content:
                if num_image_tokens > len(image_grid_thw):
                    raise ValueError("`len(images)` is less than the number of {} tokens.".format(Placeholder.IMAGE))

                content = content.replace(
                    Placeholder.IMAGE,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            while Placeholder.VIDEO in content:
                if num_video_tokens > len(video_grid_thw):
                    raise ValueError("`len(videos)` is less than the number of {} tokens.".format(Placeholder.VIDEO))

                metadata = video_metadata[num_video_tokens]

                if metadata.fps is None:
                        logger.warning_once(
                            "Qwen3VL requires frame timestamps to construct prompts,"
                            "But the `fps` of the input video could not be inferred. "
                            "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                            "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                        )
                        metadata.fps = 24 if metadata.fps is None else metadata.fps
                curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        merge_size,
                    )

                video_placeholder = ""
                frame_seqlen = video_grid_thw[num_video_tokens][1:].prod() // merge_length

                for frame_idx in range(video_grid_thw[num_video_tokens][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            "<|vision_start|>" + "<|placeholder|>" * frame_seqlen + "<|vision_end|>"
                        )

                if f"<|vision_start|>{self.video_token}<|vision_end|>" in content:
                    content = content.replace(
                        f"<|vision_start|>{self.video_token}<|vision_end|>",
                        video_placeholder,
                        1,
                    )
                else:
                    content = content.replace(Placeholder.VIDEO, video_placeholder, 1)

                content = content.replace("<|placeholder|>", self.video_token)
                num_video_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(Placeholder.IMAGE))

        if len(videos) != num_video_tokens:
            raise ValueError("The number of videos does not match the number of {} tokens".format(Placeholder.VIDEO))

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
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)