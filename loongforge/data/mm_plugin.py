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
    Any,
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


class MiniCPMV46Plugin(MMPlugin):
    """MiniCPM-V-4.6 multimodal plugin."""

    def __init__(
        self,
        image_token: Optional[str] = "<|image_pad|>",
        video_token: Optional[str] = "<|video_pad|>",
        image_token_divisor: int = 16,
        video_token_divisor: int = 16,
    ) -> None:
        super().__init__(image_token=image_token, video_token=video_token)
        self.image_token_divisor = image_token_divisor
        self.video_token_divisor = video_token_divisor
        self.image_start_token = "<image>"
        self.image_end_token = "</image>"
        self.slice_start_token = "<slice>"
        self.slice_end_token = "</slice>"
        self.image_id_start_token = "<image_id>"
        self.image_id_end_token = "</image_id>"

    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        del kwargs
        return image.convert("RGB") if image.mode != "RGB" else image

    def _get_image_processor(self, processor: "ProcessorMixin"):
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            return image_processor

        from loongforge.data.minicpm_v_4_6_image_processor import (
            MiniCPMV46ImageProcessor,
        )

        pretrained_path = getattr(processor, "name_or_path", None)
        if not pretrained_path:
            pretrained_path = getattr(processor, "init_kwargs", {}).get("name_or_path")
        if not pretrained_path:
            raise ValueError(
                "MiniCPM-V-4.6 requires a processor with image support or a tokenizer "
                "loaded from a directory containing preprocessor_config.json."
            )
        image_processor = MiniCPMV46ImageProcessor.from_pretrained(
            pretrained_path,
            downsample_mode=getattr(processor, "downsample_mode", None),
        )
        setattr(processor, "image_processor", image_processor)
        return image_processor

    def _target_sizes_to_grid_thw(self, target_sizes):
        import torch

        if target_sizes is None:
            return None
        if not torch.is_tensor(target_sizes):
            target_sizes = torch.tensor(target_sizes, dtype=torch.int32)
        target_sizes = target_sizes.to(dtype=torch.int32)
        if target_sizes.numel() == 0:
            return target_sizes.new_zeros((0, 3))
        if target_sizes.dim() != 2 or target_sizes.shape[-1] != 2:
            raise ValueError(f"Expected MiniCPM target_sizes with shape [n, 2], got {tuple(target_sizes.shape)}.")
        ones = torch.ones((target_sizes.shape[0], 1), dtype=target_sizes.dtype, device=target_sizes.device)
        return torch.cat([ones, target_sizes], dim=-1)

    def _normalize_mm_inputs(self, inputs: Dict[str, Any], *, is_video: bool = False) -> Dict[str, Any]:
        mm_inputs = dict(inputs)
        if is_video:
            target_sizes = mm_inputs.get("target_sizes_videos", mm_inputs.get("target_sizes"))
            grid_thw = self._target_sizes_to_grid_thw(target_sizes)
            if grid_thw is not None:
                mm_inputs["video_grid_thw"] = grid_thw
            if "pixel_values" in mm_inputs and "pixel_values_videos" not in mm_inputs:
                mm_inputs["pixel_values_videos"] = mm_inputs.pop("pixel_values")
        else:
            target_sizes = mm_inputs.get("target_sizes")
            grid_thw = self._target_sizes_to_grid_thw(target_sizes)
            if grid_thw is not None:
                mm_inputs["image_grid_thw"] = grid_thw
        return mm_inputs

    @override
    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        image_processor: "BaseImageProcessor" = self._get_image_processor(processor)
        mm_inputs = {}

        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 2048),
            )
            image_inputs = image_processor(images, return_tensors="pt")
            mm_inputs.update(self._normalize_mm_inputs(image_inputs, is_video=False))

        if len(videos) != 0:
            video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", None)
            if video_processor is None:
                raise ValueError(
                    "MiniCPM-V-4.6 video preprocessing requires a Transformers version "
                    "that provides MiniCPMV4_6VideoProcessor."
                )
            video_inputs = video_processor(videos, return_tensors="pt")
            mm_inputs.update(self._normalize_mm_inputs(video_inputs, is_video=True))

        return mm_inputs

    @staticmethod
    def _token_count(target_size, divisor: int) -> int:
        if hasattr(target_size, "prod"):
            return int((target_size.prod() // divisor).item())
        height, width = target_size
        return int(height * width // divisor)

    def _get_token_divisor(self, processor: Optional["ProcessorMixin"], *, is_video: bool = False) -> int:
        if processor is None:
            return self.video_token_divisor if is_video else self.image_token_divisor
        processor_name = "video_processor" if is_video else "image_processor"
        media_processor = getattr(processor, processor_name, None)
        downsample_mode = getattr(media_processor, "downsample_mode", None)
        if downsample_mode is None:
            downsample_mode = getattr(processor, "downsample_mode", None)
        if downsample_mode == "4x":
            return 4
        if downsample_mode == "16x":
            return 16
        return self.video_token_divisor if is_video else self.image_token_divisor

    def _build_image_placeholders(
        self,
        mm_inputs: Dict[str, Any],
        processor: Optional["ProcessorMixin"],
        *,
        use_image_id: bool,
    ) -> List[str]:
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        grids = mm_inputs.get("grids", [])
        num_patches_per_image = mm_inputs.get("num_patches_per_image", [])
        if not num_patches_per_image:
            num_patches_per_image = [1] * len(image_grid_thw)

        image_processor = getattr(processor, "image_processor", None)
        slice_mode = getattr(image_processor, "slice_mode", True)
        divisor = self._get_token_divisor(processor, is_video=False)

        placeholders = []
        flat_index = 0
        for image_index, patch_count in enumerate(num_patches_per_image):
            if flat_index >= len(image_grid_thw):
                raise ValueError("MiniCPM image target_sizes are shorter than num_patches_per_image.")

            source_grid = image_grid_thw[flat_index][1:]
            placeholder = self._image_placeholder(
                self._token_count(source_grid, divisor),
                image_index=image_index,
                use_image_id=use_image_id,
            )

            grid = grids[image_index] if image_index < len(grids) else [0, 0]
            num_rows, num_cols = (int(grid[0]), int(grid[1])) if grid is not None else (0, 0)
            if slice_mode and patch_count > 1 and num_rows > 0 and num_cols > 0:
                slice_grid = image_grid_thw[flat_index + 1][1:]
                slice_tokens = self._token_count(slice_grid, divisor)
                slice_placeholder = (
                    f"{self.slice_start_token}"
                    f"{self.image_token * slice_tokens}"
                    f"{self.slice_end_token}"
                )
                placeholder += "\n".join(slice_placeholder * num_cols for _ in range(num_rows))

            placeholders.append(placeholder)
            flat_index += int(patch_count)

        return placeholders

    def _build_video_placeholders(
        self,
        mm_inputs: Dict[str, Any],
        processor: Optional["ProcessorMixin"],
    ) -> List[str]:
        video_grid_thw = mm_inputs.get("video_grid_thw", [])
        if len(video_grid_thw) == 0:
            return []

        grids = mm_inputs.get("grids_videos", [])
        num_frames_per_video = mm_inputs.get("num_frames_per_video", [])
        num_patches_per_frame = mm_inputs.get("num_patches_per_frame", [])
        if not num_frames_per_video:
            num_frames_per_video = [len(video_grid_thw)]
        if not num_patches_per_frame:
            num_patches_per_frame = [1] * len(video_grid_thw)

        video_processor = getattr(processor, "video_processor", None)
        slice_mode = getattr(video_processor, "slice_mode", True)
        divisor = self._get_token_divisor(processor, is_video=True)

        placeholders = []
        flat_index = 0
        frame_index = 0
        for video_index, frame_count in enumerate(num_frames_per_video):
            del video_index
            video_placeholder = ""
            for _ in range(int(frame_count)):
                patch_count = int(num_patches_per_frame[frame_index])
                frame_grid = video_grid_thw[flat_index][1:]
                frame_placeholder = (
                    f"{self.image_start_token}"
                    f"{self.video_token * self._token_count(frame_grid, divisor)}"
                    f"{self.image_end_token}"
                )
                grid = grids[frame_index] if frame_index < len(grids) else [0, 0]
                num_rows, num_cols = (int(grid[0]), int(grid[1])) if grid is not None else (0, 0)
                if slice_mode and patch_count > 1 and num_rows > 0 and num_cols > 0:
                    slice_grid = video_grid_thw[flat_index + 1][1:]
                    slice_tokens = self._token_count(slice_grid, divisor)
                    slice_placeholder = (
                        f"{self.slice_start_token}"
                        f"{self.video_token * slice_tokens}"
                        f"{self.slice_end_token}"
                    )
                    frame_placeholder += "\n".join(slice_placeholder * num_cols for _ in range(num_rows))

                video_placeholder += frame_placeholder
                flat_index += patch_count
                frame_index += 1
            placeholders.append(video_placeholder)
        return placeholders

    def _image_placeholder(self, token_count: int, image_index: int, *, use_image_id: bool) -> str:
        placeholder = f"{self.image_start_token}{self.image_token * token_count}{self.image_end_token}"
        if use_image_id:
            placeholder = f"{self.image_id_start_token}{image_index}{self.image_id_end_token}" + placeholder
        return placeholder

    def _replace_content_item(
        self,
        item: Dict[str, Any],
        image_placeholders: List[str],
        video_placeholders: List[str],
        image_index: int,
        video_index: int,
    ) -> Tuple[str, int, int]:
        if item.get("type") == "image" or item.get("image") is not None or item.get("image_url") is not None:
            if image_index >= len(image_placeholders):
                raise ValueError(f"`len(images)` is less than the number of {Placeholder.IMAGE} tokens.")
            return image_placeholders[image_index], image_index + 1, video_index
        if item.get("type") == "video" or item.get("video") is not None:
            if video_index >= len(video_placeholders):
                raise ValueError(f"`len(videos)` is less than the number of {Placeholder.VIDEO} tokens.")
            return video_placeholders[video_index], image_index, video_index + 1
        if "text" in item:
            return item["text"], image_index, video_index
        raise ValueError("Unexpected MiniCPM content item type.")

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[Dict[str, str]], Dict]:
        self._validate_input(images, videos)
        mm_inputs = self._get_mm_inputs(images, videos, processor)

        use_image_id = getattr(
            processor,
            "default_use_image_id",
            getattr(getattr(processor, "image_processor", None), "use_image_id", True),
        )
        image_placeholders = self._build_image_placeholders(
            mm_inputs, processor, use_image_id=use_image_id
        )
        video_placeholders = self._build_video_placeholders(mm_inputs, processor)

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                image_occurrences = content.count(Placeholder.IMAGE)
                for _ in range(image_occurrences):
                    if num_image_tokens >= len(image_placeholders):
                        raise ValueError(f"`len(images)` is less than the number of {Placeholder.IMAGE} tokens.")
                    content = content.replace(Placeholder.IMAGE, image_placeholders[num_image_tokens], 1)
                    num_image_tokens += 1
                video_occurrences = content.count(Placeholder.VIDEO)
                for _ in range(video_occurrences):
                    if num_video_tokens >= len(video_placeholders):
                        raise ValueError(f"`len(videos)` is less than the number of {Placeholder.VIDEO} tokens.")
                    content = content.replace(Placeholder.VIDEO, video_placeholders[num_video_tokens], 1)
                    num_video_tokens += 1
            elif isinstance(content, Sequence):
                parts = []
                for item in content:
                    if not isinstance(item, Dict):
                        raise ValueError(f"Unexpected MiniCPM content item type: {type(item)}.")
                    part, num_image_tokens, num_video_tokens = self._replace_content_item(
                        item, image_placeholders, video_placeholders, num_image_tokens, num_video_tokens
                    )
                    parts.append(part)
                content = "\n".join(parts)
            else:
                raise ValueError(f"Unexpected MiniCPM message content type: {type(content)}.")
            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {Placeholder.IMAGE} tokens.")
        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {Placeholder.VIDEO} tokens.")
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
        del imglens, vidlens, seqlens
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
