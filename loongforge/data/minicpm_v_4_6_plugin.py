# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Image-only multimodal plugin for MiniCPM-V-4.6."""

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from typing_extensions import override

from loongforge.data.mm_plugin import MMPlugin
from loongforge.utils.constants import Placeholder

from .minicpm_v_4_6_image_processor import MiniCPMV46ImageProcessor

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image as ImageObject


class MiniCPMV46Plugin(MMPlugin):
    """Build MiniCPM image placeholders and packed image tensors."""

    def __init__(
        self,
        image_token: Optional[str] = "<|image_pad|>",
        image_token_divisor: int = 16,
    ) -> None:
        super().__init__(image_token=image_token, video_token=None)
        self.image_token_divisor = image_token_divisor
        self.image_start_token = "<image>"
        self.image_end_token = "</image>"
        self.slice_start_token = "<slice>"
        self.slice_end_token = "</slice>"
        self.image_id_start_token = "<image_id>"
        self.image_id_end_token = "</image_id>"

    @staticmethod
    def _reject_videos(videos: Sequence[Any]) -> None:
        if videos:
            raise ValueError("MiniCPM-V-4.6 video input is not supported.")

    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        del kwargs
        return image.convert("RGB") if image.mode != "RGB" else image

    @staticmethod
    def _get_image_processor(processor):
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            return image_processor

        pretrained_path = getattr(processor, "name_or_path", None)
        if not pretrained_path:
            pretrained_path = getattr(processor, "init_kwargs", {}).get(
                "name_or_path"
            )
        if not pretrained_path:
            raise ValueError(
                "MiniCPM-V-4.6 requires a processor with image support or a "
                "tokenizer directory containing preprocessor_config.json."
            )
        image_processor = MiniCPMV46ImageProcessor.from_pretrained(
            pretrained_path,
            downsample_mode=getattr(processor, "downsample_mode", None),
        )
        setattr(processor, "image_processor", image_processor)
        return image_processor

    @staticmethod
    def _target_sizes_to_grid_thw(target_sizes):
        import torch

        if target_sizes is None:
            return None
        if not torch.is_tensor(target_sizes):
            target_sizes = torch.tensor(target_sizes, dtype=torch.int32)
        target_sizes = target_sizes.to(dtype=torch.int32)
        if target_sizes.numel() == 0:
            return target_sizes.new_zeros((0, 3))
        if target_sizes.dim() != 2 or target_sizes.shape[-1] != 2:
            raise ValueError(
                "Expected MiniCPM target_sizes with shape [n, 2], got "
                f"{tuple(target_sizes.shape)}."
            )
        ones = torch.ones(
            (target_sizes.shape[0], 1),
            dtype=target_sizes.dtype,
            device=target_sizes.device,
        )
        return torch.cat([ones, target_sizes], dim=-1)

    @override
    def _get_mm_inputs(self, images, videos, processor) -> Dict[str, "torch.Tensor"]:
        self._reject_videos(videos)
        if not images:
            return {}
        images = self._regularize_images(
            images,
            image_resolution=getattr(processor, "image_resolution", 2048),
        )
        image_processor = self._get_image_processor(processor)
        mm_inputs = dict(image_processor(images, return_tensors="pt"))
        image_grid_thw = self._target_sizes_to_grid_thw(
            mm_inputs.get("target_sizes")
        )
        if image_grid_thw is not None:
            mm_inputs["image_grid_thw"] = image_grid_thw
        return mm_inputs

    @staticmethod
    def _token_count(target_size, divisor: int) -> int:
        if hasattr(target_size, "prod"):
            return int((target_size.prod() // divisor).item())
        height, width = target_size
        return int(height * width // divisor)

    def _get_token_divisor(self, processor) -> int:
        image_processor = getattr(processor, "image_processor", None)
        downsample_mode = getattr(image_processor, "downsample_mode", None)
        if downsample_mode is None:
            downsample_mode = getattr(processor, "downsample_mode", None)
        if downsample_mode == "4x":
            return 4
        if downsample_mode == "16x":
            return 16
        return self.image_token_divisor

    def _image_placeholder(
        self, token_count: int, image_index: int, *, use_image_id: bool
    ) -> str:
        placeholder = (
            f"{self.image_start_token}"
            f"{self.image_token * token_count}"
            f"{self.image_end_token}"
        )
        if use_image_id:
            placeholder = (
                f"{self.image_id_start_token}{image_index}{self.image_id_end_token}"
                + placeholder
            )
        return placeholder

    def _build_image_placeholders(
        self,
        mm_inputs: Dict[str, Any],
        processor,
        *,
        use_image_id: bool,
    ) -> List[str]:
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        grids = mm_inputs.get("grids", [])
        patch_counts = mm_inputs.get("num_patches_per_image", [])
        if not patch_counts:
            patch_counts = [1] * len(image_grid_thw)
        image_processor = getattr(processor, "image_processor", None)
        slice_mode = getattr(image_processor, "slice_mode", True)
        divisor = self._get_token_divisor(processor)

        placeholders = []
        flat_index = 0
        for image_index, patch_count in enumerate(patch_counts):
            if flat_index >= len(image_grid_thw):
                raise ValueError(
                    "MiniCPM image target_sizes are shorter than num_patches_per_image."
                )
            source_grid = image_grid_thw[flat_index][1:]
            placeholder = self._image_placeholder(
                self._token_count(source_grid, divisor),
                image_index,
                use_image_id=use_image_id,
            )
            grid = grids[image_index] if image_index < len(grids) else [0, 0]
            num_rows, num_cols = map(int, grid or [0, 0])
            if slice_mode and patch_count > 1 and num_rows > 0 and num_cols > 0:
                slice_grid = image_grid_thw[flat_index + 1][1:]
                slice_placeholder = (
                    f"{self.slice_start_token}"
                    f"{self.image_token * self._token_count(slice_grid, divisor)}"
                    f"{self.slice_end_token}"
                )
                placeholder += "\n".join(
                    slice_placeholder * num_cols for _ in range(num_rows)
                )
            placeholders.append(placeholder)
            flat_index += int(patch_count)
        return placeholders

    @staticmethod
    def _content_item_text(item, image_placeholders, image_index):
        if (
            item.get("type") == "image"
            or item.get("image") is not None
            or item.get("image_url") is not None
        ):
            if image_index >= len(image_placeholders):
                raise ValueError(
                    f"`len(images)` is less than the number of {Placeholder.IMAGE} tokens."
                )
            return image_placeholders[image_index], image_index + 1
        if item.get("type") == "video" or item.get("video") is not None:
            raise ValueError("MiniCPM-V-4.6 video input is not supported.")
        if "text" in item:
            return item["text"], image_index
        raise ValueError("Unexpected MiniCPM content item type.")

    @override
    def process_messages(
        self,
        messages,
        images,
        videos,
        processor,
    ) -> Tuple[List[Dict[str, str]], Dict]:
        self._reject_videos(videos)
        mm_inputs = self._get_mm_inputs(images, [], processor)
        image_processor = getattr(processor, "image_processor", None)
        use_image_id = getattr(
            processor,
            "default_use_image_id",
            getattr(image_processor, "use_image_id", True),
        )
        image_placeholders = self._build_image_placeholders(
            mm_inputs, processor, use_image_id=use_image_id
        )

        image_index = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                image_occurrences = content.count(Placeholder.IMAGE)
                for _ in range(image_occurrences):
                    if image_index >= len(image_placeholders):
                        raise ValueError(
                            f"`len(images)` is less than the number of {Placeholder.IMAGE} tokens."
                        )
                    content = content.replace(
                        Placeholder.IMAGE, image_placeholders[image_index], 1
                    )
                    image_index += 1
                if Placeholder.VIDEO in content:
                    raise ValueError("MiniCPM-V-4.6 video input is not supported.")
            elif isinstance(content, Sequence):
                parts = []
                for item in content:
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"Unexpected MiniCPM content item type: {type(item)}."
                        )
                    part, image_index = self._content_item_text(
                        item, image_placeholders, image_index
                    )
                    parts.append(part)
                content = "\n".join(parts)
            else:
                raise ValueError(
                    f"Unexpected MiniCPM message content type: {type(content)}."
                )
            message["content"] = content

        if len(images) != image_index:
            raise ValueError(
                f"The number of images does not match the number of {Placeholder.IMAGE} tokens."
            )
        return messages, mm_inputs

    @override
    def get_mm_inputs(
        self,
        images,
        videos,
        imglens,
        vidlens,
        seqlens,
        processor,
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        del imglens, vidlens, seqlens
        self._reject_videos(videos)
        return self._get_mm_inputs(images, [], processor)
