# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Hugging Face Transformers MiniCPM-V-4.6 under the Apache-2.0 License.
# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

"""MiniCPM-V-4.6 image processing for the pinned Transformers version."""

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_functional


def _ensure_divide(length: int | float, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


class MiniCPMV46ImageProcessor:
    """Image-only MiniCPM-V-4.6 processor compatible with Transformers 5.2."""

    model_input_names = ["pixel_values", "target_sizes"]

    def __init__(
        self,
        *,
        max_slice_nums: int = 9,
        scale_resolution: int = 448,
        patch_size: int = 14,
        slice_mode: bool = True,
        downsample_mode: str = "16x",
        use_image_id: bool = True,
        image_mean: Sequence[float] = (0.5, 0.5, 0.5),
        image_std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        if downsample_mode not in {"4x", "16x"}:
            raise ValueError(f"Unsupported MiniCPM downsample mode: {downsample_mode!r}.")
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        self.patch_size = patch_size
        self.slice_mode = slice_mode
        self.downsample_mode = downsample_mode
        self.use_image_id = use_image_id
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)

    @classmethod
    def from_pretrained(cls, pretrained_path: str | Path, **overrides: Any) -> "MiniCPMV46ImageProcessor":
        config_path = Path(pretrained_path) / "preprocessor_config.json"
        if not config_path.is_file():
            raise ValueError(
                "MiniCPM-V-4.6 image preprocessing requires "
                f"{config_path}; the pinned Transformers version cannot construct it automatically."
            )
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)

        supported_keys = {
            "max_slice_nums",
            "scale_resolution",
            "patch_size",
            "slice_mode",
            "downsample_mode",
            "use_image_id",
            "image_mean",
            "image_std",
        }
        kwargs = {key: value for key, value in config.items() if key in supported_keys}
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**kwargs)

    @staticmethod
    def find_best_resize(
        image_size: tuple[int, int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        height, width = image_size
        if height * width > scale_resolution * scale_resolution or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        divisor = patch_size * 4
        return _ensure_divide(height, divisor), _ensure_divide(width, divisor)

    @classmethod
    def get_refine_size(
        cls,
        image_size: tuple[int, int],
        grid: list[int],
        scale_resolution: int,
        patch_size: int,
    ) -> tuple[int, int]:
        height, width = image_size
        grid_y, grid_x = grid
        refine_width = _ensure_divide(width, grid_x)
        refine_height = _ensure_divide(height, grid_y)
        best_height, best_width = cls.find_best_resize(
            (refine_height / grid_y, refine_width / grid_x),
            scale_resolution,
            patch_size,
            allow_upscale=True,
        )
        return best_height * grid_y, best_width * grid_x

    @staticmethod
    def get_sliced_grid(
        image_size: tuple[int, int],
        max_slice_nums: int,
        scale_resolution: int,
    ) -> list[int] | None:
        original_height, original_width = image_size
        log_ratio = math.log(original_width / original_height)
        multiple = min(
            math.ceil(original_width * original_height / (scale_resolution * scale_resolution)),
            max_slice_nums,
        )
        if multiple <= 1:
            return None

        best_grid = [1, 1]
        min_error = float("inf")
        for num_slices in (multiple - 1, multiple, multiple + 1):
            if num_slices == 1 or num_slices > max_slice_nums:
                continue
            for num_rows in range(1, num_slices + 1):
                if num_slices % num_rows != 0:
                    continue
                num_cols = num_slices // num_rows
                error = abs(log_ratio - math.log(num_rows / num_cols))
                if error < min_error:
                    best_grid = [num_cols, num_rows]
                    min_error = error
        return best_grid

    @staticmethod
    def _to_tensor(image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, Image.Image):
            return tv_functional.pil_to_tensor(image.convert("RGB"))
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(np.ascontiguousarray(image))
        if not torch.is_tensor(image):
            raise ValueError(f"Unsupported MiniCPM image type: {type(image)}.")
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.ndim != 3:
            raise ValueError(f"Expected a rank-3 MiniCPM image, got shape {tuple(image.shape)}.")
        if image.shape[0] not in {1, 3, 4} and image.shape[-1] in {1, 3, 4}:
            image = image.permute(2, 0, 1).contiguous()
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)
        elif image.shape[0] == 4:
            image = image[:3]
        if image.shape[0] != 3:
            raise ValueError(f"Expected a MiniCPM RGB image, got shape {tuple(image.shape)}.")
        return image

    @staticmethod
    def _resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tv_functional.resize(
            image,
            [height, width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        mean = [value * 255.0 for value in self.image_mean]
        std = [value * 255.0 for value in self.image_std]
        return tv_functional.normalize(image.float(), mean, std)

    @staticmethod
    def _reshape_by_patch(image: torch.Tensor, patch_size: int) -> torch.Tensor:
        channels = image.shape[0]
        patches = torch.nn.functional.unfold(
            image.unsqueeze(0),
            (patch_size, patch_size),
            stride=(patch_size, patch_size),
        )
        patches = patches.reshape(channels, patch_size, patch_size, -1)
        return patches.permute(0, 1, 3, 2).reshape(channels, patch_size, -1)

    def __call__(
        self,
        images: Image.Image | np.ndarray | torch.Tensor | Sequence[Image.Image | np.ndarray | torch.Tensor],
        *,
        return_tensors: str | None = "pt",
        **kwargs: Any,
    ) -> dict[str, Any]:
        if return_tensors not in {None, "pt"}:
            raise ValueError("MiniCPM-V-4.6 image processing only supports PyTorch tensors.")
        if isinstance(images, (Image.Image, np.ndarray)) or torch.is_tensor(images):
            images = [images]

        max_slice_nums = int(kwargs.get("max_slice_nums", self.max_slice_nums))
        scale_resolution = int(kwargs.get("scale_resolution", self.scale_resolution))
        patch_size = int(kwargs.get("patch_size", self.patch_size))
        slice_mode = bool(kwargs.get("slice_mode", self.slice_mode))

        per_image_pixel_values = []
        per_image_target_sizes = []
        grids = []
        for raw_image in images:
            image = self._to_tensor(raw_image)
            image_size = tuple(image.shape[-2:])
            best_grid = (
                self.get_sliced_grid(image_size, max_slice_nums, scale_resolution)
                if slice_mode
                else None
            )

            source_height, source_width = self.find_best_resize(
                image_size,
                scale_resolution,
                patch_size,
                allow_upscale=best_grid is None,
            )
            source = self._normalize(self._resize(image, source_height, source_width))
            image_pixel_values = [self._reshape_by_patch(source, patch_size)]
            image_target_sizes = [[source_height // patch_size, source_width // patch_size]]

            if best_grid is not None:
                refine_height, refine_width = self.get_refine_size(
                    image_size,
                    best_grid,
                    scale_resolution,
                    patch_size,
                )
                refined = self._resize(image, refine_height, refine_width)
                grid_y, grid_x = best_grid
                slice_height = refine_height // grid_y
                slice_width = refine_width // grid_x
                for top in range(0, refine_height, slice_height):
                    for left in range(0, refine_width, slice_width):
                        image_slice = refined[
                            ...,
                            top : top + slice_height,
                            left : left + slice_width,
                        ]
                        image_slice = self._normalize(image_slice)
                        image_pixel_values.append(self._reshape_by_patch(image_slice, patch_size))
                        image_target_sizes.append(
                            [slice_height // patch_size, slice_width // patch_size]
                        )

            per_image_pixel_values.append(image_pixel_values)
            per_image_target_sizes.append(image_target_sizes)
            grids.append(best_grid if best_grid is not None else [0, 0])

        if not per_image_pixel_values:
            raise ValueError("MiniCPM-V-4.6 image processing received an empty image list.")

        pixel_values = torch.cat(
            [patch for image_patches in per_image_pixel_values for patch in image_patches],
            dim=-1,
        ).unsqueeze(0)
        target_sizes = torch.tensor(
            [size for image_sizes in per_image_target_sizes for size in image_sizes],
            dtype=torch.int32,
        )
        return {
            "pixel_values": pixel_values,
            "target_sizes": target_sizes,
            "grids": grids,
            "num_patches_per_image": [len(patches) for patches in per_image_pixel_values],
        }
