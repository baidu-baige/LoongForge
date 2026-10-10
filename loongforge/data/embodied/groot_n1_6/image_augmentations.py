# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.
#
# Copyright 2024 NVIDIA. All rights reserved.
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

"""Image augmentations for Gr00t N1.6 data transforms."""

from __future__ import annotations

import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms


try:
    import albumentations as A  # noqa: N812

    ALBUMENTATIONS_AVAILABLE = True
    DualTransformBase = A.DualTransform
except ImportError:
    A = None
    ALBUMENTATIONS_AVAILABLE = False
    DualTransformBase = object


def apply_with_replay(transform, images, replay=None):
    """Apply transformation with replay support."""
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for apply_with_replay")

    transformed_tensors = []
    current_replay = replay
    has_replay = hasattr(transform, "replay")

    for img in images:
        if has_replay:
            if current_replay is None:
                augmented_image = transform(image=np.array(img))
                current_replay = augmented_image["replay"]
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    augmented_image = transform.replay(image=np.array(img), saved_augmentations=current_replay)
            img_array = augmented_image["image"]
        else:
            augmented_image = transform(image=np.array(img))
            img_array = augmented_image["image"]

        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        elif img_array.dtype != np.uint8:
            raise ValueError(f"Unexpected data type: {img_array.dtype}")

        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        transformed_tensors.append(img_tensor)

    return transformed_tensors, current_replay


class FractionalRandomCrop(DualTransformBase):
    """Fractional random crop transformation."""

    def __init__(self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None):
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required for FractionalRandomCrop")
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params) -> np.ndarray:
        """Apply crop to image"""
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params):
        """Apply crop to bounding boxes"""
        return A.augmentations.crops.functional.crop_bboxes_by_coords(bboxes, crop_coords, params["shape"])

    def apply_to_keypoints(self, keypoints: np.ndarray, crop_coords: tuple[int, int, int, int], **params):
        """Apply crop to keypoints"""
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        """Get crop coordinates based on image dimensions"""
        image_shape = params["shape"][:2]
        height, width = image_shape
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        max_y = height - crop_height
        max_x = width - crop_width
        y_min = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x_min = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        return {"crop_coords": (x_min, y_min, x_min + crop_width, y_min + crop_height)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return transform initialization arguments names"""
        return ("crop_fraction",)


class FractionalCenterCrop(DualTransformBase):
    """Fractional center crop transform"""

    def __init__(self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None):
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required for FractionalCenterCrop")
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params) -> np.ndarray:
        """Apply center crop to image"""
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params):
        """Apply center crop to bounding boxes"""
        return A.augmentations.crops.functional.crop_bboxes_by_coords(bboxes, crop_coords, params["shape"])

    def apply_to_keypoints(self, keypoints: np.ndarray, crop_coords: tuple[int, int, int, int], **params):
        """Apply center crop to keypoints"""
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        """Get center crop coordinates based on image dimensions"""
        image_shape = params["shape"][:2]
        height, width = image_shape
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        y_min = (height - crop_height) // 2
        x_min = (width - crop_width) // 2
        return {"crop_coords": (x_min, y_min, x_min + crop_width, y_min + crop_height)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return transform initialization arguments names"""
        return ("crop_fraction",)


def build_image_transformations_albumentations(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge,
    crop_fraction,
):
    """Build image transformations using albumentations"""
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for build_image_transformations_albumentations")

    fraction_to_use = image_crop_size[0] / image_target_size[0] if crop_fraction is None else crop_fraction
    max_size = image_target_size[0] if shortest_image_edge is None else shortest_image_edge

    train_transform_list = [
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        FractionalRandomCrop(crop_fraction=fraction_to_use),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
    ]

    if random_rotation_angle is not None and random_rotation_angle != 0:
        train_transform_list.append(A.Rotate(limit=random_rotation_angle, p=1.0))

    if color_jitter_params is not None:
        train_transform_list.append(
            A.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.0),
                contrast=color_jitter_params.get("contrast", 0.0),
                saturation=color_jitter_params.get("saturation", 0.0),
                hue=color_jitter_params.get("hue", 0.0),
                p=1.0,
            )
        )

    train_transform = A.ReplayCompose(train_transform_list, p=1.0)

    eval_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
            FractionalCenterCrop(crop_fraction=fraction_to_use),
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        ]
    )

    return train_transform, eval_transform


class LetterBoxTransform:
    """Letterbox transform for maintaining aspect ratio"""
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        *leading_dims, c, h, w = img.shape
        if h == w:
            return img
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if leading_dims:
            batch_size = torch.tensor(leading_dims).prod().item()
            img_reshaped = img.reshape(batch_size, c, h, w)
            padded_img = transforms.functional.pad(
                img_reshaped, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
            )
            output_shape = leading_dims + [c, max_dim, max_dim]
            padded_img = padded_img.reshape(output_shape)
        else:
            padded_img = transforms.functional.pad(
                img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
            )
        return padded_img


def build_image_transformations(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge: int = 256,
    crop_fraction: float = 0.95,
):
    """Build train/eval torchvision image transforms for Gr00t pipelines."""
    if isinstance(color_jitter_params, str):
        parts = color_jitter_params.strip().split()
        if len(parts) % 2 != 0:
            raise ValueError(
                "color_jitter_params string must contain key/value pairs, got: "
                f"{color_jitter_params}"
            )
        color_jitter_params = {
            parts[i]: float(parts[i + 1]) for i in range(0, len(parts), 2)
        }
    if image_target_size is None:
        image_target_size = [shortest_image_edge, shortest_image_edge]

    if image_crop_size is None:
        crop_size = int(image_target_size[0] * crop_fraction)
        image_crop_size = [crop_size, crop_size]

    transform_list = [
        transforms.ToImage(),
        LetterBoxTransform(),
        transforms.Resize(size=image_target_size),
        transforms.RandomCrop(size=image_crop_size),
        transforms.Resize(size=image_target_size),
    ]
    if random_rotation_angle is not None and random_rotation_angle != 0:
        transform_list.append(
            transforms.RandomRotation(
                degrees=[-random_rotation_angle, random_rotation_angle]
            )
        )
    if color_jitter_params is not None:
        transform_list.append(transforms.ColorJitter(**color_jitter_params))
    train_image_transform = transforms.Compose(transform_list)

    eval_image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            LetterBoxTransform(),
            transforms.Resize(size=image_target_size),
            transforms.CenterCrop(size=image_crop_size),
            transforms.Resize(size=image_target_size),
        ]
    )
    return train_image_transform, eval_image_transform
