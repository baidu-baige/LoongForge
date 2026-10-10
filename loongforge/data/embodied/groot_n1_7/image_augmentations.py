# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.

"""GR00T-N1.7 image augmentation helpers.

Copyright 2024 NVIDIA. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import Sequence
import inspect
import random
import warnings

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms


_ALBUMENTATIONS_ACCEPTS_ALWAYS_APPLY = (
    "always_apply" in inspect.signature(A.BasicTransform.__init__).parameters
)


def _albumentations_transform_init_kwargs(
    p: float,
    always_apply: bool | None,
) -> dict[str, float | bool]:
    if _ALBUMENTATIONS_ACCEPTS_ALWAYS_APPLY:
        kwargs: dict[str, float | bool] = {"p": p}
        if always_apply is not None:
            kwargs["always_apply"] = always_apply
        return kwargs

    # Albumentations 2.x removed always_apply from BasicTransform.__init__.
    # Preserve the old intent when callers request it explicitly.
    return {"p": 1.0 if always_apply else p}


def _set_albumentations_seed_if_supported(transform, seed: int | None) -> None:
    """Seed Albumentations 2.x local RNGs; 1.4 uses worker-global RNGs."""
    if seed is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)


def _albumentations_integers(transform, low, high, *, size=None, dtype=None):
    generator = getattr(transform, "random_generator", None)
    if generator is not None:
        kwargs = {"size": size}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return generator.integers(low, high, **kwargs)
    kwargs = {"size": size}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return np.random.randint(low, high, **kwargs)


def _albumentations_uniform(transform, low, high):
    generator = getattr(transform, "random_generator", None)
    if generator is not None:
        return generator.uniform(low, high)
    return np.random.uniform(low, high)


class _Albumentations14Replay:
    """Reproduce the GR00T upstream Albumentations 1.4 training pipeline.

    Albumentations 2.x moved transform RNGs from the worker-global Python and
    NumPy streams into per-transform generators. Upstream GR00T uses 1.4, so
    preserving only the seed value is insufficient: it changes every crop,
    color jitter, and subsequent state-dropout decision.
    """

    use_replay = True

    def __init__(
        self,
        *,
        max_size: int,
        crop_fraction: float,
        color_jitter_params: dict | None,
    ) -> None:
        self.max_size = int(max_size)
        self.crop_fraction = float(crop_fraction)
        self.mask_transforms = []
        self._color_jitter = None
        if color_jitter_params is not None:
            self._color_jitter = A.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.0),
                contrast=color_jitter_params.get("contrast", 0.0),
                saturation=color_jitter_params.get("saturation", 0.0),
                hue=color_jitter_params.get("hue", 0.0),
                p=1.0,
            )

    @staticmethod
    def _resize_smallest(image: np.ndarray, max_size: int) -> np.ndarray:
        height, width = image.shape[:2]
        scale = max_size / min(height, width)
        if scale == 1.0:
            return image
        new_height, new_width = (round(height * scale), round(width * scale))
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def sample_replay_for_shape(self, image_shape: tuple[int, int]) -> dict:
        """Sample augmentation parameters from image geometry only."""
        # ReplayCompose 1.4 checks its p with the worker-global Python RNG even
        # when p=1.0. Basic transforms with p=1.0 do not consume this RNG.
        random.random()
        # Albumentations 1.4 normalizes max_size to a one-element sequence and
        # calls random.choice for each SmallestMaxSize, including this no-op
        # choice. Keep both draws because they precede ColorJitter sampling.
        random.choice([self.max_size])
        height, width = image_shape
        scale = self.max_size / min(height, width)
        height, width = round(height * scale), round(width * scale)
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        max_y = height - crop_height
        max_x = width - crop_width
        y_min = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
        x_min = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
        random.choice([self.max_size])

        color = None
        if self._color_jitter is not None:
            brightness = random.uniform(*self._color_jitter.brightness)
            contrast = random.uniform(*self._color_jitter.contrast)
            saturation = random.uniform(*self._color_jitter.saturation)
            hue = random.uniform(*self._color_jitter.hue)
            order = np.asarray([0, 1, 2, 3])
            order_rng = np.random.RandomState(random.randint(0, (1 << 32) - 1))
            order_rng.shuffle(order)
            color = {
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "hue": hue,
                "order": order.tolist(),
            }
        return {
            "crop_coords": (x_min, y_min, x_min + crop_width, y_min + crop_height),
            "color": color,
        }

    def sample_replay(self, image: np.ndarray) -> dict:
        """Sample augmentation parameters without applying the image transform."""
        return self.sample_replay_for_shape(image.shape[:2])

    def _apply(self, image: np.ndarray, replay: dict) -> np.ndarray:
        image = self._resize_smallest(image, self.max_size)
        x_min, y_min, x_max, y_max = replay["crop_coords"]
        image = image[y_min:y_max, x_min:x_max]
        image = self._resize_smallest(image, self.max_size)
        if replay["color"] is not None:
            image = self._color_jitter.apply(image, **replay["color"])
        return image

    def __call__(self, *, image: np.ndarray) -> dict:
        replay = self.sample_replay(image)
        return {"image": self._apply(image, replay), "replay": replay}

    def replay(self, *, image: np.ndarray, saved_augmentations: dict) -> dict:
        """Re-apply previously sampled augmentation parameters to an image."""
        return {"image": self._apply(image, saved_augmentations)}


def apply_with_replay(transform, images, masks=None, replay=None):
    """Apply an Albumentations transform to a view sequence with optional replay."""
    transformed_tensors = []
    current_replay = replay
    has_replay = transform.use_replay
    mask_transforms = transform.mask_transforms

    if masks is not None and len(masks) != len(images):
        raise ValueError(f"Number of masks ({len(masks)}) must match images ({len(images)})")

    for index, image in enumerate(images):
        image_array = np.array(image)
        mask_array = None if masks is None else np.array(masks[index])
        if mask_array is not None and mask_array.dtype == np.bool_:
            mask_array = mask_array.astype(np.uint8)

        if mask_transforms and mask_array is not None:
            for mask_transform in mask_transforms:
                result = mask_transform(image=image_array, mask=mask_array)
                image_array = result["image"]

        if has_replay:
            if current_replay is None:
                augmented_image = transform(image=image_array)
                current_replay = augmented_image["replay"]
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    augmented_image = transform.replay(
                        image=image_array,
                        saved_augmentations=current_replay,
                    )
        else:
            augmented_image = transform(image=image_array)

        image_array = augmented_image["image"]
        if image_array.dtype == np.float32:
            image_array = (image_array * 255).astype(np.uint8)
        elif image_array.dtype != np.uint8:
            raise ValueError(f"Unexpected data type: {image_array.dtype}")

        transformed_tensors.append(torch.from_numpy(image_array).permute(2, 0, 1))

    return transformed_tensors, current_replay


class MaskedColorTransform(A.ImageOnlyTransform):
    """Apply a random color tint to selected mask values."""

    def __init__(
        self,
        target_mask_values: Sequence[int],
        alpha_range: tuple[float, float] = (0.3, 1.0),
        p: float = 0.5,
        always_apply: bool | None = None,
    ):
        super().__init__(**_albumentations_transform_init_kwargs(p, always_apply))
        self.target_mask_values = list(target_mask_values)
        self.alpha_range = alpha_range

    def apply(self, img: np.ndarray, mask: np.ndarray = None, **params) -> np.ndarray:
        """Apply the transform to the current image and mask."""
        if mask is None:
            return img

        region_mask = np.zeros(mask.shape[:2], dtype=bool)
        for value in self.target_mask_values:
            region_mask |= mask == value
        if not region_mask.any():
            return img

        random_color = _albumentations_integers(self, 0, 256, size=3).astype(np.float32)
        alpha = _albumentations_uniform(self, self.alpha_range[0], self.alpha_range[1])
        result = img.copy().astype(np.float32)
        for channel in range(3):
            result[region_mask, channel] = (
                result[region_mask, channel] * (1 - alpha) + random_color[channel] * alpha
            )
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_params_dependent_on_data(self, params, data) -> dict:
        """Return transform parameters derived from the current sample."""
        return {"mask": data.get("mask")}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the init args needed for replay serialization."""
        return ("target_mask_values", "alpha_range")


class BackgroundNoiseTransform(A.ImageOnlyTransform):
    """Replace selected mask values with random RGB noise."""

    def __init__(
        self,
        p: float = 1.0,
        target_mask_values: Sequence[int] | None = None,
        always_apply: bool | None = None,
    ):
        super().__init__(**_albumentations_transform_init_kwargs(p, always_apply))
        self.target_mask_values = [0] if target_mask_values is None else list(target_mask_values)

    def apply(self, img: np.ndarray, mask: np.ndarray = None, **params) -> np.ndarray:
        """Apply the transform to the current image and mask."""
        if mask is None:
            return img

        result = img.copy()
        mask_2d = mask[..., 0] if mask.ndim == 3 else mask
        background = np.isin(mask_2d, self.target_mask_values)
        if background.any():
            noise = _albumentations_integers(self, 0, 256, size=result.shape, dtype=np.uint8)
            result[background] = noise[background]
        return result

    def get_params_dependent_on_data(self, params, data) -> dict:
        """Return transform parameters derived from the current sample."""
        return {"mask": data.get("mask")}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the init args needed for replay serialization."""
        return ("target_mask_values",)


class FractionalRandomCrop(A.DualTransform):
    """Crop a random fraction while preserving aspect ratio."""

    def __init__(
        self,
        crop_fraction: float = 0.9,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(**_albumentations_transform_init_kwargs(p, always_apply))
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params,
    ) -> np.ndarray:
        """Apply the sampled crop window to the image."""
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params,
    ) -> np.ndarray:
        """Crop bounding boxes to the sampled window."""
        return A.augmentations.crops.functional.crop_bboxes_by_coords(
            bboxes,
            crop_coords,
            params["shape"],
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params,
    ) -> np.ndarray:
        """Crop keypoints to the sampled window."""
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        """Sample crop coordinates from the current image size."""
        height, width = params["shape"][:2]
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        max_y = height - crop_height
        max_x = width - crop_width
        y_min = int(_albumentations_integers(self, 0, max_y + 1)) if max_y > 0 else 0
        x_min = int(_albumentations_integers(self, 0, max_x + 1)) if max_x > 0 else 0
        return {"crop_coords": (x_min, y_min, x_min + crop_width, y_min + crop_height)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the init args needed for replay serialization."""
        return ("crop_fraction",)


class FractionalCenterCrop(A.DualTransform):
    """Crop a centered fraction while preserving aspect ratio."""

    def __init__(
        self,
        crop_fraction: float = 0.9,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(**_albumentations_transform_init_kwargs(p, always_apply))
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params,
    ) -> np.ndarray:
        """Apply the sampled crop window to the image."""
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params,
    ) -> np.ndarray:
        """Crop bounding boxes to the sampled window."""
        return A.augmentations.crops.functional.crop_bboxes_by_coords(
            bboxes,
            crop_coords,
            params["shape"],
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params,
    ) -> np.ndarray:
        """Crop keypoints to the sampled window."""
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        """Sample crop coordinates from the current image size."""
        height, width = params["shape"][:2]
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        y_min = (height - crop_height) // 2
        x_min = (width - crop_width) // 2
        return {"crop_coords": (x_min, y_min, x_min + crop_width, y_min + crop_height)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the init args needed for replay serialization."""
        return ("crop_fraction",)


class LetterBoxPad(A.DualTransform):
    """Pad non-square images to square with black bars."""

    def __init__(self, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(**_albumentations_transform_init_kwargs(p, always_apply))

    def apply(
        self,
        img: np.ndarray,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        **params,
    ) -> np.ndarray:
        """Pad the image to a square canvas when needed."""
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return img
        return cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )

    def get_params_dependent_on_data(self, params, data) -> dict[str, int]:
        """Compute the padding needed to square the current image."""
        height, width = params["shape"][:2]
        if height == width:
            return {"pad_top": 0, "pad_bottom": 0, "pad_left": 0, "pad_right": 0}
        max_dim = max(height, width)
        pad_h = max_dim - height
        pad_w = max_dim - width
        return {
            "pad_top": pad_h // 2,
            "pad_bottom": pad_h - pad_h // 2,
            "pad_left": pad_w // 2,
            "pad_right": pad_w - pad_w // 2,
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the init args needed for replay serialization."""
        return ()


def build_image_transformations_albumentations(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge,
    crop_fraction,
    extra_augmentation_config: dict | None = None,
    letter_box_transform: bool = False,
    seed: int | None = None,
):
    """Build official GR00T-N1.7 Albumentations transforms."""
    if crop_fraction is None:
        fraction_to_use = image_crop_size[0] / image_target_size[0]
    else:
        fraction_to_use = crop_fraction

    if shortest_image_edge is None:
        max_size = image_target_size[0]
    else:
        max_size = shortest_image_edge

    extra_augmentation_config = extra_augmentation_config or {}
    use_albumentations14_replay = (
        not _ALBUMENTATIONS_ACCEPTS_ALWAYS_APPLY
        and not letter_box_transform
        and (random_rotation_angle is None or random_rotation_angle == 0)
        and not extra_augmentation_config.get("background_noise_transforms")
        and not extra_augmentation_config.get("masked_region_transforms")
    )
    if use_albumentations14_replay:
        train_transform = _Albumentations14Replay(
            max_size=max_size,
            crop_fraction=fraction_to_use,
            color_jitter_params=color_jitter_params,
        )
        eval_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
                FractionalCenterCrop(crop_fraction=fraction_to_use),
                A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
            ]
        )
        eval_transform.use_replay = False
        eval_transform.mask_transforms = []
        return train_transform, eval_transform

    train_transform_list = []
    if letter_box_transform:
        train_transform_list.append(LetterBoxPad())
    train_transform_list.extend(
        [
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
            FractionalRandomCrop(crop_fraction=fraction_to_use),
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        ]
    )

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
    train_transform.use_replay = True
    _set_albumentations_seed_if_supported(train_transform, seed)

    mask_transforms = []
    for noise_config in extra_augmentation_config.get("background_noise_transforms", []):
        mask_transforms.append(
            BackgroundNoiseTransform(
                p=float(noise_config.get("p", 1.0)),
                target_mask_values=noise_config.get("target_mask_values", [0]),
            )
        )
    for transform_config in extra_augmentation_config.get("masked_region_transforms", []):
        mask_transforms.append(
            MaskedColorTransform(
                target_mask_values=transform_config.get("target_mask_values", []),
                alpha_range=tuple(transform_config.get("alpha_range", [0.3, 1.0])),
                p=transform_config.get("p", 0.5),
            )
        )
    if seed is not None:
        for index, mask_transform in enumerate(mask_transforms):
            _set_albumentations_seed_if_supported(mask_transform, seed + index + 1)
    train_transform.mask_transforms = mask_transforms

    eval_transform_list = []
    if letter_box_transform:
        eval_transform_list.append(LetterBoxPad())
    eval_transform_list.extend(
        [
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
            FractionalCenterCrop(crop_fraction=fraction_to_use),
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        ]
    )
    eval_transform = A.Compose(eval_transform_list)
    eval_transform.use_replay = False
    eval_transform.mask_transforms = []
    _set_albumentations_seed_if_supported(eval_transform, seed)

    return train_transform, eval_transform


class LetterBoxTransform:
    """Torchvision equivalent of letterbox padding."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        *leading_dims, channels, height, width = img.shape
        if height == width:
            return img

        max_dim = max(height, width)
        pad_h = max_dim - height
        pad_w = max_dim - width
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if leading_dims:
            batch_size = torch.tensor(leading_dims).prod().item()
            img = img.reshape(batch_size, channels, height, width)
            padded_img = transforms.functional.pad(
                img,
                padding=[pad_left, pad_top, pad_right, pad_bottom],
                fill=0,
            )
            return padded_img.reshape(leading_dims + [channels, max_dim, max_dim])

        return transforms.functional.pad(
            img,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=0,
        )


def build_image_transformations(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge: int | None = None,
    crop_fraction: float | None = None,
):
    """Build torchvision train/eval image transforms."""
    if image_target_size is None:
        image_target_size = [shortest_image_edge or 256, shortest_image_edge or 256]
    if image_crop_size is None:
        fraction_to_use = 0.95 if crop_fraction is None else crop_fraction
        crop_size = int(image_target_size[0] * fraction_to_use)
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
            transforms.RandomRotation(degrees=[-random_rotation_angle, random_rotation_angle])
        )
    if color_jitter_params is not None:
        transform_list.append(transforms.ColorJitter(**color_jitter_params))

    return transforms.Compose(transform_list), transforms.Compose(
        [
            transforms.ToImage(),
            LetterBoxTransform(),
            transforms.Resize(size=image_target_size),
            transforms.CenterCrop(size=image_crop_size),
            transforms.Resize(size=image_target_size),
        ]
    )
