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

"""Utility helpers for Gr00t N1.6 preprocessing and statistics conversion."""

from __future__ import annotations

import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms

from loongforge.models.embodied.gr00t_n1_6.embodiment_configs import (
    EMBODIMENT_STAT_CONFIGS,
    ActionRepresentation,
    ModalityConfig,
)


try:
    import albumentations as A  # noqa: N812

    ALBUMENTATIONS_AVAILABLE = True
    DualTransformBase = A.DualTransform
except ImportError:
    A = None
    ALBUMENTATIONS_AVAILABLE = False
    DualTransformBase = object


def apply_sin_cos_encoding(values: np.ndarray) -> np.ndarray:
    """Apply sin-cos encoding to input values."""
    sin_values = np.sin(values)
    cos_values = np.cos(values)
    return np.concatenate([sin_values, cos_values], axis=-1)


def nested_dict_to_numpy(data):
    """Convert nested dictionary to numpy arrays recursively."""
    if isinstance(data, dict):
        return {key: nested_dict_to_numpy(value) for key, value in data.items()}
    if isinstance(data, list):
        return np.array(data)
    return data


def normalize_values_minmax(values: np.ndarray, params: dict) -> np.ndarray:
    """Normalize values using min-max normalization."""
    min_vals = params["min"]
    max_vals = params["max"]
    normalized = np.zeros_like(values)
    mask = ~np.isclose(max_vals, min_vals)
    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / (
        max_vals[..., mask] - min_vals[..., mask]
    )
    normalized[..., mask] = 2 * normalized[..., mask] - 1
    return normalized


def unnormalize_values_minmax(normalized_values: np.ndarray, params: dict) -> np.ndarray:
    """Unnormalize values using min-max parameters."""
    min_vals = params["min"]
    max_vals = params["max"]
    range_vals = max_vals - min_vals
    return (np.clip(normalized_values, -1.0, 1.0) + 1.0) / 2.0 * range_vals + min_vals


def normalize_values_meanstd(values: np.ndarray, params: dict) -> np.ndarray:
    """Normalize values using mean-std normalization."""
    mean_vals = params["mean"]
    std_vals = params["std"]
    mask = std_vals != 0
    normalized = np.zeros_like(values)
    normalized[..., mask] = (values[..., mask] - mean_vals[..., mask]) / std_vals[..., mask]
    normalized[..., ~mask] = values[..., ~mask]
    return normalized


def unnormalize_values_meanstd(normalized_values: np.ndarray, params: dict) -> np.ndarray:
    """Unnormalize values using mean-std parameters."""
    mean_vals = params["mean"]
    std_vals = params["std"]
    mask = std_vals != 0
    unnormalized = np.zeros_like(normalized_values)
    unnormalized[..., mask] = normalized_values[..., mask] * std_vals[..., mask] + mean_vals[..., mask]
    unnormalized[..., ~mask] = normalized_values[..., ~mask]
    return unnormalized


def parse_modality_configs(
    modality_configs: dict[str, dict[str, ModalityConfig | dict]],
) -> dict[str, dict[str, ModalityConfig]]:
    """Parse modality configurations from dictionary format."""
    parsed_modality_configs: dict[str, dict[str, ModalityConfig]] = {}
    for embodiment_tag, modality_config in modality_configs.items():
        parsed_modality_configs[embodiment_tag] = {}
        for modality, config in modality_config.items():
            if isinstance(config, dict):
                parsed_modality_configs[embodiment_tag][modality] = ModalityConfig(**config)
            else:
                parsed_modality_configs[embodiment_tag][modality] = config
    return parsed_modality_configs


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


def compute_relative_action_stats(dataset, embodiment_tag: str) -> dict[str, dict[str, list]]:
    """Compute per-joint relative-action statistics from dataset episodes."""
    if embodiment_tag not in EMBODIMENT_STAT_CONFIGS:
        raise ValueError(f"Unknown embodiment: {embodiment_tag}")

    config = EMBODIMENT_STAT_CONFIGS[embodiment_tag]
    modality_meta = config["modality_meta"]
    action_modality = config["modality_config"]["action"]
    state_modality = config["modality_config"]["state"]
    action_configs = action_modality.action_configs

    action_delta_indices = np.array(action_modality.delta_indices)
    state_delta_indices = state_modality.delta_indices

    relative_joint_groups = [
        joint_group
        for joint_group, action_config in zip(action_modality.modality_keys, action_configs or [], strict=False)
        if action_config.rep == ActionRepresentation.RELATIVE
    ]

    if not relative_joint_groups:
        return {}

    all_relative_chunks = {jg: [] for jg in relative_joint_groups}

    episode_data_dict = (
        dataset.hf_dataset.to_pandas()
        if hasattr(dataset.hf_dataset, "to_pandas")
        else dataset.hf_dataset
    )
    unique_episodes = episode_data_dict["episode_index"].unique()
    for episode_idx in unique_episodes:
        episode_mask = episode_data_dict["episode_index"] == episode_idx
        episode_frames = episode_data_dict[episode_mask]

        states_list = episode_frames["observation.state"].tolist()
        actions_list = episode_frames["action"].tolist()
        states = np.array([s.numpy() if isinstance(s, torch.Tensor) else np.array(s) for s in states_list])
        actions = np.array([a.numpy() if isinstance(a, torch.Tensor) else np.array(a) for a in actions_list])

        usable_length = len(episode_frames) - action_delta_indices[-1]

        for joint_group in relative_joint_groups:
            start_idx = modality_meta["action"][joint_group]["start"]
            end_idx = modality_meta["action"][joint_group]["end"]

            state_slice = states[:, start_idx:end_idx]
            action_slice = actions[:, start_idx:end_idx]

            for i in range(usable_length):
                state_ind = state_delta_indices[-1] + i
                last_state = state_slice[state_ind]

                action_inds = action_delta_indices + i
                action_chunk = action_slice[action_inds]

                relative_chunk = action_chunk - last_state
                all_relative_chunks[joint_group].append(relative_chunk)

    relative_stats = {}
    for joint_group, chunks in all_relative_chunks.items():
        chunks_array = np.stack(chunks, axis=0)
        relative_stats[joint_group] = {
            "min": np.min(chunks_array, axis=0).tolist(),
            "max": np.max(chunks_array, axis=0).tolist(),
            "mean": np.mean(chunks_array, axis=0).tolist(),
            "std": np.std(chunks_array, axis=0).tolist(),
            "q01": np.quantile(chunks_array, 0.01, axis=0).tolist(),
            "q99": np.quantile(chunks_array, 0.99, axis=0).tolist(),
        }

    return relative_stats
