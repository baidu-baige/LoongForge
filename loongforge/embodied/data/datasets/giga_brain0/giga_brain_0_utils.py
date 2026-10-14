# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from giga-models (https://github.com/open-gigaai/giga-models)
# giga_models/pipelines/vla/giga_brain_0/giga_brain_0_utils.py, unmodified, so
# GigaBrain-0's data preprocessing numerics (normalization / delta actions /
# image transform / prompt tokenization) match the reference training exactly.
# Licensed under the Apache License, Version 2.0.

import math
import random
from typing import Any

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoProcessor, AutoTokenizer


def _load_fast_action_processor(fast_tokenizer_path: str):
    """Load the FAST action tokenizer (physical-intelligence/fast UniversalActionProcessor).

    transformers<5 loads it directly through ``AutoProcessor`` + trust_remote_code.
    transformers>=5 changed ``ProcessorMixin`` sub-processor auto-loading, which
    mis-resolves the custom ``bpe_tokenizer`` attribute as a subfolder tokenizer and
    fails. In that case we construct the processor manually: load its ``bpe_tokenizer``
    from the repo root and instantiate the class with the params from
    ``processor_config.json``. Both paths yield bit-identical tokenization.
    """
    try:
        return AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
    except Exception:
        import json
        import os

        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        with open(os.path.join(fast_tokenizer_path, 'processor_config.json')) as f:
            cfg = json.load(f)
        processor_cls = get_class_from_dynamic_module(
            'processing_action_tokenizer.UniversalActionProcessor', fast_tokenizer_path
        )
        bpe_tokenizer = AutoTokenizer.from_pretrained(fast_tokenizer_path)
        return processor_cls(
            bpe_tokenizer,
            scale=cfg['scale'],
            vocab_size=cfg['vocab_size'],
            min_token=cfg['min_token'],
            action_dim=cfg.get('action_dim'),
            time_horizon=cfg.get('time_horizon'),
        )


class Normalize:
    """Normalizes a tensor using mean/std or quantile-based scaling."""

    def __init__(self, stats: dict[int, dict[str, list[float]]], *, use_quantiles: bool = False, enable_clamp: bool = False):
        """Initializes the normalization transform.

        Args:
            stats: A dictionary mapping embodiment IDs to normalization statistics.
            use_quantiles: If True, use 1% and 99% quantiles for scaling.
                Otherwise, use mean and standard deviation.
            enable_clamp: If True, clamp the output to [-1, 1].
        """
        self.EPSILON = 1e-6
        self.use_quantiles = use_quantiles
        self.enable_clamp = enable_clamp

        required_attrs = ['mean', 'std']
        if self.use_quantiles:
            required_attrs = ['q01', 'q99']

        for attr in required_attrs:
            for key in stats:
                if attr not in stats[key]:
                    raise AttributeError(f'stats object is missing the following attribute: {attr}')

        if self.use_quantiles:
            self.q01 = dict()
            self.q99 = dict()
            for key in stats:
                self.q01[int(key)] = torch.tensor(stats[key]['q01'], dtype=torch.float32)
                self.q99[int(key)] = torch.tensor(stats[key]['q99'], dtype=torch.float32)
        else:
            self.mean = dict()
            self.std = dict()
            for key in stats:
                self.mean[int(key)] = torch.tensor(stats[key]['mean'], dtype=torch.float32)
                self.std[int(key)] = torch.tensor(stats[key]['std'], dtype=torch.float32)

    def to(self, device: str | torch.device):
        if self.use_quantiles:
            for key in self.q01:
                self.q01[key] = self.q01[key].to(device)
            for key in self.q99:
                self.q99[key] = self.q99[key].to(device)
        else:
            for key in self.mean:
                self.mean[key] = self.mean[key].to(device)
            for key in self.std:
                self.std[key] = self.std[key].to(device)
        return self

    def __call__(self, x: torch.Tensor, embodiment_id: int = 0) -> torch.Tensor:
        """Applies normalization to the input tensor.

        Args:
            x: The input tensor to normalize.
            embodiment_id: The embodiment ID to use for selecting normalization stats.

        Returns:
            The normalized tensor.
        """
        x_dim = x.shape[-1]
        if self.use_quantiles:
            x = (x - self.q01[embodiment_id][..., :x_dim]) / (
                self.q99[embodiment_id][..., :x_dim] - self.q01[embodiment_id][..., :x_dim] + self.EPSILON
            ) * 2.0 - 1.0
        else:
            x = (x - self.mean[embodiment_id][..., :x_dim]) / (self.std[embodiment_id][..., :x_dim] + self.EPSILON)

        if self.enable_clamp:
            x = x.clamp(-1.0, 1.0)

        return x


class Unnormalize:
    """Unnormalizes a tensor using mean/std or quantile-based scaling."""

    def __init__(self, stats: dict[int, dict[str, list[float]]], *, use_quantiles: bool = False):
        """Initializes the unnormalization transform.

        Args:
            stats: A dictionary mapping embodiment IDs to normalization statistics.
            use_quantiles: If True, use 1% and 99% quantiles for scaling.
                Otherwise, use mean and standard deviation.
        """
        self.EPSILON = 1e-6
        self.stats = stats
        self.use_quantiles = use_quantiles

        required_attrs = ['mean', 'std']
        if self.use_quantiles:
            required_attrs = ['q01', 'q99']

        for attr in required_attrs:
            for key in stats:
                if attr not in stats[key]:
                    raise AttributeError(f'stats object is missing the following attribute: {attr}')

        if self.use_quantiles:
            self.q01 = dict()
            self.q99 = dict()
            for key in stats:
                self.q01[int(key)] = torch.tensor(stats[key]['q01'], dtype=torch.float32)
                self.q99[int(key)] = torch.tensor(stats[key]['q99'], dtype=torch.float32)
        else:
            self.mean = dict()
            self.std = dict()
            for key in stats:
                self.mean[int(key)] = torch.tensor(stats[key]['mean'], dtype=torch.float32)
                self.std[int(key)] = torch.tensor(stats[key]['std'], dtype=torch.float32)

    def to(self, device: str | torch.device):
        if self.use_quantiles:
            for key in self.q01:
                self.q01[key] = self.q01[key].to(device)
            for key in self.q99:
                self.q99[key] = self.q99[key].to(device)
        else:
            for key in self.mean:
                self.mean[key] = self.mean[key].to(device)
            for key in self.std:
                self.std[key] = self.std[key].to(device)
        return self

    def __call__(self, x: torch.Tensor, embodiment_id: int = 0) -> torch.Tensor:
        """Applies unnormalization to the input tensor.

        Args:
            x: The input tensor to unnormalize.
            embodiment_id: The embodiment ID to use for selecting normalization stats.

        Returns:
            The unnormalized tensor.
        """
        x_dim = x.shape[-1]
        if self.use_quantiles:
            return (x + 1.0) / 2.0 * (self.q99[embodiment_id][..., :x_dim] - self.q01[embodiment_id][..., :x_dim] + self.EPSILON) + self.q01[
                embodiment_id
            ][..., :x_dim]
        else:
            return x * (self.std[embodiment_id][..., :x_dim] + self.EPSILON) + self.mean[embodiment_id][..., :x_dim]


class DeltaActions:
    """Repacks absolute actions into delta action space."""

    def __init__(self, mask: dict[int, list[bool]]):
        self.mask = dict()
        assert mask is not None, 'mask is required'
        for key in mask:
            self.mask[int(key)] = torch.tensor(mask[key])

    def to(self, device: str | torch.device):
        for key in self.mask:
            self.mask[key] = self.mask[key].to(device)
        return self

    def __call__(self, data: dict) -> dict:
        if 'action' not in data or 'observation.state' not in data:
            return data

        embodiment_id = data['embodiment_id']

        dims = self.mask[embodiment_id].shape[-1]
        state, action = data['observation.state'], data['action']
        action[..., :dims] -= torch.where(self.mask[embodiment_id], state[..., :dims], torch.zeros_like(state[..., :dims])).unsqueeze(-2)
        data['action'] = action
        return data


class AbsoluteActions:
    """Repacks delta actions into absolute action space."""

    def __init__(self, mask: dict[int, list[bool]]):
        self.mask = dict()
        assert mask is not None, 'mask is required'
        for key in mask:
            self.mask[int(key)] = torch.tensor(mask[key])

    def to(self, device: str | torch.device):
        for key in self.mask:
            self.mask[key] = self.mask[key].to(device)
        return self

    def __call__(self, data: dict) -> dict:
        if 'action' not in data or 'observation.state' not in data:
            return data

        embodiment_id = data['embodiment_id']

        state, action = data['observation.state'], data['action']
        dims = self.mask[embodiment_id].shape[-1]
        action[..., :dims] += torch.where(self.mask[embodiment_id], state[..., :dims], torch.zeros_like(state[..., :dims])).unsqueeze(-2)
        data['action'] = action
        return data


class PadStatesAndActions:
    """Zero-pads states and actions to the model action dimension."""

    def __init__(self, action_dim: int):
        """Initializes the padding transform.

        Args:
            action_dim: The target dimension to pad to.
        """
        self.action_dim = action_dim

    def _pad_to_dim(self, x: torch.Tensor, target_dim: int, axis: int = -1) -> torch.Tensor:
        """Pad an array to the target dimension with zeros along the specified
        axis."""
        current_dim = x.shape[axis]
        if current_dim < target_dim:
            shape = list(x.shape)
            shape[-1] = target_dim
            new_vector = torch.zeros(*shape, dtype=x.dtype, device=x.device)
            new_vector[..., :current_dim] = x
            x = new_vector
        return x

    def __call__(self, data: dict) -> dict:
        """Pads 'observation.state' and 'action' tensors in the data dict.

        Args:
            data: A dictionary containing 'observation.state' and optionally 'action'.

        Returns:
            The data dictionary with padded tensors.
        """
        data['observation.state'] = self._pad_to_dim(data['observation.state'], self.action_dim, axis=-1)
        if 'action' in data:
            data['action'] = self._pad_to_dim(data['action'], self.action_dim, axis=-1)
        return data


# This function is used for the original PaliGemma model inference.
def resize_image(img: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize an image to the given (width, height) without preserving aspect
    ratio.

    Args:
        img: Input image, shape (C, H, W), with values typically in [0, 1].
        width: Target width (W).
        height: Target height (H).

    Returns:
        A torch.Tensor of shape (C, height, width).
    """
    # Validate input dimensions
    if img.ndim != 3:
        raise ValueError(f'(C,H,W) expected, but got {img.shape}')

    resized_img = F.interpolate(img.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
    return resized_img


def resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = -1.0) -> tuple[torch.Tensor, dict]:
    """Resize an image to fit inside the given (width, height) while preserving
    aspect ratio, then pad with the specified value so that the final image
    exactly matches the target size.

    Args:
        img: Input image, shape (C, H, W), with values typically in [0, 1].
        width: Target width (W).
        height: Target height (H).
        pad_value: Value to use for padding, defaults to -1.

    Returns:
        A tuple containing:
        - A torch.Tensor of shape (C, height, width).
        - A dictionary with transformation parameters.
    """
    # Validate input dimensions
    if img.ndim != 3:
        raise ValueError(f'(C,H,W) expected, but got {img.shape}')

    cur_height, cur_width = img.shape[1:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(img.unsqueeze(0), size=(resized_height, resized_width), mode='bilinear', align_corners=False).squeeze(0)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded_img = F.pad(resized_img, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    transform_params = {
        'original_size': (cur_width, cur_height),
        'ratio': ratio,
        'padding': (pad_left, pad_top),
    }
    return padded_img, transform_params


class RandomPoseTransform:
    """Applies a random crop, resize, and rotation to an image."""

    def __init__(self, crop_size: tuple[int, int], resize_size: tuple[int, int], rotation_degrees: tuple[float, float]):
        """Initializes the random pose transform.

        Args:
            crop_size: The size (h, w) of the random crop.
            resize_size: The size (h, w) to resize the cropped image to.
            rotation_degrees: The range of degrees for random rotation.
        """
        self.crop_size_h, self.crop_size_w = crop_size
        self.resize_size_h, self.resize_size_w = resize_size
        self.rotation_degrees = rotation_degrees

    def generate_params(self, h: int, w: int) -> dict[str, Any]:
        """Generates random transform parameters for an image of a given size.

        Args:
            h: Image height.
            w: Image width.

        Returns:
            A dictionary of transform parameters.
        """
        if h < self.crop_size_h or w < self.crop_size_w:
            raise ValueError(f'Required crop size {(self.crop_size_h, self.crop_size_w)} is larger than input image size {(h, w)}')
        i = torch.randint(0, h - self.crop_size_h + 1, size=(1,)).item()
        j = torch.randint(0, w - self.crop_size_w + 1, size=(1,)).item()
        crop_box = (j, i, self.crop_size_w, self.crop_size_h)  # x,y,w,h
        angle = transforms.RandomRotation.get_params(self.rotation_degrees)
        return {
            'crop_box': crop_box,
            'crop_size': (self.crop_size_w, self.crop_size_h),
            'resize_size': (self.resize_size_w, self.resize_size_h),
            'angle': angle,
        }

    def apply_with_params(self, img: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        """Applies the transform to an image using pre-generated parameters.

        Args:
            img: The input image.
            params: A dictionary of transform parameters from `generate_params`.

        Returns:
            The transformed image.
        """
        j, i, tw, th = params['crop_box']
        img = transforms.functional.crop(img, i, j, th, tw)
        img = transforms.functional.resize(img, (self.resize_size_h, self.resize_size_w))
        if params.get('angle') is not None:
            img = transforms.functional.rotate(img, params['angle'])
        return img

    def __call__(self, img: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Applies random transformations to the image.

        Args:
            img: The input image.

        Returns:
            A tuple of the transformed image and the applied parameters.
        """
        h, w = img.shape[-2:]
        params = self.generate_params(h, w)
        transformed_img = self.apply_with_params(img, params)
        return transformed_img, params


class ImageTransform:
    """Preprocesses a dictionary of images with optional augmentation."""

    def __init__(
        self,
        is_train: bool,
        resize_imgs_with_padding: tuple[int, int],
        present_img_keys: list[str] | None = None,
        enable_image_aug: bool = False,
        enable_depth_img: bool = False,
        depth_img_prefix_name: str | None = None,
        depth_img_mask_ratio: float = 0.5,
    ):
        """Initializes the image transform pipeline.

        Args:
            is_train: Whether the transform is used for training.
            resize_imgs_with_padding: Target size (w, h) for resizing with padding.
            present_img_keys: List of image keys to process from the input data dict.
            enable_image_aug: If True, applies color jitter and random pose transforms.
            enable_depth_img: If True, concatenates depth images to RGB images.
            depth_img_prefix_name: The prefix key for depth images in the data dict.
            depth_img_mask_ratio: The ratio of depth images to mask out during training.
        """
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.present_img_keys = present_img_keys
        if self.present_img_keys is None:
            self.present_img_keys = [
                'observation.images.cam_high',
                'observation.images.cam_left_wrist',
                'observation.images.cam_right_wrist',
            ]
        self.enable_image_aug = enable_image_aug
        self.width, self.height = resize_imgs_with_padding
        self.enable_depth_img = enable_depth_img
        self.depth_img_prefix_name = depth_img_prefix_name
        self.depth_img_mask_ratio = depth_img_mask_ratio if is_train else 0.0

        if self.enable_image_aug:
            self.color_jitter_transform = transforms.ColorJitter(
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
            )
            self.pose_transform = RandomPoseTransform(
                crop_size=(int(self.height * 0.95), int(self.width * 0.95)),
                resize_size=(self.height, self.width),
                rotation_degrees=(-5, 5),
            )

    def __call__(self, data: dict) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
        """Preprocesses input images from a data dictionary.

        This transform selects images based on `present_img_keys`, optionally
        concatenates depth, applies augmentations if in training mode, resizes
        and pads, and normalizes pixel values to [-1, 1].

        Args:
            data: A dictionary containing image data.

        Returns:
            A tuple containing:
            - images: The list of processed image tensors (C, H, W).
            - img_masks: A list of boolean masks, one for each image.
            - image_transform_params: A dictionary of applied transformation parameters.
        """
        images = []
        img_masks = []
        image_transform_params = {}

        for key in self.present_img_keys:
            if key not in data:
                raise ValueError(f'{key} not found in data. Please check the present_img_keys in the config or the dataset.')

            img = data[key]
            if self.enable_depth_img:
                assert self.depth_img_prefix_name is not None, 'depth_img_prefix_name is required'
                depth_img_key = key.replace('observation.images', self.depth_img_prefix_name)
                if depth_img_key in data and random.random() >= self.depth_img_mask_ratio:
                    depth_img = data[depth_img_key][0:1]
                else:
                    depth_img = torch.zeros_like(img[0:1])
                img = torch.cat([img, depth_img], dim=0)

            # [C, H, W] -> preprocess
            if self.resize_imgs_with_padding is not None:
                target_w, target_h = self.resize_imgs_with_padding
                original_h, original_w = img.shape[-2:]
                if original_h != target_h or original_w != target_w:
                    # in original PaliGemma model, the image is resized without padding. But we use padding here for better 3D perception capability.
                    img, rwp_params = resize_with_pad(img, *self.resize_imgs_with_padding, pad_value=0)
                    if key == 'observation.images.cam_high':
                        image_transform_params['resize_with_pad'] = rwp_params

            if self.enable_image_aug:
                if key == 'observation.images.cam_high':
                    img, pose_params = self.pose_transform(img)
                    image_transform_params['pose_transform'] = pose_params
                img[:3, :, :] = self.color_jitter_transform(img[:3, :, :])

            # Normalize pixel values to [-1, 1]. The original PaliGemma model uses this normalization (mean = 0.5, std = 0.5).
            img = img * 2.0 - 1.0

            images.append(img)
            img_masks.append(torch.tensor(True, dtype=torch.bool, device=img.device))

        return images, img_masks, image_transform_params


class TrajectoryTransform:
    """Transforms 2D trajectory data, including coordinate adjustments and
    normalization."""

    def __init__(self, step_interval: int | None = None, minmax_value: list[float] | None = None):
        """Initializes the trajectory transform.

        Args:
            step_interval: Interval for subsampling trajectory points.
            minmax_value: A list or tuple of [x_min, y_min, x_max, y_max] for
                clipping and normalization.
        """
        self.step_interval = step_interval
        self.minmax_value = minmax_value
        if minmax_value is not None:
            assert minmax_value[2] > 0 and minmax_value[3] > 0, 'x_max and y_max must be greater than 0'
            self.min_value = torch.tensor([minmax_value[0], minmax_value[1], minmax_value[0], minmax_value[1]])
            self.max_value = torch.tensor([minmax_value[2], minmax_value[3], minmax_value[2], minmax_value[3]])
        else:
            self.min_value = None
            self.max_value = None
        self.traj_size = 4  # x_left, y_left, x_right, y_right

    def to(self, device: str | torch.device):
        if self.min_value is not None:
            self.min_value = self.min_value.to(device)
        if self.max_value is not None:
            self.max_value = self.max_value.to(device)
        return self

    def __call__(self, data: dict, chunk_size: int, image_transform_params: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes 2D trajectory data from a dictionary.

        This includes subsampling, applying inverse geometric transforms from image
        augmentations, and normalizing coordinates.

        Args:
            data: A dictionary containing trajectory data.
            chunk_size: The size of the trajectory chunk.
            image_transform_params: Optional dictionary of image transforms to invert.

        Returns:
            A tuple of (transformed trajectory, padding mask).
        """
        if 'perception.2d_traj' not in data or 'perception.2d_traj_is_pad' not in data:
            traj_chunk_size = (chunk_size // self.step_interval) if self.step_interval is not None else chunk_size
            return -torch.ones(traj_chunk_size, self.traj_size, dtype=torch.float32), torch.ones(traj_chunk_size, self.traj_size, dtype=torch.bool)

        if self.step_interval is not None:
            traj = data['perception.2d_traj'][:: self.step_interval]
            traj_is_pad = data['perception.2d_traj_is_pad'][:: self.step_interval]
        else:
            traj = data['perception.2d_traj']
            traj_is_pad = data['perception.2d_traj_is_pad']

        traj[torch.isnan(traj)] = -100

        if image_transform_params is not None:
            coords = traj.view(-1, 2, 2)

            if 'resize_with_pad' in image_transform_params:
                rwp = image_transform_params['resize_with_pad']
                ratio = rwp['ratio']
                pad_x, pad_y = rwp['padding']
                coords = coords / ratio
                coords[..., 0] += pad_x
                coords[..., 1] += pad_y

            if 'pose_transform' in image_transform_params:
                pose_p = image_transform_params['pose_transform']

                if pose_p.get('crop_box'):
                    crop_x, crop_y, _, _ = pose_p['crop_box']
                    coords[..., 0] -= crop_x
                    coords[..., 1] -= crop_y

                crop_w, crop_h = pose_p['crop_size']
                resize_w, resize_h = pose_p['resize_size']
                if crop_w > 0 and crop_h > 0:
                    scale_x = resize_w / crop_w
                    scale_y = resize_h / crop_h
                    coords[..., 0] *= scale_x
                    coords[..., 1] *= scale_y

                if pose_p.get('angle') is not None:
                    angle_rad = -math.radians(pose_p['angle'])
                    cos_a = math.cos(angle_rad)
                    sin_a = math.sin(angle_rad)

                    center_x, center_y = resize_w / 2, resize_h / 2

                    coords[..., 0] -= center_x
                    coords[..., 1] -= center_y

                    x_new = coords[..., 0] * cos_a - coords[..., 1] * sin_a
                    y_new = coords[..., 0] * sin_a + coords[..., 1] * cos_a
                    coords[..., 0] = x_new
                    coords[..., 1] = y_new

                    coords[..., 0] += center_x
                    coords[..., 1] += center_y

            traj = coords.view(-1, self.traj_size)

        traj_is_pad = traj_is_pad[:, None].expand(traj_is_pad.shape[0], self.traj_size)
        if self.minmax_value is not None:
            # set traj_is_pad to True if the traj value is out of the minmax value
            traj_is_pad = traj_is_pad | (traj < self.min_value[None, ...]) | (traj > self.max_value[None, ...])
            traj = traj.clamp(self.min_value[None, ...], self.max_value[None, ...])
            traj = traj / self.max_value[None, ...]
        return traj, traj_is_pad


class PromptTokenizerTransform:
    """Encodes task, state, and action information into token sequences for the
    policy model."""

    def __init__(
        self,
        is_train: bool,
        tokenizer_model_path: str,
        fast_tokenizer_path: str,
        max_length: int,
        discrete_state_input: bool = True,
        encode_action_input: bool = False,
        encoded_action_horizon: int | None = None,
        encode_sub_task_input: bool = False,
        text_token_length: int | None = 257152,
        autoregressive_inference_mode: bool = False,
        sample_ratios: dict | None = None,
    ):
        """Initializes the prompt and tokenizer transform.

        Args:
            is_train: Whether the transform is used for training. This affects
                whether random sampling of prompt formats is used.
            tokenizer_model_path: Path to the main tokenizer model.
            fast_tokenizer_path: Path to the fast tokenizer for actions.
            max_length: Maximum sequence length for padding.
            discrete_state_input: If True, discretize and include state in the prompt.
            encode_action_input: If True, encode actions into the prompt.
            encoded_action_horizon: The horizon for downsampling actions before encoding.
            encode_sub_task_input: If True, include sub-tasks in the prompt.
            text_token_length: The vocabulary size to consider for text tokens.
            autoregressive_inference_mode: If True, configure for autoregressive inference.
            sample_ratios: A dictionary of ratios for sampling different prompt formats.
        """
        self.is_train = is_train
        self.device = 'cpu'
        self.paligemma_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        self.paligemma_tokenizer.add_bos_token = True
        self.processor = AutoProcessor.from_pretrained(tokenizer_model_path)
        self.fast_tokenizer = _load_fast_action_processor(fast_tokenizer_path)

        self.encode_action_input = encode_action_input
        self.discrete_state_input = discrete_state_input
        self.encode_sub_task_input = encode_sub_task_input

        self.encoded_action_horizon = encoded_action_horizon
        self.fast_skip_tokens = 128
        self.max_length = max_length
        self.text_token_length = text_token_length

        self.autoregressive_inference_mode = autoregressive_inference_mode

        self.sample_generator = None
        if is_train and sample_ratios is not None:
            self.sample_generator = SampleGenerator(sample_ratios)

    def to(self, device: str | torch.device):
        self.device = device
        return self

    def encode_action(self, action: torch.Tensor) -> dict:
        """Encodes a continuous action tensor into a sequence of discrete
        tokens.

        Args:
            action: The action tensor to encode.

        Returns:
            A dictionary containing 'input_ids' and 'attention_mask' for the
            encoded action.
        """
        if self.encoded_action_horizon is not None:
            action_len = action.shape[1]
            horizon = int(self.encoded_action_horizon)
            assert action_len % horizon == 0, 'Action length must be divisible by encoded action horizon'
            step = action_len // horizon
            selected_indices = torch.arange(step - 1, action_len, step, device=action.device)
            action = action[:, selected_indices, :]

        batch_tokens = self.fast_tokenizer(action.to(torch.float32))
        fast_out = self.processor.tokenizer.pad({'input_ids': batch_tokens}, return_tensors='pt')

        # Assuming batch size is 1.
        act_ids = fast_out['input_ids'].squeeze(0)
        act_mask = fast_out['attention_mask'].squeeze(0)

        # Remap action tokens to the PaliGemma token space.
        vocab_size = self.paligemma_tokenizer.vocab_size
        if self.text_token_length is not None:
            vocab_size = min(vocab_size, self.text_token_length)

        act_ids = vocab_size - 1 - self.fast_skip_tokens - act_ids
        act_ids[act_mask == 0] = self.paligemma_tokenizer.pad_token_id

        # Prepare BOS, separator, and EOS tokens.
        bos = self.paligemma_tokenizer('Action: ', add_special_tokens=False, return_tensors='pt')
        eos = self.paligemma_tokenizer('|<eos>', add_special_tokens=False, return_tensors='pt')

        # Concatenate all parts to form the final sequence.
        final_act_ids = torch.cat(
            [
                bos['input_ids'].squeeze(0).to(act_ids.device),
                act_ids,
                eos['input_ids'].squeeze(0).to(act_ids.device),
            ],
            dim=0,
        )

        final_act_mask = torch.cat(
            [
                bos['attention_mask'].squeeze(0).to(act_mask.device),
                act_mask,
                eos['attention_mask'].squeeze(0).to(act_mask.device),
            ],
            dim=0,
        )

        return {'input_ids': final_act_ids, 'attention_mask': final_act_mask}

    def encode_sub_task(self, sub_task: str, add_eos: bool = True) -> dict:
        """Encodes a sub-task string into a sequence of tokens.

        Args:
            sub_task: The sub-task string to encode.
            add_eos: If True, append an EOS token.

        Returns:
            A dictionary containing 'input_ids' and 'attention_mask' for the
            encoded sub-task.
        """
        bos = self.paligemma_tokenizer('Subtask: ', add_special_tokens=False, return_tensors='pt')
        subtask_out = self.paligemma_tokenizer(
            [sub_task],
            add_special_tokens=False,
            return_tensors='pt',
            padding='longest',
            truncation=False,
        )
        final_subtask_ids = torch.cat(
            [
                bos['input_ids'].squeeze(0),
                subtask_out['input_ids'].squeeze(0),
            ],
            dim=0,
        )
        final_subtask_mask = torch.cat(
            [
                bos['attention_mask'].squeeze(0),
                subtask_out['attention_mask'].squeeze(0),
            ],
            dim=0,
        )

        if add_eos:
            eos = self.paligemma_tokenizer('<eos>', add_special_tokens=False, return_tensors='pt')
            final_subtask_ids = torch.cat([final_subtask_ids, eos['input_ids'].squeeze(0)], dim=0)
            final_subtask_mask = torch.cat([final_subtask_mask, eos['attention_mask'].squeeze(0)], dim=0)

        return {'input_ids': final_subtask_ids, 'attention_mask': final_subtask_mask}

    def create_input_tokens(
        self, task: str, state: torch.Tensor | None = None, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Creates the final token sequence from state, task, and optional
        action.

        This method combines different modalities into a single token sequence based
        on the configured encoding scheme (e.g., including discretized state,
        encoded actions, or predicting sub-tasks).

        Args:
            task: The task description string.
            state: The optional state tensor.
            action: The optional action tensor.

        Returns:
            A tuple containing:
            - final_ids: The final token IDs.
            - padded_mask: The attention mask for the padded sequence.
            - att_mask: Attention mask for language modeling loss.
            - loss_mask: Mask to indicate which tokens to use for loss calculation.
            - fast_action_indicator: A mask indicating tokens that correspond to actions.
            - predict_subtask: A boolean indicating if the sub-task is being predicted.
        """
        prefix_texts = []
        cleaned = task.lower().strip().replace('_', ' ')

        main_task = cleaned.split(' subtask: ')[0]
        sub_task = None
        if ' subtask: ' in cleaned:
            sub_task = cleaned.split(' subtask: ')[1].split('\n')[0]

        # Randomly sample the input type
        encode_sub_task_input = self.encode_sub_task_input
        is_sub_task_train = self.is_train
        encode_action_input = self.encode_action_input
        if self.sample_generator is not None:
            encode_sub_task_input, is_sub_task_train, encode_action_input = self.sample_generator.get_sample()
        encode_sub_task_input = encode_sub_task_input and sub_task is not None

        # Create the prefix text
        # The version of pretrained model to init the model is `pt` instead of `mix`, so the prefix text should end with a `\n`
        # Situation 1: Only main task
        predict_subtask = encode_sub_task_input and is_sub_task_train
        if predict_subtask or not self.discrete_state_input:
            prefix_texts.append(f'Task: {main_task}\n')

        # Situation 2: Main task and state
        # Situation 3: Main task, subtask and state
        elif self.discrete_state_input:
            assert state is not None, 'state is required when discrete_state_input is True'
            bins = torch.linspace(-1, 1, 256 + 1, device=self.device)[:-1]
            discretized = torch.bucketize(state, bins) - 1
            state_str = ' '.join(str(val.item()) for val in discretized)
            if encode_sub_task_input and not is_sub_task_train:
                prefix_texts.append(f'Task: {main_task}, Subtask: {sub_task}, State: {state_str};\n')
            else:
                prefix_texts.append(f'Task: {main_task}, State: {state_str};\n')

        else:
            raise ValueError('Invalid prefix text mode')

        prefix_out = self.paligemma_tokenizer(
            prefix_texts,
            add_special_tokens=True,
            return_tensors='pt',
            padding='longest',
            truncation=False,
        )
        prefix_ids = prefix_out['input_ids'][0]
        prefix_mask = prefix_out['attention_mask'][0]
        prefix_length = len(prefix_ids)
        fast_action_indicator = torch.zeros(prefix_length, dtype=torch.int32)

        assert prefix_length < self.max_length, f'Prefix length {prefix_length} is greater than max length {self.max_length}'

        final_ids = prefix_ids
        final_mask = prefix_mask
        # Create the suffix text
        # Situation 1: Predict subtask
        if predict_subtask:
            encoded_sub_task = self.encode_sub_task(sub_task, add_eos=True)
            sub_task_ids = encoded_sub_task['input_ids']
            sub_task_mask = encoded_sub_task['attention_mask']

            final_ids = torch.cat([final_ids, sub_task_ids], dim=0)
            final_mask = torch.cat([final_mask, sub_task_mask], dim=0)
            fast_action_indicator = torch.cat([fast_action_indicator, torch.zeros_like(sub_task_mask)], dim=0)

        # Situation 2: Predict discrete action
        if encode_action_input and action is not None:
            encoded_action = self.encode_action(action[None])
            act_ids = encoded_action['input_ids']
            act_mask = encoded_action['attention_mask']

            final_ids = torch.cat([final_ids, act_ids], dim=0)
            final_mask = torch.cat([final_mask, act_mask], dim=0)
            fast_action_indicator = torch.cat([fast_action_indicator, torch.ones_like(act_mask)], dim=0)

        if final_ids.shape[0] > self.max_length and not self.autoregressive_inference_mode:
            final_ids = final_ids[: self.max_length]
            final_mask = final_mask[: self.max_length]
            fast_action_indicator = fast_action_indicator[: self.max_length]

        batch_inputs = {
            'input_ids': final_ids.tolist(),
            'attention_mask': final_mask.tolist(),
        }
        # Padding and set loss mask
        padding_side = 'left' if self.autoregressive_inference_mode else 'right'
        padded_output = self.paligemma_tokenizer.pad(
            batch_inputs, padding='max_length', padding_side=padding_side, max_length=self.max_length, return_tensors='pt'
        )
        final_ids = padded_output['input_ids']
        padded_mask = padded_output['attention_mask']

        att_mask = (padded_mask != 0).cumsum(dim=0) > prefix_length
        att_mask = att_mask & padded_mask

        loss_mask = (padded_mask != 0).cumsum(dim=0) > prefix_length
        loss_mask = loss_mask & padded_mask

        fast_action_indicator = F.pad(fast_action_indicator, (0, self.max_length - fast_action_indicator.shape[0]), mode='constant', value=0)

        return (
            final_ids.to(dtype=torch.int32, device=self.device),
            padded_mask.to(dtype=torch.bool, device=self.device),
            att_mask.to(dtype=torch.bool, device=self.device),
            loss_mask.to(dtype=torch.bool, device=self.device),
            fast_action_indicator.to(dtype=torch.bool, device=self.device),
            predict_subtask,
        )

    def __call__(self, data: dict) -> dict:
        """Applies the full prompt tokenization pipeline to a data dictionary.

        Args:
            data: A dictionary containing 'task', optionally 'observation.state' and 'action'.

        Returns:
            The output of `create_input_tokens`.
        """
        if 'task' not in data:
            raise ValueError('No task found in data')

        task = data['task']
        state = data.get('observation.state', None)
        action = data.get('action', None)

        if action is not None and not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        return self.create_input_tokens(task, state, action)

    def extract_actions(self, tokens: list[list[int]], action_horizon: int, action_dim: int) -> torch.Tensor:
        """Extract continuous actions from predicted FAST action tokens.

        Args:
            tokens: The predicted FAST action tokens (PaliGemma token IDs).
            action_horizon: The time horizon (number of time steps) for the continuous action.
                This should match the horizon used during encoding.
            action_dim: The dimension of the continuous action vector.

        Returns:
            The extracted continuous actions as a tensor of shape (1, action_horizon, action_dim).
        """
        assert len(tokens) == 1, 'Only support batch size 1'
        sequence = tokens[0].tolist() if hasattr(tokens[0], 'tolist') else list(tokens[0])

        bos_tokens = self.paligemma_tokenizer('Action: ', add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0).tolist()
        eos_tokens = self.paligemma_tokenizer('|<eos>', add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0).tolist()

        def find_subsequence(sequence_ids: list[int], pattern: list[int], start: int = 0) -> int:
            if not pattern:
                return -1
            max_start = len(sequence_ids) - len(pattern)
            for idx in range(start, max_start + 1):
                if sequence_ids[idx : idx + len(pattern)] == pattern:
                    return idx
            return -1

        bos_idx = find_subsequence(sequence, bos_tokens)
        if bos_idx == -1:
            return torch.zeros((1, 0, action_dim), dtype=torch.float32)

        action_start = bos_idx + len(bos_tokens)
        eos_idx = find_subsequence(sequence, eos_tokens, start=action_start)
        if eos_idx == -1:
            eos_idx = len(sequence)

        paligemma_action_ids = sequence[action_start:eos_idx]

        if not paligemma_action_ids:
            return torch.zeros((1, 0, action_dim), dtype=torch.float32)

        vocab_size = self.paligemma_tokenizer.vocab_size
        if self.text_token_length is not None:
            vocab_size = min(vocab_size, self.text_token_length)
        base_token_id = vocab_size - 1 - self.fast_skip_tokens

        fast_tokens: list[int] = []
        for paligemma_id in paligemma_action_ids:
            if paligemma_id > base_token_id:
                continue
            fast_id = base_token_id - paligemma_id
            if fast_id < 0:
                continue
            fast_tokens.append(int(fast_id))

        if not fast_tokens:
            return torch.zeros((1, 0, action_dim), dtype=torch.float32)

        decoded_actions = self.fast_tokenizer.decode(
            [fast_tokens],
            time_horizon=action_horizon,
            action_dim=action_dim,
        )
        actions = torch.tensor(decoded_actions, dtype=torch.float32)
        return actions


class SampleGenerator:
    """Generates random prompt format samples based on given ratios."""

    def __init__(self, sample_ratios: dict[str, float]):
        """Initializes the sample generator.

        Args:
            sample_ratios: A dictionary mapping prompt format names to their
                sampling ratios. The sum of ratios should be 1.0.
        """
        valid_sample_names = [
            'task_only',
            'task_with_subtask',
            'task_only_using_subtask_regression',
            'task_only_using_fast_regression',
            'task_with_subtask_using_fast_regression',
        ]
        assert all(
            sample_name in valid_sample_names for sample_name in sample_ratios.keys()
        ), f'sample_name should be one of {valid_sample_names}, got {sample_ratios.keys()}'
        assert all(
            sample_ratio >= 0 for sample_ratio in sample_ratios.values()
        ), f'sample_ratio should be greater than or equal to 0, got {sample_ratios.values()}'
        assert all(
            sample_ratio <= 1 for sample_ratio in sample_ratios.values()
        ), f'sample_ratio should be less than or equal to 1, got {sample_ratios.values()}'
        # sum of sample_ratios should be 1
        if 'identity' not in sample_ratios:
            sample_ratios['identity'] = 1.0 - sum(sample_ratios.values())
        assert math.isclose(sum(sample_ratios.values()), 1.0, abs_tol=1e-6), f'sum of sample_ratios should be 1, got {sum(sample_ratios.values())}'
        self.sample_ratios = sample_ratios

    def get_sample(self) -> tuple[bool, bool, bool]:
        """Randomly selects a prompt format based on the configured ratios.

        Returns:
            A tuple of booleans: (encode_sub_task_input, is_sub_task_train, encode_action_input).
        """
        sample_type = random.random()
        sample_name = None
        prob_acc = 0.0
        for name, sample_ratio in self.sample_ratios.items():
            prob_acc += sample_ratio
            if sample_type < prob_acc:
                sample_name = name
                break

        encode_sub_task_input = sample_name in ['task_with_subtask', 'task_with_subtask_using_fast_regression', 'task_only_using_subtask_regression']
        is_sub_task_train = sample_name == 'task_only_using_subtask_regression'
        encode_action_input = sample_name in ['task_only_using_fast_regression', 'task_with_subtask_using_fast_regression']
        return encode_sub_task_input, is_sub_task_train, encode_action_input
