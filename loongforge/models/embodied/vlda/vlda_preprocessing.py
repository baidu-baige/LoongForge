# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted from VLDA/openpi.

"""Separate SigLIP augmentation and V-JEPA target preprocessing."""

import logging
from collections.abc import Sequence

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
IMAGE_RESOLUTION = (224, 224)
IMAGE_RESOLUTION_VJEPA = (256, 256)


def preprocess_observation(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution_siglip: tuple[int, int] = IMAGE_RESOLUTION,
    image_resolution_vjepa: tuple[int, int] = IMAGE_RESOLUTION_VJEPA,
):
    """Preprocess both SigLIP and V-JEPA2 images from a VLDAObservation.

    Returns:
        siglip_output: SimpleProcessedObservation (images, masks, state, prompt, ...)
        vjepa_images: dict of processed V-JEPA2 online encoder inputs (CHW, 256, ImageNet norm)
        vjepa_future_images: dict of processed V-JEPA2 teacher encoder inputs
    """
    siglip_output = preprocess_siglip_observation(
        observation, train=train, image_keys=image_keys, image_resolution=image_resolution_siglip
    )
    if observation.vjepa_image and observation.vjepa_future_image:
        (vjepa_images, vjepa_future_images) = preprocess_observation_vjepa(
            observation, image_resolution=image_resolution_vjepa
        )
    else:
        (vjepa_images, vjepa_future_images) = (None, None)
    return (siglip_output, vjepa_images, vjepa_future_images)


_VJEPA_MEAN = (0.485, 0.456, 0.406)
_VJEPA_STD = (0.229, 0.224, 0.225)


def preprocess_observation_vjepa(observation, *, image_resolution: tuple[int, int] = IMAGE_RESOLUTION_VJEPA):
    """Resize and center-crop each image, repeat it twice, then apply ImageNet normalization."""
    mean = torch.tensor(_VJEPA_MEAN).view(1, 3, 1, 1, 1)
    std = torch.tensor(_VJEPA_STD).view(1, 3, 1, 1, 1)

    def _process(img: torch.Tensor) -> torch.Tensor:
        """Convert a single V-JEPA2 image to the expected format."""
        img = (img + 1.0) / 2.0
        if img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)
        if img.shape[-2:] != image_resolution:
            img = TF.resize(img, size=min(image_resolution), antialias=True)
            img = TF.center_crop(img, output_size=image_resolution)
        img = img.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        m = mean.to(device=img.device, dtype=img.dtype)
        s = std.to(device=img.device, dtype=img.dtype)
        img = (img - m) / s
        return img

    vjepa_images = {key: _process(img) for (key, img) in observation.vjepa_image.items()}
    vjepa_future_images = {key: _process(img) for (key, img) in observation.vjepa_future_image.items()}
    return (vjepa_images, vjepa_future_images)


def preprocess_siglip_observation(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Preserve VLDA batch-shared crop, rotation, and color augmentation."""
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")
    batch_shape = observation.state.shape[:-1]
    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        is_channels_first = image.shape[1] == 3
        if is_channels_first:
            image = image.permute(0, 2, 3, 1)
        image = image.contiguous()
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = resize_with_pad_torch(image, *image_resolution)
        if train:
            image = image / 2.0 + 0.5
            if "wrist" not in key:
                (height, width) = image.shape[1:3]
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2), size=(height, width), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
                angle = torch.rand(1, device=image.device) * 10 - 5
                if torch.abs(angle) > 0.1:
                    angle_rad = angle * torch.pi / 180.0
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)
                    (grid_y, grid_x) = torch.meshgrid(grid_y, grid_x, indexing="ij")
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)
                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="zeros", align_corners=False
                    ).permute(0, 2, 3, 1)
            brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6
            image = image * brightness_factor
            contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean
            saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor
            image = torch.clamp(image, 0, 1)
            image = image * 2.0 - 1.0
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)
        out_images[key] = image
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


def resize_with_pad_torch(images, height, width):
    """Resize float images in [-1, 1] with black padding."""
    image = images.permute(0, 3, 1, 2)
    ratio = max(image.shape[-2] / height, image.shape[-1] / width)
    h, w = int(image.shape[-2] / ratio), int(image.shape[-1] / ratio)
    image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False).clamp(-1, 1)
    top, left = (height - h) // 2, (width - w) // 2
    return F.pad(image, (left, width - w - left, top, height - h - top), value=-1).permute(0, 2, 3, 1)
