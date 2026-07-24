# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
ImageTransform - Generic image preprocessing transform for VLA models.

Supports configurable resize strategies and normalization modes:
  - Resize: resize_with_pad | resize_only | center_crop | none
  - Normalize: siglip ([-1,1]) | imagenet (mean/std) | identity ([0,1])

Default behavior (resize_with_pad + siglip) matches PI0.5 / SigLIP pipeline.
"""

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

from loongforge.embodied.data.datasets.transforms.base import BaseTransform


# ═══════════════════════════════════════════════════════════════
# Image resize utilities (formerly image_utils.py)
# ═══════════════════════════════════════════════════════════════

def resize_with_pad(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Aspect-ratio preserving resize with zero-padding."""
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)
    return padded_images


def resize_only(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Simple resize without padding."""
    squeeze = images.dim() == 3
    if squeeze:
        images = images.unsqueeze(0)
    resized = F.interpolate(
        images,
        size=(height, width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    if squeeze:
        resized = resized.squeeze(0)
    return resized


def center_crop_resize(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Center crop to match target aspect ratio, then resize."""
    squeeze = images.dim() == 3
    if squeeze:
        images = images.unsqueeze(0)

    _, _, cur_h, cur_w = images.shape
    target_ratio = width / height

    if cur_w / cur_h > target_ratio:
        crop_h = cur_h
        crop_w = int(cur_h * target_ratio)
    else:
        crop_w = cur_w
        crop_h = int(cur_w / target_ratio)

    start_h = (cur_h - crop_h) // 2
    start_w = (cur_w - crop_w) // 2
    cropped = images[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

    resized = F.interpolate(
        cropped,
        size=(height, width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    if squeeze:
        resized = resized.squeeze(0)
    return resized


class ImageTransform(BaseTransform):
    """Generic image preprocessing transform for VLA models.

    Converts images (PIL / tensor / ndarray) to torch.Tensor [C, H, W] with
    configurable resize strategy and normalization mode.

    Resize strategies:
        "resize_with_pad" — Aspect-ratio preserving resize + zero-padding (default)
        "resize_only" — Simple resize to target size (distorts aspect ratio)
        "center_crop" — Center crop to target aspect ratio, then resize
        "none" — Skip resizing while still applying dtype/layout conversion and normalization

    Normalization modes:
        "siglip" — Maps [0,1] to [-1,1] via t*2-1 (default, for SigLIP/PaliGemma)
        "imagenet" — Standard ImageNet mean/std normalization
        "identity" — No normalization, output stays in [0,1]
    """

    RESIZE_STRATEGIES = ["resize_with_pad", "resize_only", "center_crop", "none"]
    NORMALIZE_MODES = ["siglip", "imagenet", "identity"]

    # ImageNet standard normalization parameters
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        apply_to: List[str],
        image_size: int = 224,
        resize_strategy: str = "resize_with_pad",
        normalize_mode: str = "identity",
        interpolation_mode: str = "bilinear",
        training: bool = True,
    ):
        """
        Args:
            apply_to: Keys in data dict to transform
            image_size: Target size (square: H=W=image_size)
            resize_strategy: One of RESIZE_STRATEGIES
            normalize_mode: One of NORMALIZE_MODES
            interpolation_mode: Interpolation for resize ("bilinear", "nearest", etc.)
            training: Whether in training mode
        """
        super().__init__(apply_to=apply_to, training=training)
        self.image_size = image_size
        self.resize_strategy = resize_strategy
        self.normalize_mode = normalize_mode
        self.interpolation_mode = interpolation_mode

        assert resize_strategy in self.RESIZE_STRATEGIES, (
            f"Invalid resize_strategy: {resize_strategy}. Valid: {self.RESIZE_STRATEGIES}"
        )
        assert normalize_mode in self.NORMALIZE_MODES, (
            f"Invalid normalize_mode: {normalize_mode}. Valid: {self.NORMALIZE_MODES}"
        )

        # Pre-compute ImageNet tensors for efficiency
        if normalize_mode == "imagenet":
            self._mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
            self._std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the image transform to every configured key in ``data``."""
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, list):
                data[key] = [self.process_single(img) for img in value]
            elif isinstance(value, torch.Tensor) and value.ndim == 4:
                # Multi-frame tensor [T, C, H, W] — process as batch
                data[key] = self.process_batch(list(value))
            else:
                data[key] = self.process_single(value)
        return data

    def process_single(self, img) -> torch.Tensor:
        """Process a single image.

        Steps:
            1. Convert to tensor [C, H, W] in [0, 1]
            2. Apply resize strategy
            3. Apply normalization
        """
        # Step 1: Convert to tensor [C, H, W] in [0, 1]
        t = self._to_tensor(img)

        # Step 2: Resize
        if (
            self.resize_strategy != "none"
            and (t.shape[-2] != self.image_size or t.shape[-1] != self.image_size)
        ):
            t = self._resize(t.unsqueeze(0)).squeeze(0)

        # Step 3: Normalize
        t = self._normalize(t)
        return t

    def process_batch(self, images: List, image_size: int = None) -> torch.Tensor:
        """Process a list of images to (B, C, H, W) tensor.

        More efficient batch version using a single resize call.
        """
        size = image_size or self.image_size

        tensors = []
        for img in images:
            tensors.append(self._to_tensor(img))

        batch = torch.stack(tensors)
        if self.resize_strategy != "none" and (batch.shape[-2] != size or batch.shape[-1] != size):
            batch = self._resize(batch, size)

        # Normalize
        batch = self._normalize(batch)
        return batch

    def _to_tensor(self, img) -> torch.Tensor:
        """Convert input (PIL/ndarray/tensor) to [C, H, W] float in [0, 1]."""
        if not isinstance(img, torch.Tensor):
            return to_tensor(img)  # PIL/ndarray → (C, H, W) float [0, 1]
        t = img.float()
        if t.max() > 1.0:
            t = t / 255.0
        return t

    def _resize(self, batch: torch.Tensor, size: int = None) -> torch.Tensor:
        """Apply configured resize strategy to a (B, C, H, W) tensor."""
        s = size or self.image_size
        if self.resize_strategy == "resize_with_pad":
            return resize_with_pad(batch, s, s, mode=self.interpolation_mode)
        elif self.resize_strategy == "resize_only":
            return resize_only(batch, s, s, mode=self.interpolation_mode)
        elif self.resize_strategy == "center_crop":
            return center_crop_resize(batch, s, s, mode=self.interpolation_mode)
        elif self.resize_strategy == "none":
            return batch
        raise ValueError(f"Unknown resize_strategy: {self.resize_strategy}")

    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        """Apply configured normalization to tensor in [0, 1]."""
        if self.normalize_mode == "siglip":
            return t * 2.0 - 1.0
        elif self.normalize_mode == "imagenet":
            mean = self._mean.to(t.device, t.dtype)
            std = self._std.to(t.device, t.dtype)
            return (t - mean) / std
        elif self.normalize_mode == "identity":
            return t
        raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")
