# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted from MIT-licensed VPP2.
"""Image conversion, resizing, cropping and normalization for VPP2 datasets."""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as TF
import torchvision.transforms.functional as TF_F
from PIL import Image


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        x = x.to(torch.float32) / 255.0
        return x


class Pad(nn.Module):
    def __init__(self, padding, fill=0, padding_mode="constant"):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Can only pad tensor of 4 dims."
        return self.pad(x)


class CenterCropAspectResize(nn.Module):
    """Center-crop a video tensor to an aspect ratio, then resize it.

    The LeRobot loader supplies ``[T,C,H,W]`` tensors.  RoboColiseum's head
    and wrist cameras have different native geometries, so a direct resize
    would make training and benchmark inference disagree geometrically.  This
    transform establishes a common 4:3 camera contract before the views are
    composed into the model's T-shaped canvas.
    """

    def __init__(self, size, aspect_ratio: float = 4.0 / 3.0):
        super().__init__()
        if len(size) != 2 or int(size[0]) <= 0 or int(size[1]) <= 0:
            raise ValueError(f"size must be [height,width], got {size!r}")
        if float(aspect_ratio) <= 0:
            raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio!r}")
        self.size = [int(size[0]), int(size[1])]
        self.aspect_ratio = float(aspect_ratio)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"CenterCropAspectResize expects [T,C,H,W], got {tuple(x.shape)}")
        height, width = (int(value) for value in x.shape[-2:])
        current_ratio = width / height
        if current_ratio > self.aspect_ratio:
            crop_height = height
            crop_width = max(1, int(round(height * self.aspect_ratio)))
        else:
            crop_width = width
            crop_height = max(1, int(round(width / self.aspect_ratio)))
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        x = TF_F.crop(x, top, left, crop_height, crop_width)
        return TF_F.resize(
            x,
            size=self.size,
            interpolation=TF_F.InterpolationMode.BILINEAR,
            antialias=True,
        )


def obtain_image_size(data: torch.Tensor | Image.Image) -> tuple[int, int]:
    r"""Return spatial size from a PIL image or image/video tensor.

    Args:
        data (torch.Tensor | Image.Image): Input image or video tensor.
    Returns:
        width (int): Input width.
        height (int): Input height.
    """

    if isinstance(data, Image.Image):
        width, height = data.size
    elif isinstance(data, torch.Tensor):
        height, width = data.size()[-2:]
    else:
        raise ValueError("data to random crop should be PIL Image or tensor")

    return width, height


class ResizeSmallestSideAspectPreserving:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor | Image.Image:
        r"""Resize while preserving aspect ratio.

        The output is scaled so both spatial dimensions are at least the
        requested target size.

        Args:
            video (torch.Tensor | Image.Image): Input image or video tensor.
        Returns:
            torch.Tensor | Image.Image: Resized image or video tensor.
        """

        assert self.args is not None, "Please specify args in augmentations"

        img_w, img_h = self.args["img_w"], self.args["img_h"]

        orig_w, orig_h = obtain_image_size(video)
        scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert target_size[0] >= img_h and target_size[1] >= img_w, (
            f"Resize error. orig {(orig_w, orig_h)} desire {(img_w, img_h)} compute {target_size}"
        )

        return TF_F.resize(
            video,
            size=target_size,  # type: ignore
            interpolation=self.args.get("interpolation", TF_F.InterpolationMode.BICUBIC),
            antialias=True,
        )


class CenterCrop:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor | Image.Image:
        r"""Center crop to the requested spatial size.

        Args:
            video (torch.Tensor | Image.Image): Input image or video tensor.
        Returns:
            torch.Tensor | Image.Image: Center cropped image or video tensor.
        """
        assert (self.args is not None) and ("img_w" in self.args) and ("img_h" in self.args), (
            "Please specify size in args"
        )

        img_w, img_h = self.args["img_w"], self.args["img_h"]
        return TF_F.center_crop(video, [img_h, img_w])


class Normalize:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor:
        r"""Convert to tensor if needed and normalize by mean/std.

        Args:
            video (torch.Tensor | Image.Image): Input image or video tensor.
        Returns:
            torch.Tensor: Normalized image or video tensor.
        """
        assert self.args is not None, "Please specify args"

        mean = self.args["mean"]
        std = self.args["std"]

        if isinstance(video, torch.Tensor):
            data = video.to(dtype=torch.float32)
            if video.dtype == torch.uint8:
                data = data / 255.0
            data = data.to(dtype=torch.get_default_dtype())
        else:
            data = TF_F.to_tensor(video)  # division by 255 is applied in to_tensor()

        return TF_F.normalize(tensor=data, mean=mean, std=std)
