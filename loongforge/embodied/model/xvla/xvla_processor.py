# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from X-VLA (https://github.com/2toinf/X-VLA).
# Copyright 2025 2toINF. All rights reserved.

"""Standalone XVLA text/image processing cores for inference.

This module intentionally has **no dependency** on
``loongforge.embodied.data``, ``torchdata``, or the distributed init stack, so
it can be imported from an inference-only context (e.g. from
:meth:`XVLAPolicy.predict_action`) without triggering the training-side
DataLoader / DistributedContext import chain.

Two lightweight classes are provided:

* :class:`XVLATokenizerCore` — lazy-loads a BART-compatible tokenizer from a
  checkpoint directory and exposes ``encode_language_batch``.
* :class:`XVLAImageProcessorCore` — lazy-loads an ``AutoImageProcessor``,
  converts arbitrary per-view images (PIL / CHW tensor / HWC ndarray) to
  ``image_input`` / ``image_mask`` tensors.

The corresponding training-time transforms in
``loongforge.embodied.data.transforms.xvla.xvla_transform`` subclass these
cores to add the :class:`BaseTransform` ``apply()`` hook, so the two paths
share one implementation.
"""

from __future__ import annotations

import numpy as np
import torch

from PIL import Image
from transformers import AutoImageProcessor
from transformers import AutoTokenizer
from typing import Any, Dict, List
from torchvision.transforms.functional import to_pil_image


class XVLATokenizerCore:
    """Text tokenization for XVLA inference (no training-package deps).

    Loads a BART-compatible tokenizer from a checkpoint directory on first
    access and exposes a batch tokenization helper.
    """

    def __init__(
        self,
        tokenizer_path: str = "",
        language_max_length: int = 50,
    ):
        """
        Args:
            tokenizer_path: Path to the pretrained checkpoint directory that
                            contains ``tokenizer_config.json`` (etc.).
            language_max_length: Maximum token length for padding/truncation.
        """
        self.tokenizer_path = tokenizer_path
        self.language_max_length = language_max_length
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer on first access."""
        if self._tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

            # Ensure a pad token exists (BART/Florence2 may lack one by default)
            if tokenizer.pad_token is None:
                vocab = tokenizer.get_vocab()
                if "<pad>" in vocab:
                    tokenizer.add_special_tokens({"pad_token": "<pad>"})
                elif tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "<pad>"})

            self._tokenizer = tokenizer
        return self._tokenizer

    def encode_language_batch(
        self, instructions: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of instructions; returns ``{"input_ids": [B, L]}``."""
        inputs = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding="max_length",
            max_length=self.language_max_length,
            truncation=True,
        )
        return {"input_ids": inputs["input_ids"]}


class XVLAImageProcessorCore:
    """Image encoding for XVLA inference (no training-package deps).

    Loads an ``AutoImageProcessor`` on first access and converts per-sample
    per-view images into the ``image_input`` / ``image_mask`` tensors the
    model expects.
    """

    def __init__(
        self,
        tokenizer_path: str = "",
        num_views: int = 3,
    ):
        """
        Args:
            tokenizer_path: Path to the pretrained checkpoint directory that
                            contains ``preprocessor_config.json`` (etc.).
            num_views: Expected number of camera views; missing views are
                       zero-padded and marked invalid in ``image_mask``.
        """
        self.tokenizer_path = tokenizer_path
        self.num_views = num_views
        self._image_processor = None

    @property
    def image_processor(self):
        """Lazy-load the image processor on first access."""
        if self._image_processor is None:
            self._image_processor = AutoImageProcessor.from_pretrained(
                self.tokenizer_path
            )
        return self._image_processor

    @staticmethod
    def _to_pil(img):
        """Convert an image to a PIL Image.

        Accepts:
          * ``PIL.Image.Image`` — returned as-is.
          * ``torch.Tensor`` in CHW layout (float in [0, 1] or uint8/float in
            [0, 255]).
          * ``np.ndarray`` in HWC layout (H, W, C) with C in {1, 3, 4}, or in
            CHW layout as a fallback. Float in [0, 1] or uint8/float in
            [0, 255].
        """

        if isinstance(img, Image.Image):
            return img

        if isinstance(img, np.ndarray):
            # HWC ndarray: hand to torchvision directly (it expects HWC for
            # numpy inputs). Normalize float ranges > 1 to [0, 1] first.
            if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                arr = img
                if np.issubdtype(arr.dtype, np.floating):
                    if arr.max() > 1.0:
                        arr = arr / 255.0
                    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
                return to_pil_image(arr)
            # Fallback: treat as CHW and go through the tensor path.
            img = torch.as_tensor(img)

        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)
        img = img.float()
        if img.max() > 1.0:
            img = img / 255.0
        return to_pil_image(img.clamp(0.0, 1.0))

    def _encode_single(self, pil_images: List) -> Dict[str, torch.Tensor]:
        """Encode a single sample's list of PIL images into image_input / image_mask."""
        processed = self.image_processor(pil_images, return_tensors="pt")["pixel_values"]
        V_exist = processed.size(0)

        if V_exist < self.num_views:
            processed = torch.cat(
                [processed,
                 processed.new_zeros(self.num_views - V_exist, *processed.shape[1:])],
                dim=0,
            )

        image_mask = torch.zeros(self.num_views, dtype=torch.bool, device=processed.device)
        image_mask[:V_exist] = True

        return {"image_input": processed, "image_mask": image_mask}

    def encode_image_batch(
        self, batch_pil_images: List[List]
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of per-sample PIL image lists; returns batched tensors.

        Returns:
            {
              "image_input": [B, num_views, C, H, W],
              "image_mask":  [B, num_views],
            }
        """
        batch_imgs, batch_masks = [], []
        for pil_images in batch_pil_images:
            encoded = self._encode_single(pil_images)
            batch_imgs.append(encoded["image_input"])
            batch_masks.append(encoded["image_mask"])
        return {
            "image_input": torch.stack(batch_imgs, dim=0),
            "image_mask": torch.stack(batch_masks, dim=0),
        }
