# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""XVLA batch-level collator.

The collator delegates image encoding and text tokenization to
:class:`XVLAEncodeImageTransform` and :class:`XVLATokenizeTransform`, which
replace the former ``XVLAProcessor``.  Per-sample dicts coming from the
dataset expose:

    {
        "observation.images.<cam>": tensor [3, H, W] (uint8/float in [0, 255] or [0, 1]),
        "observation.state":        tensor [proprio_dim],
        "action":                   tensor [T, action_dim]  (already normalized by ActionTransform),
        "task" / "prompt":          str,
        "domain_id":                optional int / tensor,
    }

and the collator produces the X-VLA-native batch keys consumed directly by the
model forward (see ``XVLAPolicy._prepare_inputs``):

    {
        "input_ids":   [B, L],
        "image_input": [B, num_views, C, H, W],
        "image_mask":  [B, num_views],
        "proprio":     [B, proprio_dim],
        "action":      [B, T, action_dim],
        "domain_id":   [B],
    }
"""

import os
from typing import Any, Dict, List

import torch

from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    register_preprocessor,
)
from loongforge.embodied.data.datasets.xvla.transforms.xvla_transform import (
    XVLAEncodeImageTransform,
    XVLATokenizeTransform,
)


class XVLABatch(dict):
    """Dictionary batch with a tensor-recursive ``to(device)`` helper."""

    def to(self, device: torch.device) -> "XVLABatch":
        """Move all tensor values in this batch to the specified device."""
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            if isinstance(value, list):
                return [move(v) for v in value]
            if isinstance(value, tuple):
                return tuple(move(v) for v in value)
            return value

        for key, value in list(self.items()):
            self[key] = move(value)
        return self


@register_preprocessor("xvla")
class XVLAPreprocessor(BasePreprocessor):
    """DataLoader collate_fn for XVLA.

    Uses :class:`XVLAEncodeImageTransform` and :class:`XVLATokenizeTransform`
    for image encoding and language tokenization respectively, replacing the
    former ``XVLAProcessor`` dependency.
    """

    def __init__(self, processor_path: str = "", num_views: int = 3):
        """
        Args:
            processor_path: Path to the pretrained checkpoint directory.
                            Can be overridden at runtime via PROCESSOR_PATH or
                            TOKENIZER_PATH environment variables.
            num_views: Number of camera views per sample expected by the model.
        """
        self.processor_path = processor_path
        self.num_views = num_views
        self._image_transform = None
        self._tokenize_transform = None

    @classmethod
    def from_config(
        cls, model_cfg, data_cfg=None, training_args=None, dataset_stats=None, dataset=None,
    ) -> "XVLAPreprocessor":
        """Construct an XVLAPreprocessor from typed ModelConfig + DataConfig."""
        num_views = (
            (data_cfg.num_image_views if data_cfg is not None else None)
            or model_cfg.num_image_views
            or 3
        )
        processor_path = (
            (training_args.tokenizer_path if training_args is not None else None)
            or os.environ.get("PROCESSOR_PATH", "")
            or os.environ.get("TOKENIZER_PATH", "")
        )
        return cls(processor_path=processor_path, num_views=num_views)

    @property
    def image_transform(self) -> XVLAEncodeImageTransform:
        """Lazy-initialize the image encoding transform."""
        if self._image_transform is None:
            self._image_transform = XVLAEncodeImageTransform(
                tokenizer_path=self.processor_path,
                num_views=self.num_views,
            )
        return self._image_transform

    @property
    def tokenize_transform(self) -> XVLATokenizeTransform:
        """Lazy-initialize the tokenization transform."""
        if self._tokenize_transform is None:
            self._tokenize_transform = XVLATokenizeTransform(
                tokenizer_path=self.processor_path,
            )
        return self._tokenize_transform

    def __call__(self, examples: List[Dict[str, Any]]) -> XVLABatch:
        """Collate a list of sample dicts into an X-VLA-native batch."""
        batch_texts: List[str] = [
            str(ex.get("prompt") or ex.get("task") or ex.get("lang") or "")
            for ex in examples
        ]

        batch = XVLABatch()

        if all("image_input" in ex for ex in examples):
            # Images were already preprocessed by the dataset (reference path).
            batch["input_ids"] = self.tokenize_transform.encode_language_batch(batch_texts)["input_ids"]
            batch["image_input"] = torch.stack([
                ex["image_input"] if isinstance(ex["image_input"], torch.Tensor)
                else torch.as_tensor(ex["image_input"], dtype=torch.float32)
                for ex in examples
            ])
            batch["image_mask"] = torch.stack([
                ex["image_mask"] if isinstance(ex["image_mask"], torch.Tensor)
                else torch.as_tensor(ex["image_mask"], dtype=torch.bool)
                for ex in examples
            ])
        else:
            # Raw per-view images: encode via XVLAEncodeImageTransform.
            batch_images: List[List[Any]] = []
            for ex in examples:
                image_keys = sorted(k for k in ex.keys() if k.startswith("observation.images."))
                batch_images.append([
                    XVLAEncodeImageTransform._to_pil(ex[k]) for k in image_keys
                ])
            encoded_imgs = self.image_transform.encode_image_batch(batch_images)
            encoded_lang = self.tokenize_transform.encode_language_batch(batch_texts)
            batch["input_ids"] = encoded_lang["input_ids"]
            batch["image_input"] = encoded_imgs["image_input"]
            batch["image_mask"] = encoded_imgs["image_mask"]

        # Proprio (state) — stack as-is; model pads/trims to action-space dim.
        if any("observation.state" in ex for ex in examples):
            proprio = [
                ex["observation.state"]
                if isinstance(ex.get("observation.state"), torch.Tensor)
                else torch.as_tensor(ex.get("observation.state"), dtype=torch.float32)
                for ex in examples
            ]
            batch["proprio"] = torch.stack(proprio)

        # Action (already normalized / chunked by ActionTransform).
        batch["action"] = torch.stack([
            ex["action"] if isinstance(ex["action"], torch.Tensor)
            else torch.as_tensor(ex["action"], dtype=torch.float32)
            for ex in examples
        ])

        # Domain id (default 0 when missing).
        domain_ids = []
        for ex in examples:
            d = ex.get("domain_id", 0)
            domain_ids.append(int(d.item()) if isinstance(d, torch.Tensor) else int(d))
        batch["domain_id"] = torch.tensor(domain_ids, dtype=torch.long)

        return batch
