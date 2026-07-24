# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pi05 batch-level collator and tokenization utilities.

Batch-level collator (DataLoader collate_fn):
    @register_preprocessor("PaliGemmaPi05")
    Pi05Preprocessor: transformed samples → Pi05PreparedBatch (CPU tensors)

Utilities:
    tokenize_prompts(prompts, tokenizer, max_length) -> dict
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from loongforge.embodied.data.datasets.transforms.collator import BasePreprocessor, PreparedBatch, register_preprocessor

def tokenize_prompts(
    prompts: list,
    tokenizer,
    max_length: int = 200,
    padding: str = "max_length",
    padding_side: str = "right",
) -> dict:
    """Tokenize a list of prompts with the PaliGemma tokenizer."""
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            max_length=max_length,
            padding=padding,
            truncation=True,
        )
    finally:
        tokenizer.padding_side = original_padding_side
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


@dataclass
class Pi05PreparedBatch(PreparedBatch):
    """Preprocessed batch for Pi05 model.

    All tensors on CPU after collation; call .to(device) before forward().
    """
    images_list: List[torch.Tensor] = None   # List of (B, 3, H, W) per view
    img_masks: List[torch.Tensor] = None     # List of (B,) bool per view
    input_ids: torch.Tensor = None           # (B, seq_len)
    attention_mask: torch.Tensor = None      # (B, seq_len) bool
    actions: torch.Tensor = None             # (B, T, D)


@register_preprocessor("pi05")
class Pi05Preprocessor(BasePreprocessor):
    """DataLoader collate_fn for PaliGemmaPi05.

    Only handles batch-level collation. Per-sample transforms (image, action,
    state discretization) are applied via the injected `transform` pipeline.
    """

    def __init__(
        self,
        image_size: int = 224,
        num_images: int = 2,
        image_mask: Optional[List[bool]] = None,
        max_token_len: int = 200,
        tokenizer_path: str = "",
    ):
        self.image_size = image_size
        self.num_images = num_images
        self.image_mask = image_mask or [True] * num_images
        self.max_token_len = max_token_len
        self.tokenizer_path = tokenizer_path

    @classmethod
    def from_config(
        cls, model_cfg, data_cfg, training_args=None, dataset_stats=None, dataset=None,
    ) -> "Pi05Preprocessor":
        """Construct from typed ModelConfig + DataConfig (+ TrainingArgs).

        Tokenizer path: training_args.tokenizer_path (--tokenizer-path) > TOKENIZER_PATH env.
        Image/token processing fields come from DataConfig.
        """
        tokenizer_path = (
            (training_args.tokenizer_path if training_args is not None else None)
            or os.environ.get("TOKENIZER_PATH", "")
        )

        return cls(
            image_size=data_cfg.image_size,
            num_images=data_cfg.num_images,
            image_mask=data_cfg.image_mask,
            max_token_len=data_cfg.max_token_len,
            tokenizer_path=tokenizer_path,
        )

    @property
    def tokenizer(self):
        """Lazy-loaded tokenizer."""
        if not hasattr(self, "_tokenizer") or self._tokenizer is None:
            path = self.tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
            if not path:
                raise ValueError(
                    "Tokenizer path not set. Pass tokenizer_path or set TOKENIZER_PATH env."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(path)
        return self._tokenizer

    def __call__(self, examples: List[Dict[str, Any]]) -> Pi05PreparedBatch:
        """Collate pre-transformed samples into Pi05PreparedBatch."""
        # Stack per-sample images_list and img_masks into batch tensors
        images_list = [
            torch.stack([ex["images_list"][v] for ex in examples])
            for v in range(self.num_images)
        ]
        img_masks = [
            torch.tensor([ex["img_masks"][v] for ex in examples], dtype=torch.bool)
            for v in range(self.num_images)
        ]

        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        attention_mask = torch.stack([ex["attention_mask"] for ex in examples])

        actions = torch.stack([
            ex["action"] if isinstance(ex["action"], torch.Tensor)
            else torch.as_tensor(ex["action"], dtype=torch.float32)
            for ex in examples
        ])

        return Pi05PreparedBatch(
            images_list=images_list,
            img_masks=img_masks,
            input_ids=input_ids,
            attention_mask=attention_mask,
            actions=actions,
        )
