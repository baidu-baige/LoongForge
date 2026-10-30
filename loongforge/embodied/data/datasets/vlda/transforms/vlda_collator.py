# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLDA observation batch and registry-based collation."""

from dataclasses import dataclass

import torch
from torch.utils.data import default_collate

from loongforge.embodied.data.datasets.transforms.collator import BasePreprocessor, PreparedBatch, register_preprocessor


@dataclass
class VLDAPreparedBatch(PreparedBatch):
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    actions: torch.Tensor | None = None
    vjepa_image: dict[str, torch.Tensor] | None = None
    vjepa_future_image: dict[str, torch.Tensor] | None = None
    neg_actions: torch.Tensor | None = None
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None


@register_preprocessor("vlda")
class VLDAPreprocessor(BasePreprocessor):
    @classmethod
    def from_config(cls, model_cfg, data_cfg, training_args=None, dataset_stats=None, dataset=None):
        return cls()

    def __call__(self, examples):
        return VLDAPreparedBatch(**default_collate(examples))
