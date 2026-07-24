# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA collator adapter for embodied training."""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

import torch.nn.functional as F

from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    PreparedBatch,
    register_preprocessor,
)


@dataclass
class LingBotVAPreparedBatch(PreparedBatch):
    """Model-ready LingBot-VA batch."""

    latents: torch.Tensor
    actions: torch.Tensor
    actions_mask: torch.Tensor
    text_emb: torch.Tensor
    frame_ids: torch.Tensor
    sample_meta: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Return the legacy dict shape expected by the existing LingBot logic."""
        return {
            "latents": self.latents,
            "actions": self.actions,
            "actions_mask": self.actions_mask,
            "text_emb": self.text_emb,
            "frame_ids": self.frame_ids,
            "_lingbot_sample_meta": self.sample_meta,
        }


@register_preprocessor("lingbot_va")
class LingBotVAPreprocessor(BasePreprocessor):
    """Batch-level LingBot-VA preprocessor backed by the existing collator."""

    def __init__(self, pad_to_multiple: int = 1):
        """Initialize the instance.

        Args:
            pad_to_multiple: Input value for this operation.
        """
        self.pad_to_multiple = int(pad_to_multiple)

    @classmethod
    def from_config(
        cls,
        model_cfg,
        data_cfg,
        training_args=None,
        dataset_stats=None,
        dataset=None,
    ) -> "LingBotVAPreprocessor":
        """Run from config.

        Args:
            model_cfg: Input value for this operation.
            data_cfg: Input value for this operation.
            training_args: Input value for this operation.
            dataset_stats: Input value for this operation.
            dataset: Input value for this operation.

        Returns:
            The computed result.
        """
        return cls(pad_to_multiple=int(data_cfg.pad_to_multiple))

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> LingBotVAPreparedBatch:
        """Apply the callable to its inputs.

        Args:
            examples: Input value for this operation.

        Returns:
            The computed result.
        """
        if not examples:
            raise ValueError("LingBotVAPreprocessor received an empty example list")
        return LingBotVAPreparedBatch(
            latents=_pad_stack(
                [item["latents"] for item in examples], self.pad_to_multiple
            ).contiguous(),
            actions=_pad_stack(
                [item["actions"] for item in examples], self.pad_to_multiple
            ).contiguous(),
            actions_mask=_pad_stack(
                [item["actions_mask"].to(torch.bool) for item in examples],
                self.pad_to_multiple,
            ).contiguous(),
            text_emb=_pad_stack_text(
                [item["text_emb"] for item in examples]
            ).contiguous(),
            frame_ids=_pad_stack_1d(
                [
                    torch.as_tensor(item.get("frame_ids", []), dtype=torch.long)
                    for item in examples
                ]
            ).contiguous(),
            sample_meta=[
                dict(item.get("_lingbot_sample_meta", {})) for item in examples
            ],
        )


def _pad_stack(tensors: List[torch.Tensor], pad_to_multiple: int) -> torch.Tensor:
    """Run pad stack.

    Args:
        tensors: Input value for this operation.
        pad_to_multiple: Input value for this operation.

    Returns:
        The computed result.
    """
    max_shape = [
        max(tensor.shape[dim] for tensor in tensors) for dim in range(tensors[0].ndim)
    ]
    if pad_to_multiple > 1:
        for dim in range(1, len(max_shape)):
            max_shape[dim] = _align(max_shape[dim], pad_to_multiple)
    padded = []
    for tensor in tensors:
        pad = []
        for dim in reversed(range(tensor.ndim)):
            pad.extend([0, max_shape[dim] - tensor.shape[dim]])
        padded.append(F.pad(tensor, pad))
    return torch.stack(padded, dim=0)


def _pad_stack_text(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Run pad stack text.

    Args:
        tensors: Input value for this operation.

    Returns:
        The computed result.
    """
    max_shape = [
        max(tensor.shape[dim] for tensor in tensors) for dim in range(tensors[0].ndim)
    ]
    padded = []
    for tensor in tensors:
        pad = []
        for dim in reversed(range(tensor.ndim)):
            pad.extend([0, max_shape[dim] - tensor.shape[dim]])
        padded.append(F.pad(tensor, pad))
    return torch.stack(padded, dim=0)


def _pad_stack_1d(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Run pad stack 1d.

    Args:
        tensors: Input value for this operation.

    Returns:
        The computed result.
    """
    max_len = max(tensor.numel() for tensor in tensors)
    if max_len == 0:
        return torch.empty((len(tensors), 0), dtype=torch.long)
    return torch.stack(
        [F.pad(tensor, (0, max_len - tensor.numel()), value=-1) for tensor in tensors],
        dim=0,
    )


def _align(value: int, multiple: int) -> int:
    """Run align.

    Args:
        value: Input value for this operation.
        multiple: Input value for this operation.

    Returns:
        The computed result.
    """
    return ((value + multiple - 1) // multiple) * multiple
