# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain-0 batch-level collator.

``GigaBrainTransform`` (per-sample) already produces the reference
``GigaBrain0Transform`` output dict:

    {
        "lang_tokens":            tensor [L],
        "lang_masks":              tensor [L] (bool),
        "lang_att_masks":          tensor [L],
        "lang_loss_masks":         tensor [L],
        "fast_action_indicator":   tensor [L] (bool) or None,
        "observation.state":       tensor [max_action_dim],
        "action":                  tensor [T, max_action_dim],
        "images":                  list[tensor [C, H, W]] (len == num views),
        "image_masks":             list[tensor[]] (bool scalars, len == num views),
        "action_loss_mask":        tensor [T] (bool),
        "embodiment_id":           tensor[] (long scalar),
        # optional (only if traj_cfg is set): "traj", "traj_loss_mask"
    }

This collator stacks that per-sample dict into the batch layout
``GigaBrainPolicy.forward`` / ``GigaBrain0Policy.forward`` expects — a list of
per-view stacked image tensors (``images[v]`` -> [B, C, H, W]) rather than a
single [B, V, C, H, W] tensor, matching the reference
``GigaBrain0Trainer.forward_step`` batch_dict layout exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    register_preprocessor,
)


class GigaBrainBatch(dict):
    """Dictionary batch with a tensor-recursive ``to(device)`` helper."""

    def to(self, device: torch.device) -> "GigaBrainBatch":
        """Move all tensor values (including list-of-tensor fields) to ``device``."""
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, list):
                return [move(v) for v in value]
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            return value

        for key, value in list(self.items()):
            self[key] = move(value)
        return self


@register_preprocessor("giga_brain")
class GigaBrainPreprocessor(BasePreprocessor):
    """DataLoader collate_fn for GigaBrain-0.

    Stateless: all per-sample work (normalization, delta actions, tokenizing,
    image encoding) already happened in ``GigaBrainTransform``. This collator
    only stacks per-sample tensors into batch tensors.
    """

    @classmethod
    def from_config(
        cls, model_cfg, data_cfg=None, training_args=None, dataset_stats=None, dataset=None,
    ) -> "GigaBrainPreprocessor":
        return cls()

    def __call__(self, examples: List[Dict[str, Any]]) -> GigaBrainBatch:
        batch = GigaBrainBatch()

        batch["lang_tokens"] = torch.stack([ex["lang_tokens"] for ex in examples])
        batch["lang_masks"] = torch.stack([ex["lang_masks"] for ex in examples])
        batch["lang_att_masks"] = torch.stack([ex["lang_att_masks"] for ex in examples])
        batch["lang_loss_masks"] = torch.stack([ex["lang_loss_masks"] for ex in examples])

        if examples[0].get("fast_action_indicator") is not None:
            batch["fast_action_indicator"] = torch.stack(
                [ex["fast_action_indicator"] for ex in examples]
            )
        else:
            batch["fast_action_indicator"] = None

        batch["observation.state"] = torch.stack([ex["observation.state"] for ex in examples])
        batch["action"] = torch.stack([ex["action"] for ex in examples])
        batch["action_loss_mask"] = torch.stack([ex["action_loss_mask"] for ex in examples])
        batch["embodiment_id"] = torch.stack([ex["embodiment_id"] for ex in examples])

        # images/image_masks are lists of per-view tensors (reference layout:
        # embed_prefix stacks over the view axis via torch.stack(images, dim=0)).
        num_views = len(examples[0]["images"])
        batch["images"] = [
            torch.stack([ex["images"][v] for ex in examples]) for v in range(num_views)
        ]
        batch["image_masks"] = [
            torch.stack([ex["image_masks"][v] for ex in examples]) for v in range(num_views)
        ]

        if "traj" in examples[0]:
            batch["traj"] = torch.stack([ex["traj"] for ex in examples])
            batch["traj_loss_mask"] = torch.stack([ex["traj_loss_mask"] for ex in examples])

        return batch
