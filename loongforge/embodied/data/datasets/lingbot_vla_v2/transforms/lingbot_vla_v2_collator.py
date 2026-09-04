# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Batch-level preprocessor (collate_fn) for lingbot_vla_v2.

Wraps the vendored benchmark ``VLADataCollatorWithPacking`` so batches are
bit-identical to upstream, and repackages the resulting dict into a
``PreparedBatch`` per the LoongForge data contract (CPU tensors; the Trainer
moves them to the device).

``pil_images`` / ``future_pil_images`` stay as tensors inside ``data`` — the
teachers consume them on GPU inside the trainer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    PreparedBatch,
    register_preprocessor,
)


@dataclass
class LingbotVLAV2PreparedBatch(PreparedBatch):
    """Model-ready batch: a plain dict of tensors keyed like the benchmark."""

    data: Dict[str, Any] = field(default_factory=dict)


@register_preprocessor("lingbot_vla_v2")
class LingbotVLAV2Preprocessor(BasePreprocessor):
    """collate_fn delegating to the vendored VLADataCollatorWithPacking."""

    def __init__(self):
        from loongforge.embodied.model.lingbot_vla_v2.vendor.data import (
            VLADataCollatorWithPacking,
        )

        self._collator = VLADataCollatorWithPacking()

    @classmethod
    def from_config(
        cls, model_cfg, data_cfg, training_args=None, dataset_stats=None, dataset=None
    ) -> "LingbotVLAV2Preprocessor":
        """Build the preprocessor; no config knob affects the vendored collator."""
        return cls()

    def __call__(self, examples: List[Dict[str, Any]]) -> LingbotVLAV2PreparedBatch:
        batch = self._collator(examples)
        return LingbotVLAV2PreparedBatch(data=batch)
