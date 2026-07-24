# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.6 batch collator for the embodied trainer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    PreparedBatch,
    register_preprocessor,
)
from loongforge.embodied.data.datasets.groot_n1_6.transforms.processor_groot_n1_6 import (
    Gr00tN1d6DataCollator,
)


@dataclass
class GrootN1d6PreparedBatch(PreparedBatch):
    """Model-ready batch for GR00T-N1.6."""

    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor = None
    pixel_values: Any = None
    state: torch.Tensor = None
    actions: torch.Tensor = None
    action_mask: torch.Tensor = None
    embodiment_id: torch.Tensor = None
    action_is_pad: Optional[torch.Tensor] = None

    def to_model_inputs(self) -> Dict[str, Any]:
        """Convert to the GR00T-N1.6 model input dictionary."""
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "pixel_values": self.pixel_values,
            "state": self.state,
            "action": self.actions,
            "action_mask": self.action_mask,
            "embodiment_id": self.embodiment_id,
        }
        if self.action_is_pad is not None:
            inputs["action_is_pad"] = self.action_is_pad
        return {k: v for k, v in inputs.items() if v is not None}


def _vlm_tokenizer_path(model_cfg: Any) -> str:
    return (
        os.environ.get("EAGLE_LOCAL_PATH")
        or model_cfg.vlm_tokenizer_path
        or "aravindhs-NV/eagle3-processor-groot-n1d6"
    )


@register_preprocessor("Gr00tN1d6")
class GrootN1d6Preprocessor(BasePreprocessor):
    """Batch-level GR00T preprocessor.

    Per-sample transforms are expected to produce ``vlm_content``, normalized
    ``state``/``action`` arrays, masks, and embodiment IDs. This collator only
    tokenizes/pads text+images and stacks already prepared numeric fields.
    """

    def __init__(
        self,
        model_cfg: Any,
        data_cfg: Any,
        dataset_stats: Optional[Dict[str, Any]] = None,
        dataset: Any = None,
        max_length: Optional[int] = None,
    ):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.dataset_stats = dataset_stats
        self.dataset = dataset
        self.max_length = max_length
        self.preprocess_mode = data_cfg.groot_preprocess_mode
        self._collator = None

    @classmethod
    def from_config(
        cls,
        model_cfg,
        data_cfg,
        training_args=None,
        dataset_stats: Optional[Dict[str, Any]] = None,
        dataset: Any = None,
    ) -> "GrootN1d6Preprocessor":
        """Construct GrootN1d6Preprocessor from typed ModelConfig + DataConfig."""
        max_length = training_args.cuda_graph_pad_length if training_args is not None else None
        if max_length == 0:
            max_length = None
        if max_length is None:
            max_length = data_cfg.max_token_len
        return cls(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            dataset_stats=dataset_stats,
            dataset=dataset,
            max_length=max_length,
        )

    @property
    def collator(self):
        """Return the Gr00tN1d6DataCollator instance, initializing it on first access."""
        if self._collator is None:
            kwargs = {"local_files_only": True, "trust_remote_code": True}
            self._collator = Gr00tN1d6DataCollator(
                model_name=self.model_cfg.model_name,
                vlm_tokenizer_path=_vlm_tokenizer_path(self.model_cfg),
                model_type=self.model_cfg.backbone_model_type,
                transformers_loading_kwargs=kwargs,
                max_length=self.max_length,
            )
        return self._collator

    def __call__(self, examples: List[Dict[str, Any]]) -> GrootN1d6PreparedBatch:
        """Collate a list of sample dicts into a batched GrootN1d6PreparedBatch."""
        if not examples:
            raise ValueError("GrootN1d6Preprocessor received an empty batch")
        if self.preprocess_mode != "sample":
            raise ValueError(
                "groot_preprocess_mode must be 'sample' after GR00T preprocessing "
                "was moved into per-sample transforms; "
                f"got {self.preprocess_mode!r}"
            )
        processed = self._call_sample_preprocessed(examples)

        return self._to_prepared_batch(processed)

    def _call_sample_preprocessed(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "vlm_content" not in examples[0]:
            raise KeyError(
                "GR00T examples must contain 'vlm_content'. "
                "Ensure GrootN1d6FeatureTransform is registered in the per-sample pipeline."
            )
        return dict(self.collator(examples).data["inputs"])

    def _to_prepared_batch(self, processed: Dict[str, Any]) -> GrootN1d6PreparedBatch:
        action = processed.get("action")
        if action is None:
            raise KeyError("GR00T preprocessor did not produce required 'action' tensor")

        action_mask = processed.get("action_mask")
        if action_mask is None:
            action_mask = torch.ones_like(action, dtype=torch.float32)

        embodiment_id = processed.get("embodiment_id")
        if not isinstance(embodiment_id, torch.Tensor):
            batch_size = action.shape[0]
            embodiment_id = torch.full(
                (batch_size,),
                int(embodiment_id if embodiment_id is not None else 0),
                dtype=torch.long,
            )
        else:
            embodiment_id = embodiment_id.long().flatten()

        return GrootN1d6PreparedBatch(
            input_ids=processed.get("input_ids"),
            attention_mask=processed.get("attention_mask"),
            pixel_values=processed.get("pixel_values"),
            state=processed.get("state"),
            actions=action,
            action_mask=action_mask,
            embodiment_id=embodiment_id,
            action_is_pad=processed.get("action_is_pad"),
        )
