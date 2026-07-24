# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.

"""GR00T-N1.7 collator for the embodied trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    PreparedBatch,
    register_preprocessor,
)
from loongforge.embodied.data.datasets.groot_n1_7.transforms.data_configuration_groot_n1_7 import (
    GrootN1d7DataConfig,
)
from loongforge.embodied.model.groot_n1_7.model_configuration_groot_n1_7 import GrootN1d7Config


@dataclass
class GrootN1d7PreparedBatch(PreparedBatch):
    """Model-ready batch for GR00T-N1.7."""

    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor = None
    pixel_values: Any = None
    image_grid_thw: torch.Tensor = None
    mm_token_type_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    state: torch.Tensor = None
    actions: torch.Tensor = None
    action_mask: torch.Tensor = None
    embodiment_id: torch.Tensor = None
    action_is_pad: Optional[torch.Tensor] = None

    def to_model_inputs(self) -> Dict[str, Any]:
        """Convert to model input dictionary."""
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
            "mm_token_type_ids": self.mm_token_type_ids,
            "position_ids": self.position_ids,
            "state": self.state,
            "action": self.actions,
            "action_mask": self.action_mask,
            "embodiment_id": self.embodiment_id,
            "action_is_pad": self.action_is_pad,
        }
        return {key: value for key, value in inputs.items() if value is not None}


class Gr00tN1d7DataCollator:
    """Tokenize/pad Qwen3-VL text+images and stack numeric fields."""

    def __init__(
        self,
        model_name: str,
        model_type: str = "qwen",
        transformers_loading_kwargs: dict | None = None,
        max_length: int | None = None,
    ):
        loading_kwargs = dict(transformers_loading_kwargs or {})
        from transformers import Qwen3VLProcessor

        self.processor = Qwen3VLProcessor.from_pretrained(model_name, **loading_kwargs)
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> BatchFeature:
        """Collate sample-preprocessed features."""
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                text_list = []
                image_inputs = []
                for value in values:
                    text = value.get("text")
                    if text is None:
                        text = self.processor.apply_chat_template(
                            value["conversation"],
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                    text_list.append(text)
                    image_inputs.extend(value["images"])
                vlm_inputs = self.processor(
                    text=text_list,
                    images=image_inputs,
                    return_tensors="pt",
                    padding="max_length" if self.max_length else True,
                    max_length=self.max_length,
                    truncation=self.max_length is not None,
                )
                for vlm_key, vlm_value in vlm_inputs.items():
                    batch[vlm_key] = vlm_value
            else:
                batch[key] = torch.from_numpy(np.stack(values))

        return BatchFeature(data={"inputs": batch})


@register_preprocessor("Gr00tN1d7")
class GrootN1d7Preprocessor(BasePreprocessor):
    """Batch-level preprocessor for GR00T-N1.7."""

    def __init__(
        self,
        policy_cfg: GrootN1d7Config,
        data_cfg: Any,
        max_length: Optional[int] = None,
    ):
        self.policy_cfg = policy_cfg
        self.data_cfg = data_cfg
        self.max_length = max_length
        self._collator = None

    @classmethod
    def from_config(
        cls,
        model_cfg,
        data_cfg,
        training_args=None,
        dataset_stats=None,
        dataset=None,
    ) -> "GrootN1d7Preprocessor":
        """Construct from LoongForge model config."""
        policy_cfg = GrootN1d7Config.from_config(model_cfg)
        data_cfg = GrootN1d7DataConfig() if data_cfg is None else data_cfg
        max_length = training_args.cuda_graph_pad_length if training_args is not None else None
        if max_length == 0:
            max_length = None
        if max_length is None:
            max_length = data_cfg.max_token_len
        return cls(policy_cfg=policy_cfg, data_cfg=data_cfg, max_length=max_length)

    @property
    def collator(self) -> Gr00tN1d7DataCollator:
        """Create Qwen3-VL collator lazily."""
        if self._collator is None:
            self._collator = Gr00tN1d7DataCollator(
                model_name=self.policy_cfg.model_name,
                model_type=self.policy_cfg.backbone_model_type,
                transformers_loading_kwargs={
                    "trust_remote_code": True,
                    "local_files_only": True,
                },
                max_length=self.max_length,
            )
        return self._collator

    def __call__(self, examples: List[Dict[str, Any]]) -> GrootN1d7PreparedBatch:
        """Collate sample-preprocessed examples."""
        if not examples:
            raise ValueError("GrootN1d7Preprocessor received an empty batch")
        if "vlm_content" not in examples[0]:
            raise KeyError("GR00T-N1.7 examples must contain 'vlm_content'")
        processed = dict(self.collator(examples).data["inputs"])
        return self._to_prepared_batch(processed)

    def _to_prepared_batch(self, processed: Dict[str, Any]) -> GrootN1d7PreparedBatch:
        action = processed.get("action")
        if action is None:
            raise KeyError("GR00T-N1.7 preprocessor did not produce required 'action' tensor")

        action_mask = processed.get("action_mask")
        if action_mask is None:
            action_mask = torch.ones_like(action, dtype=torch.float32)

        embodiment_id = processed.get("embodiment_id")
        if not isinstance(embodiment_id, torch.Tensor):
            embodiment_id = torch.full(
                (action.shape[0],),
                int(embodiment_id if embodiment_id is not None else 0),
                dtype=torch.long,
            )
        else:
            embodiment_id = embodiment_id.long().flatten()

        return GrootN1d7PreparedBatch(
            input_ids=processed.get("input_ids"),
            attention_mask=processed.get("attention_mask"),
            pixel_values=processed.get("pixel_values"),
            image_grid_thw=processed.get("image_grid_thw"),
            mm_token_type_ids=processed.get("mm_token_type_ids"),
            position_ids=processed.get("position_ids"),
            state=processed.get("state"),
            actions=action,
            action_mask=action_mask,
            embodiment_id=embodiment_id,
            action_is_pad=processed.get("action_is_pad"),
        )
