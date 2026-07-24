# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from X-VLA (https://github.com/2toinf/X-VLA).
# Copyright 2025 2toINF. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XVLA per-sample transforms.

Thin :class:`BaseTransform` wrappers around the processing cores in
:mod:`loongforge.embodied.model.xvla.xvla_processor`. The cores hold the
tokenizer / image_processor loading + encode logic and have no dependency on
the training-side ``loongforge.embodied.data`` package, so they can be reused
from inference paths (e.g. :meth:`XVLAPolicy.predict_action`) without
triggering the DataLoader / DistributedContext import chain.

Two transforms are provided here:

* :class:`XVLATokenizeTransform`  — tokenizes ``task``/``prompt`` into
  ``input_ids``. Mirrors :class:`Pi05TokenizeTransform`.

* :class:`XVLAEncodeImageTransform` — converts ``observation.images.*``
  tensors into the ``image_input`` / ``image_mask`` tensors expected by the
  model. Mirrors :class:`Pi05CollateImagesTransform`.

Both classes inherit the batch helpers (``encode_language_batch`` /
``encode_image_batch`` / ``_to_pil``) from their processing core, so the
collator path continues to work unchanged.
"""

import os
from typing import Any, Dict, List

import torch

from loongforge.embodied.data.datasets.transforms.base import BaseTransform
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)
from loongforge.embodied.model.xvla.xvla_processor import (
    XVLAImageProcessorCore,
    XVLATokenizerCore,
)
from loongforge.embodied.model.xvla.model_configuration_xvla import resolve_domain_id


class XVLATokenizeTransform(BaseTransform, XVLATokenizerCore):
    """Tokenize language instruction into ``input_ids``.

    Writes ``input_ids`` ([L]) into the sample dict.
    Replaces ``XVLAProcessor.encode_language``.
    """

    def __init__(
        self,
        tokenizer_path: str = "",
        language_max_length: int = 50,
        task_key: str = "task",
        prompt_key: str = "prompt",
        training: bool = True,
    ):
        BaseTransform.__init__(self, apply_to=["input_ids"], training=training)
        XVLATokenizerCore.__init__(
            self,
            tokenizer_path=tokenizer_path,
            language_max_length=language_max_length,
        )
        self.task_key = task_key
        self.prompt_key = prompt_key

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize the language instruction and write ``input_ids`` into data."""
        instruction = (
            data.get(self.prompt_key)
            or data.get(self.task_key)
            or data.get("lang")
            or ""
        )
        inputs = self.tokenizer(
            [str(instruction)],
            return_tensors="pt",
            padding="max_length",
            max_length=self.language_max_length,
            truncation=True,
        )
        # Squeeze batch dim: per-sample transform produces [L], not [1, L]
        data["input_ids"] = inputs["input_ids"].squeeze(0)
        return data


class XVLAEncodeImageTransform(BaseTransform, XVLAImageProcessorCore):
    """Encode ``observation.images.*`` views into ``image_input`` / ``image_mask``.

    Writes:
    * ``image_input``: tensor [num_views, C, H, W]
    * ``image_mask``:  bool tensor [num_views]

    into the sample dict. When the dataset already provides ``image_input``
    (the reference-aligned HDF5VLADataset path), this transform is a no-op so
    the pipeline is idempotent.

    Replaces ``XVLAProcessor.encode_image``.
    """

    def __init__(
        self,
        tokenizer_path: str = "",
        num_views: int = 3,
        training: bool = True,
    ):
        BaseTransform.__init__(
            self, apply_to=["image_input", "image_mask"], training=training
        )
        XVLAImageProcessorCore.__init__(
            self, tokenizer_path=tokenizer_path, num_views=num_views
        )

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encode per-view images and write ``image_input`` / ``image_mask`` into data.

        If ``image_input`` already exists (reference-aligned path), this
        transform is a no-op so the pipeline remains idempotent.
        """
        if "image_input" in data:
            return data

        image_keys = sorted(k for k in data.keys() if k.startswith("observation.images."))
        pil_images = [self._to_pil(data[k]) for k in image_keys]

        encoded = self._encode_single(pil_images)
        data["image_input"] = encoded["image_input"]   # [num_views, C, H, W]
        data["image_mask"] = encoded["image_mask"]     # [num_views]
        return data


class XVLADomainIdTransform(BaseTransform):
    """Resolve ``robot_type`` into the per-sample ``domain_id``.

    Moves the domain-id generation out of :class:`HDF5VLADataset` without
    changing the logic. The dataset now passes the raw ``robot_type`` string
    through in the sample dict; this transform maps it to a domain index via
    ``DOMAIN_ID_MAP`` and writes a 0-dim ``long`` tensor under ``domain_id``,
    exactly as the dataset did inline
    (``_DOMAIN_ID.get(robot_type, 0)`` -> ``torch.tensor(..., torch.long)``).
    The transient ``robot_type`` key is consumed here so it does not reach the
    collator. Missing ``robot_type`` maps to 0 (same fallback as before).
    """

    def __init__(self, robot_type_key: str = "robot_type", training: bool = True):
        BaseTransform.__init__(self, apply_to=["domain_id"], training=training)
        self.robot_type_key = robot_type_key

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Consume ``robot_type`` and write ``domain_id`` (0-dim long tensor)."""
        robot_type = data.pop(self.robot_type_key, "")
        data["domain_id"] = torch.tensor(
            resolve_domain_id(robot_type), dtype=torch.long
        )
        return data


@register_transform_builder("xvla")
def build_xvla_transforms(ctx: TransformBuilderContext):
    """Append XVLA-specific per-sample transforms if model_type is xvla.

    Three transforms are appended in order:
    1. XVLAEncodeImageTransform – encode ``observation.images.*`` views into
       ``image_input`` / ``image_mask`` via XVLAProcessor.encode_image.
       No-op when the dataset already provides ``image_input`` (reference path).
    2. XVLATokenizeTransform – tokenize the language instruction into
       ``input_ids`` via XVLAProcessor.encode_language.
    3. XVLADomainIdTransform – resolve the sample's ``robot_type`` into
       ``domain_id`` (moved out of ``HDF5VLADataset``).
    """
    transforms: list = []
    model_cfg = ctx.model_cfg
    model_type = model_cfg.model_type
    if model_type != "xvla":
        return transforms

    tokenizer_path = ctx.training_args.tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
    num_views = (
        ctx.data_cfg.num_image_views
        or model_cfg.num_image_views
        or 3
    )

    transforms.append(XVLAEncodeImageTransform(
        tokenizer_path=tokenizer_path,
        num_views=num_views,
    ))
    transforms.append(XVLATokenizeTransform(
        tokenizer_path=tokenizer_path,
    ))
    transforms.append(XVLADomainIdTransform())
    return transforms
