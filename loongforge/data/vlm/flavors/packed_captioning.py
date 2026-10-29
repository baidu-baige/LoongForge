# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""PackedCaptioningSample"""

from dataclasses import dataclass
from typing import List, Optional
from megatron.energon.flavors.base_dataset import Sample
import torch


@dataclass
class PackedCaptioningSample(Sample):
    """Sample type for packed captioning."""

    # sample_id: str
    images: List[torch.Tensor]
    prompts: Optional[List[str]]
    captions: List[str]
