# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""PackedCaptioningSample"""

from dataclasses import dataclass
from typing import List, Optional
from megatron.energon.flavors.base_dataset import Sample
import torch


@dataclass
class PackedVQASample(Sample):
    """Sample type for packed vqasample."""

    images: List[torch.Tensor]
    contexts: List[str]
    answers: Optional[List[List[str]]] = None
    answer_weights: Optional[List[torch.Tensor]] = None
