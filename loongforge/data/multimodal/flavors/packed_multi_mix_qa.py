# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MultiMixQASample"""

from dataclasses import dataclass
from typing import List, Optional
from megatron.energon.flavors.base_dataset import Sample
from importlib.metadata import version
if version('megatron-energon') < "7.0.0":
    from megatron.energon.flavors.webdataset import VideoData as AVData
else:
    from megatron.energon.flavors.webdataset import AVData
import torch


@dataclass
class PackedMultiMixQASample(Sample):
    """Sample type for packed multi mix qa."""

    images: Optional[List[List[torch.Tensor]]]
    videos: Optional[List[list[AVData]]]
    contexts: List[str]
    answers: Optional[List[List[str]]] = None
    answer_weights: Optional[List[torch.Tensor]] = None
