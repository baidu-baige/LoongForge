"""MultiMixQASample"""

from dataclasses import dataclass
from typing import List, Optional
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import VideoData
import torch


@dataclass
class PackedMultiMixQASample(Sample):
    """Sample type for packed multi mix qa."""

    images: Optional[List[List[torch.Tensor]]]
    videos: Optional[List[list[VideoData]]]
    contexts: List[str]
    answers: Optional[List[List[str]]] = None
    answer_weights: Optional[List[torch.Tensor]] = None
