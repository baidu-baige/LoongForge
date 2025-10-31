""" MultiMixQASample """

from dataclasses import dataclass
from typing import List, Optional
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import VideoData
import torch


@dataclass
class MultiMixQASample(Sample):
    """Sample type for mix question answering."""

    #: The context/question for the video, image or pure text QA.
    messages: List[dict]

    #: The video data containing the image and audio info.
    video: List[VideoData] = None

    #: The input image tensor in the shape (C, H, W)
    image: List[torch.Tensor] = None

    # system
    system: Optional[str] = None