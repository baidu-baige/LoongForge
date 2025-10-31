""" MultiVidQASample """

from dataclasses import dataclass
from typing import List, Optional
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import VideoData


@dataclass
class MultiVidQASample(Sample):
    """Sample type for video question answering."""

    #: The video data containing the image and audio info.
    video: List[VideoData]
    #: The context/question for the video.
    messages: List[dict]
    # system
    system: Optional[str] = None
