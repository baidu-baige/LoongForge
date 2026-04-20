# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MultiVidQASample"""

from dataclasses import dataclass
from typing import List, Optional
from megatron.energon.flavors.base_dataset import Sample
from importlib.metadata import version
if version('megatron-energon') < "7.0.0":
    from megatron.energon.flavors.webdataset import VideoData as AVData
else:
    from megatron.energon.flavors.webdataset import AVData


@dataclass
class MultiVidQASample(Sample):
    """Sample type for video question answering."""

    #: The video data containing the image and audio info.
    video: List[AVData]
    #: The context/question for the video.
    messages: List[dict]
    # system
    system: Optional[str] = None
