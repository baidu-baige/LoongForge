# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""PackedMultiMixQASample"""

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
    """Packed list of full MultiMixQA child samples.

    Each child sample is represented column-wise:
    - contexts[i]: user turns of child i
    - answers[i]: assistant turns of child i
    - images[i] / videos[i]: media group of child i

    A single-turn QA is represented as one-element lists, for example:
    contexts[i] = ["<image>\nquestion"]
    answers[i] = ["answer"]
    """

    images: Optional[List[List[torch.Tensor]]]
    videos: Optional[List[list[AVData]]]
    contexts: List[List[str]]
    answers: List[List[str]]
    answer_weights: Optional[List[torch.Tensor]] = None
