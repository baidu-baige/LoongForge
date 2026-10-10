# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Energon sample types produced by the cookers in ``task_encoder.py``.

Unpacked (online) types: MultiMixQASample, MultiVidQASample, ChatMixSample.
Offline-packed types: PackedCaptioningSample, PackedVQASample,
PackedMultiMixQASample, PackedChatMixSample.
"""

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Dict, List, Optional

import torch
from megatron.energon.flavors.base_dataset import Sample
from packaging.version import Version

# Energon < 7 needs ``__subflavor__`` on samples and exposes ``VideoData`` instead of ``AVData``.
ENERGON_LT_7 = Version(version("megatron-energon")) < Version("7.0.0")

if ENERGON_LT_7:
    from megatron.energon.flavors.webdataset import VideoData as AVData
else:
    from megatron.energon.flavors.webdataset import AVData


@dataclass
class MultiMixQASample(Sample):
    """Sample type for mix question answering."""

    #: The context/question for the video, image or pure text QA.
    messages: List[dict]

    #: The video data containing the image and audio info.
    video: List[AVData] = None

    #: The input image tensor in the shape (C, H, W)
    image: List[torch.Tensor] = None

    # system
    system: Optional[str] = None


@dataclass
class MultiVidQASample(Sample):
    """Sample type for video question answering."""

    #: The video data containing the image and audio info.
    video: List[AVData]
    #: The context/question for the video.
    messages: List[dict]
    # system
    system: Optional[str] = None


@dataclass
class ChatMixSample(Sample):
    """Unpacked multimodal sample using full chat schema (incl. tool calling).

    Counterpart of :class:`PackedChatMixSample` for the streaming / online path:
    each sample carries a single conversation (`messages`) plus optional tool
    definitions, optional system prompt, and optional image / video media.
    """

    #: OpenAI Chat Completions-style messages for a single conversation.
    messages: List[Dict[str, Any]]

    #: Optional list of image tensors referenced by the messages.
    image: Optional[List[torch.Tensor]] = None

    #: Optional list of video data referenced by the messages.
    video: Optional[List[AVData]] = None

    #: Optional system prompt (extracted from messages by the cooker).
    system: Optional[str] = None

    #: Optional tool definitions available to the assistant for this conversation.
    tools: Optional[List[Dict[str, Any]]] = None


@dataclass
class PackedCaptioningSample(Sample):
    """Sample type for packed captioning."""

    # sample_id: str
    images: List[torch.Tensor]
    prompts: Optional[List[str]]
    captions: List[str]


@dataclass
class PackedVQASample(Sample):
    """Sample type for packed vqasample."""

    images: List[torch.Tensor]
    contexts: List[str]
    answers: Optional[List[List[str]]] = None
    answer_weights: Optional[List[torch.Tensor]] = None


@dataclass
class PackedMultiMixQASample(Sample):
    """Packed list of full MultiMixQA child samples.

    Each child sample is represented column-wise:
    - contexts[i]: user turns of child i
    - answers[i]: assistant turns of child i
    - images[i] / videos[i]: media group of child i

    A single-turn QA is represented as one-element lists, for example:
    contexts[i] = ["<image>\\nquestion"]
    answers[i] = ["answer"]
    """

    images: Optional[List[List[torch.Tensor]]]
    videos: Optional[List[list[AVData]]]
    contexts: List[List[str]]
    answers: List[List[str]]
    answer_weights: Optional[List[torch.Tensor]] = None


@dataclass
class PackedChatMixSample(Sample):
    """Offline-packed multimodal sample using full chat schema (incl. tool calling).

    Counterpart of :class:`ChatMixSample` for the offline-packed path: each sample
    carries N original conversations packed together, where every entry of
    ``packed_messages`` is a self-contained dict (typically with its own
    ``messages`` and optional ``tools`` keys), aligned 1:1 with ``packed_images``
    / ``packed_videos`` groups.
    """

    #: List of N packed conversations; each entry is a dict carrying its own
    #: ``messages`` (OpenAI Chat Completions-style) plus optional ``tools`` /
    #: ``source`` metadata.
    packed_messages: List[Dict[str, Any]]

    #: Optional per-conversation image groups; ``packed_images[i]`` belongs to
    #: ``packed_messages[i]``. ``None`` when ``media_type='video'`` or
    #: ``'text'``.
    packed_images: Optional[List[List[torch.Tensor]]] = None

    #: Optional per-conversation video groups; ``packed_videos[i]`` belongs to
    #: ``packed_messages[i]``. ``None`` when ``media_type='image'`` or
    #: ``'text'``.
    packed_videos: Optional[List[List[AVData]]] = None

    #: Optional system prompt shared across the packed conversations.
    system: Optional[str] = None
