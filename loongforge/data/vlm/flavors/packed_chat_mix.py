# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Packed chat-format multimodal sample (supports tool calling).

Counterpart of :class:`ChatMixSample` for the offline-packed path: each sample
carries N original conversations packed together, where every entry of
``packed_messages`` is a self-contained dict (typically with its own
``messages`` and optional ``tools`` keys), aligned 1:1 with ``packed_images``
/ ``packed_videos`` groups.
"""

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Dict, List, Optional

import torch
from megatron.energon.flavors.base_dataset import Sample

if version("megatron-energon") < "7.0.0":
    from megatron.energon.flavors.webdataset import VideoData as AVData
else:
    from megatron.energon.flavors.webdataset import AVData


@dataclass
class PackedChatMixSample(Sample):
    """Offline-packed multimodal sample using full chat schema (incl. tool calling)."""

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
