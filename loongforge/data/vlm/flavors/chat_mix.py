# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unpacked chat-format multimodal sample (supports tool calling).

Counterpart of :class:`PackedChatMixSample` for the streaming / online path:
each sample carries a single conversation (`messages`) plus optional tool
definitions, optional system prompt, and optional image / video media.
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
class ChatMixSample(Sample):
    """Unpacked multimodal sample using full chat schema (incl. tool calling)."""

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
