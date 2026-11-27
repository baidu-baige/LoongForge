# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible chat-completions client with image parts and an image history window."""

from __future__ import annotations

import base64
import copy
import io
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

OMITTED_IMAGE = "[earlier image omitted]"


def image_part(rgb: np.ndarray, frame: Optional[str] = None) -> Dict[str, Any]:
    """An internal image content part; converted to a data URL only when sent."""
    return {"type": "image", "image": rgb, "frame": frame}


def text_part(text: str) -> Dict[str, Any]:
    """A text content part."""
    return {"type": "text", "text": text}


def _png_data_url(rgb: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def to_openai(messages: List[Dict[str, Any]], max_image_messages: int = 0) -> List[Dict[str, Any]]:
    """Convert internal messages for the API.

    Only the images of the last ``max_image_messages`` image-bearing messages are
    sent (0 sends all); older ones become a short text placeholder so the
    request size stays bounded on long episodes.
    """
    bearing = [
        i
        for i, m in enumerate(messages)
        if isinstance(m.get("content"), list) and any(p.get("type") == "image" for p in m["content"])
    ]
    keep = set(bearing if max_image_messages <= 0 else bearing[-max_image_messages:])
    out = []
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = []
        for part in content:
            if part.get("type") != "image":
                parts.append(part)
            elif index in keep:
                parts.append({"type": "image_url", "image_url": {"url": _png_data_url(part["image"])}})
            else:
                parts.append(text_part(OMITTED_IMAGE))
        out.append({**message, "content": parts})
    return out


def for_transcript(message: Dict[str, Any]) -> Dict[str, Any]:
    """A JSON-safe copy of an internal message, images replaced by their frame names."""
    message = copy.copy(message)
    if isinstance(message.get("content"), list):
        message["content"] = [
            {"type": "image", "frame": p.get("frame")} if p.get("type") == "image" else p for p in message["content"]
        ]
    return message


@dataclass
class Usage:
    """Token totals of one episode."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0


@dataclass
class LLMConfig:
    """Endpoint and sampling settings."""

    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout_s: float = 600.0
    max_retries: int = 3
    max_image_messages: int = 0
    extra_body: Dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """The endpoint failed after all retries."""


class LLMClient:
    """Thin wrapper around ``openai.OpenAI().chat.completions``."""

    def __init__(self, config: LLMConfig) -> None:
        from openai import OpenAI

        self.config = config
        self.usage = Usage()
        self._client = OpenAI(
            base_url=config.base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=config.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            timeout=config.timeout_s,
            max_retries=0,
        )

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """One completion; returns ``{"content", "reasoning", "tool_calls"}``."""
        request: Dict[str, Any] = {
            "model": self.config.model,
            "messages": to_openai(messages, self.config.max_image_messages),
        }
        if tools:
            request["tools"] = tools
        if self.config.temperature is not None:
            request["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            request["max_tokens"] = self.config.max_tokens
        if self.config.extra_body:
            request["extra_body"] = self.config.extra_body
        last: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**request)
                break
            except Exception as exc:  # network, 5xx, rate limits: retry with backoff
                last = exc
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(min(30.0, 2.0 * 2**attempt))
        else:
            raise LLMError(str(last)) from last
        self.usage.calls += 1
        if response.usage is not None:
            self.usage.prompt_tokens += response.usage.prompt_tokens or 0
            self.usage.completion_tokens += response.usage.completion_tokens or 0
        message = response.choices[0].message
        calls = [
            {"id": call.id, "name": call.function.name, "arguments": call.function.arguments}
            for call in (message.tool_calls or [])
        ]
        return {
            "content": message.content or "",
            "reasoning": getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None),
            "tool_calls": calls,
        }
