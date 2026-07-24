# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge policy adapters for the standalone eval server."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.protocol import PROTOCOL_VERSION
from loongforge.embodied.eval.servers.predict_action_interface import PredictActionModel, call_predict_action


@dataclass
class PredictActionModelSpec:
    """Model instance and metadata needed by the generic eval policy."""

    model: PredictActionModel
    metadata: Dict[str, Any]


class GenericPredictActionPolicy:
    """Generic eval RPC policy for models exposing predict_action."""

    def __init__(
        self,
        model: PredictActionModel,
        metadata: Dict[str, Any],
        dataset_statistics_path: str = "",
        action_dim: int = 7,
        request_id_prefix: str = "predict-action",
    ) -> None:
        """Initialize the generic predict_action eval policy."""
        self._model = model
        self._dataset_stats = self._load_dataset_stats(dataset_statistics_path)
        self._chunk_cache: Dict[str, tuple[int, np.ndarray]] = {}
        self._request_id_prefix = request_id_prefix
        self._metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "action_dim": int(action_dim),
            "action_unnormalization": "model_predict_action",
            "supports_preempt": False,
            **metadata,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return policy metadata for the eval server."""
        return dict(self._metadata)

    def reset(self, episode_id: str) -> Dict[str, Any]:
        """Reset action cache for an episode."""
        self._chunk_cache.pop(episode_id, None)
        return {"episode_id": episode_id}

    def predict_action(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
        episode_id: str = "default",
        episode_step: int = 0,
        state: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Handle an eval RPC predict_action request with chunk caching."""
        disable_action_cache = bool(kwargs.pop("disable_action_cache", False))
        return_action_chunk = bool(kwargs.pop("return_action_chunk", False))
        cache_entry = None if disable_action_cache else self._chunk_cache.get(episode_id)
        if cache_entry is not None:
            chunk_index, chunk = cache_entry
            chunk_index += 1
            if chunk_index < chunk.shape[0]:
                self._chunk_cache[episode_id] = (chunk_index, chunk)
                return {
                    "actions": chunk if return_action_chunk else chunk[chunk_index : chunk_index + 1],
                    "inference_latency_ms": None,
                    "request_id": self._request_id(episode_id, episode_step),
                }

        image_input = self._build_image_input(images)
        start_time = time.perf_counter()
        chunk = call_predict_action(
            self._model,
            images=[image_input],
            instructions=[instruction],
            state=state,
            dataset_stats=self._dataset_stats,
            action_dim=int(self._metadata["action_dim"]),
            **kwargs,
        )
        inference_latency_ms = (time.perf_counter() - start_time) * 1000.0
        if not disable_action_cache:
            self._chunk_cache[episode_id] = (0, chunk)
        return {
            "actions": chunk if return_action_chunk else chunk[0:1],
            "inference_latency_ms": inference_latency_ms,
            "request_id": self._request_id(episode_id, episode_step),
        }

    def _request_id(self, episode_id: str, episode_step: int) -> str:
        """Build a stable request id for eval responses."""
        return f"{self._request_id_prefix}-{episode_id}-{episode_step}"

    @staticmethod
    def _build_image_input(images: Dict[str, np.ndarray]) -> list[np.ndarray]:
        """Convert canonical eval image views into the common predict_action image input."""
        primary = images.get("primary")
        if primary is None:
            primary = images.get("head")
        if primary is None:
            raise ValueError("images.primary or images.head is required for predict_action inference")
        image_input = [np.asarray(primary)]

        left = images.get("left")
        right = images.get("right")
        if left is not None and right is not None:
            # Bimanual benchmarks (RoboTwin): official X-VLA client sends
            # image0=head, image1=left, image2=right.
            image_input.append(np.asarray(left))
            image_input.append(np.asarray(right))
            return image_input

        wrist = images.get("wrist")
        if wrist is None:
            wrist = right if right is not None else left
        if wrist is not None:
            image_input.append(np.asarray(wrist))
        return image_input

    @staticmethod
    def _load_dataset_stats(path: str) -> Optional[Dict[str, Any]]:
        """Load dataset statistics passed to model predict_action calls."""
        if not path:
            return None
        stats_path = Path(path).expanduser()
        if not stats_path.exists():
            raise FileNotFoundError(f"dataset statistics file not found: {path}")
        with stats_path.open("r", encoding="utf-8") as f:
            return json.load(f)


def __getattr__(name):
    # Lazy re-exports to avoid circular imports between servers/ and factories/.
    if name in ("PI05ModelFactory", "LoongForgePI05Policy"):
        from loongforge.embodied.eval.factories.pi05_factory import PI05ModelFactory, LoongForgePI05Policy
        return {"PI05ModelFactory": PI05ModelFactory, "LoongForgePI05Policy": LoongForgePI05Policy}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
