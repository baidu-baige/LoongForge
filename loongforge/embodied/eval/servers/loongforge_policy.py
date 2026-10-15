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
        self._episode_kwargs = self._supported_episode_kwargs(model)
        self._metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "action_dim": int(action_dim),
            "action_unnormalization": "model_predict_action",
            "supports_preempt": False,
            **metadata,
        }

    @staticmethod
    def _supported_episode_kwargs(model: PredictActionModel) -> tuple[str, ...]:
        """Which episode-tracking kwargs this model's ``predict_action`` accepts.

        Streaming models (per-episode KV cache / observation buffer) need to know
        which episode a call belongs to and whether it is the episode's first step.
        Models that do not declare these parameters are called exactly as before,
        so nothing is forwarded to them.
        """
        import inspect

        try:
            params = inspect.signature(model.predict_action).parameters
        except (TypeError, ValueError):
            return ()
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return ("episode_id", "episode_step")
        return tuple(name for name in ("episode_id", "episode_step") if name in params)

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
        images: list,
        instructions: list,
        episode_id: str = "default",
        episode_step: int = 0,
        state: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Handle a single-sample or explicit batched predict_action request.

        Payload schema (produced by every per-model ``PayloadBuilder``):
        For a normal request, ``images`` is a list of views and ``instructions``
        is a one-item list. With ``batch_request=True``, ``images`` is
        ``[batch][views]``, ``instructions`` is ``[batch]``, and ``state`` is
        ``[batch, state_dim]``. RPC-control fields
        (``disable_action_cache`` / ``return_action_chunk``) live at top level.
        The value handed to ``model.predict_action`` is a batch-of-1
        list-of-list of views: ``[[view_primary, view_wrist]]``.
        """
        batch_request = bool(kwargs.pop("batch_request", False))
        disable_action_cache = bool(kwargs.pop("disable_action_cache", False))
        return_action_chunk = bool(kwargs.pop("return_action_chunk", False))
        if batch_request and not disable_action_cache:
            raise ValueError("Batched predict_action requires disable_action_cache=True")
        if batch_request:
            batch_size = len(images)
            if batch_size < 1 or len(instructions) != batch_size:
                raise ValueError(
                    f"Invalid batched request: images={batch_size}, instructions={len(instructions)}"
                )
            if state is not None and np.asarray(state).shape[0] != batch_size:
                raise ValueError(
                    f"Invalid batched state: expected first dimension {batch_size}, "
                    f"got {np.asarray(state).shape}"
                )
            start_time = time.perf_counter()
            chunk = call_predict_action(
                self._model,
                images=images,
                instructions=instructions,
                state=state,
                dataset_stats=self._dataset_stats,
                action_dim=int(self._metadata["action_dim"]),
                preserve_batch=True,
                **kwargs,
            )
            if chunk.ndim != 3:
                raise ValueError(
                    f"Batched model output must be [B, H, D], got {chunk.shape}"
                )
            inference_latency_ms = (time.perf_counter() - start_time) * 1000.0
            if chunk.shape[0] != batch_size:
                raise ValueError(
                    f"Batched model output has batch {chunk.shape[0]}, expected {batch_size}"
                )
            return {
                "actions": chunk if return_action_chunk else chunk[:, 0:1, :],
                "inference_latency_ms": inference_latency_ms,
                "request_id": self._request_id("batch", episode_step),
            }

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

        start_time = time.perf_counter()
        episode_kwargs = {
            "episode_id": episode_id,
            "episode_step": episode_step,
        }
        chunk = call_predict_action(
            self._model,
            images=[images],
            instructions=instructions,
            state=state,
            dataset_stats=self._dataset_stats,
            action_dim=int(self._metadata["action_dim"]),
            **{k: episode_kwargs[k] for k in self._episode_kwargs},
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
