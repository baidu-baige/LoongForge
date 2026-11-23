# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge policy adapters for the standalone eval server."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.protocol import PROTOCOL_VERSION
from loongforge.embodied.eval.servers.predict_action_interface import PredictActionModel, call_predict_action

logger = logging.getLogger(__name__)


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
        # ``reset`` runs on the RPC event loop while inference runs in an
        # executor, so cache access needs its own lock. A generation counter
        # drops a chunk whose inference started before the reset and would
        # otherwise be written back afterwards.
        self._cache_lock = threading.Lock()
        self._cache_generation: Dict[str, int] = {}
        self._request_id_prefix = request_id_prefix
        # One model instance serves every sim client pinned to this server, and
        # the RPC server dispatches their requests from an executor. Models that
        # keep state across calls (Wan-style causal-VAE feat caches, streaming
        # KV caches, per-episode buffers) must never see two calls at once: a
        # VAE cache collision raises a shape error, a KV-cache collision silently
        # mixes episodes. Serialize at the policy boundary so the guarantee does
        # not depend on the transport path (with or without batching).
        self._infer_lock = threading.RLock()
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

    @property
    def supports_batching(self) -> bool:
        """Whether the underlying model exposes a real batch contract (T-06)."""
        return bool(self._metadata.get("supports_batching", False))

    def reset(self, episode_id: str) -> Dict[str, Any]:
        """Reset action cache for an episode."""
        with self._cache_lock:
            self._chunk_cache.pop(episode_id, None)
            self._cache_generation[episode_id] = self._cache_generation.get(episode_id, 0) + 1
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
        """Handle an eval RPC predict_action request with chunk caching.

        Payload schema (produced by every per-model ``PayloadBuilder``):
        ``images`` is a list of views ordered per model expectation and
        ``instructions`` is a list-of-str batch (batch=1). RPC-control fields
        (``disable_action_cache`` / ``return_action_chunk``) live at top level.
        The value handed to ``model.predict_action`` is a batch-of-1
        list-of-list of views: ``[[view_primary, view_wrist]]``.
        """
        with self._infer_lock:
            return self._predict_action(images, instructions, episode_id, episode_step, state, **kwargs)

    def _predict_action(
        self,
        images: list,
        instructions: list,
        episode_id: str = "default",
        episode_step: int = 0,
        state: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Unlocked body of :meth:`predict_action` (caller holds ``_infer_lock``)."""
        disable_action_cache = bool(kwargs.pop("disable_action_cache", False))
        return_action_chunk = bool(kwargs.pop("return_action_chunk", False))
        # Optional generation captured at submit time by the server-side batcher.
        # When absent (direct call, or a policy with no ``capture_cache_generation``)
        # the write-back guard falls back to the generation read below.
        arrival_generation = kwargs.pop("_cache_generation", None)
        # When the client owns the chunk it will not come back until the chunk is
        # exhausted. Caching server-side in that case would replay the *old*
        # chunk on the next request instead of running inference.
        use_cache = not (disable_action_cache or return_action_chunk)
        with self._cache_lock:
            generation = self._cache_generation.get(episode_id, 0)
            cache_entry = self._chunk_cache.get(episode_id) if use_cache else None
        if cache_entry is not None:
            chunk_index, chunk = cache_entry
            chunk_index += 1
            if chunk_index < chunk.shape[0]:
                stale = False
                with self._cache_lock:
                    if self._cache_generation.get(episode_id, 0) != generation:
                        # A reset landed between the read and this advance; the
                        # cached chunk belongs to the old episode.
                        stale = True
                    else:
                        self._chunk_cache[episode_id] = (chunk_index, chunk)
                if not stale:
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
        if use_cache:
            write_generation = arrival_generation if arrival_generation is not None else generation
            with self._cache_lock:
                # A reset that landed while this inference was in flight must not
                # see the old episode's chunk resurrected here.
                if self._cache_generation.get(episode_id, 0) == write_generation:
                    self._chunk_cache[episode_id] = (0, chunk)
        return {
            "actions": chunk if return_action_chunk else chunk[0:1],
            "inference_latency_ms": inference_latency_ms,
            "request_id": self._request_id(episode_id, episode_step),
        }

    def has_cached_action(self, payload: Dict[str, Any]) -> bool:
        """Return True when this payload can be served from the chunk cache.

        Used by the server-side batcher to keep cache hits out of the inference
        queue. Client-owned chunks (``return_action_chunk``) are never cached
        server-side, so they always need inference.
        """
        if payload.get("disable_action_cache") or payload.get("return_action_chunk"):
            return False
        with self._cache_lock:
            entry = self._chunk_cache.get(str(payload.get("episode_id", "default")))
        return entry is not None and entry[0] + 1 < entry[1].shape[0]

    def capture_cache_generation(self, payload: Dict[str, Any]) -> int:
        """Return the cache generation for a payload's episode.

        Called by ``AsyncBatcher.submit`` on the event loop, so it is strictly
        ordered against ``reset`` (which is also routed inline on the event loop).
        The write-back guard must compare against this value, not a generation
        re-read after the batch is assembled, otherwise a chunk whose request
        predated a reset would be written back and replayed.
        """
        with self._cache_lock:
            return self._cache_generation.get(str(payload.get("episode_id", "default")), 0)

    def predict_action_batch(self, payloads: list, generations: Optional[list] = None) -> list:
        """Process several payloads in one model call when supported.

        Batch rows use the same fixed noise seed as the batch=1 path, so a
        request's result does not depend on whether it shared a batch. If the
        model has no batch contract (or any payload is not batch-compatible),
        fall back to the sequential path.

        ``generations`` are the cache generations captured at submit time (one
        per payload); they drive the write-back guard so a chunk whose request
        predated a reset is never cached.
        """
        if not self.supports_batching or len(payloads) <= 1:
            if generations is None:
                generations = [None] * len(payloads)
            return [
                self.predict_action(**payload, _cache_generation=gen)
                for payload, gen in zip(payloads, generations)
            ]
        try:
            return self._predict_action_batch(payloads, generations)
        except Exception:
            logger.warning(
                "Batched inference failed; falling back to sequential requests.",
                exc_info=True,
            )
            if generations is None:
                generations = [None] * len(payloads)
            return [
                self.predict_action(**payload, _cache_generation=gen)
                for payload, gen in zip(payloads, generations)
            ]

    def _predict_action_batch(self, payloads: list, generations: Optional[list] = None) -> list:
        modes = {
            (bool(payload.get("disable_action_cache")), bool(payload.get("return_action_chunk")))
            for payload in payloads
        }
        if len(modes) != 1:
            raise ValueError("cannot batch payloads with different cache modes")
        if generations is None:
            # Legacy call path: no arrival-time generation, so read it here (the
            # race is only closed when the batcher threads ``capture_cache_generation``).
            generations = []
            for payload in payloads:
                with self._cache_lock:
                    generations.append(
                        self._cache_generation.get(str(payload.get("episode_id", "default")), 0)
                    )
        elif len(generations) != len(payloads):
            raise ValueError("generations length must match payloads length")
        # No fixed seed by default:每一行各自采样（与官方 openpi 客户端一致）。
        # 设置 ``metadata.noise_seed`` 时才退化为可复现模式（所有行共用该种子）。
        base_seed = self._metadata.get("noise_seed")

        images = []
        instructions = []
        states = []
        for payload in payloads:
            payload_images = payload.get("images")
            payload_instructions = payload.get("instructions")
            if not isinstance(payload_images, (list, tuple)) or not payload_images:
                raise ValueError("batched inference expects a non-empty image list")
            views = list(payload_images)
            if isinstance(views[0], (list, tuple)):
                # Already batch-wrapped ([[view, ...]]).
                if len(views) != 1:
                    raise ValueError("batched inference expects batch=1 image payloads")
                views = list(views[0])
            images.append(views)
            if isinstance(payload_instructions, str):
                instructions.append(payload_instructions)
            elif isinstance(payload_instructions, (list, tuple)) and len(payload_instructions) == 1:
                instructions.append(payload_instructions[0])
            else:
                raise ValueError("batched inference expects one instruction per payload")
            raw_state = payload.get("state")
            # ``np.asarray(None)`` is a NaN scalar: keep the missing state
            # explicit and let the sequential path handle it instead of feeding
            # the model a NaN row.
            states.append(
                None if raw_state is None else np.asarray(raw_state, dtype=np.float32).reshape(-1)
            )
        if any(state is None for state in states) or len({state.shape for state in states}) != 1:
            return [
                self.predict_action(**payload, _cache_generation=gen)
                for payload, gen in zip(payloads, generations)
            ]
        state_batch = np.stack(states, axis=0)

        start = time.perf_counter()
        seed_kwargs = (
            {"seeds": [int(base_seed)] * len(payloads)} if base_seed is not None else {}
        )
        # Same instance-level state as the sequential path: hold the lock so a
        # batched call cannot overlap a plain request from the no-batcher path.
        with self._infer_lock:
            result = self._model.predict_action(
                images=images,
                instructions=instructions,
                state=state_batch,
                dataset_stats=self._dataset_stats,
                **seed_kwargs,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        actions = np.asarray(result, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None, :, :]
        if actions.shape[0] != len(payloads):
            raise ValueError(
                f"batched model returned {actions.shape[0]} rows for {len(payloads)} payloads"
            )
        # Models emit the padded max_action_dim; trim exactly like the
        # sequential path in call_predict_action so both paths agree.
        action_dim = int(self._metadata["action_dim"])
        if actions.shape[-1] < action_dim:
            raise ValueError(
                f"batched model returned action dim {actions.shape[-1]}, expected at least {action_dim}"
            )
        actions = actions[..., :action_dim]

        responses = []
        for payload, row, generation in zip(payloads, actions, generations):
            episode_id = str(payload.get("episode_id", "default"))
            if not payload.get("disable_action_cache") and not payload.get("return_action_chunk"):
                with self._cache_lock:
                    # Drop the write if the episode was reset while this batch ran.
                    if self._cache_generation.get(episode_id, 0) == generation:
                        self._chunk_cache[episode_id] = (0, row)
            responses.append(
                {
                    "actions": row if payload.get("return_action_chunk") else row[0:1],
                    "inference_latency_ms": latency_ms,
                    "request_id": self._request_id(
                        episode_id, int(payload.get("episode_step", 0))
                    ),
                }
            )
        return responses

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
