# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Protocol-compliant mock policy for smoke tests."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.embodied.eval.protocol import PROTOCOL_VERSION


class MockPolicy:
    """Provide MockPolicy behavior."""

    def __init__(
        self,
        action_chunk_size: int = 8,
        action_dim: int = 7,
        default_unnorm_key: str = "libero_franka",
        available_unnorm_keys: Optional[List[str]] = None,
    ) -> None:
        """Run __init__."""
        self._action_chunk_size = action_chunk_size
        self._action_dim = action_dim
        self._default_unnorm_key = default_unnorm_key
        self._available_unnorm_keys = available_unnorm_keys or [default_unnorm_key]
        self._chunk_cache: Dict[str, tuple[int, np.ndarray]] = {}

    @property
    def metadata(self) -> Dict[str, Any]:
        """Run metadata."""
        return {
            "protocol_version": PROTOCOL_VERSION,
            "env": "vla_eval_mock_policy_server",
            "ckpt_path": "mock://policy",
            "action_chunk_size": self._action_chunk_size,
            "supports_preempt": False,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "action_keys": [
                "action.x",
                "action.y",
                "action.z",
                "action.roll",
                "action.pitch",
                "action.yaw",
                "action.gripper",
            ],
            "state_keys": [],
        }

    def reset(self, episode_id: str) -> Dict[str, Any]:
        """Run reset."""
        self._chunk_cache.pop(episode_id, None)
        return {"episode_id": episode_id}

    def predict_action(self, episode_id: str = "default", episode_step: int = 0, **_: Any) -> Dict[str, Any]:
        """Run predict_action."""
        cache_entry = self._chunk_cache.get(episode_id)
        if cache_entry is not None:
            chunk_index, chunk = cache_entry
            chunk_index += 1
            if chunk_index < self._action_chunk_size:
                self._chunk_cache[episode_id] = (chunk_index, chunk)
                return {
                    "actions": chunk[chunk_index : chunk_index + 1],
                    "inference_latency_ms": None,
                    "request_id": f"mock-{episode_id}-{episode_step}",
                }

        chunk = self._generate_chunk()
        self._chunk_cache[episode_id] = (0, chunk)
        return {
            "actions": chunk[0:1],
            "inference_latency_ms": random.uniform(5.0, 15.0),
            "request_id": f"mock-{episode_id}-{episode_step}",
        }

    def _generate_chunk(self) -> np.ndarray:
        """Run _generate_chunk."""
        chunk = np.zeros((self._action_chunk_size, self._action_dim), dtype=np.float32)
        chunk[:, :3] = np.random.uniform(-0.01, 0.01, (self._action_chunk_size, 3))
        chunk[:, 3:6] = np.random.uniform(-0.005, 0.005, (self._action_chunk_size, 3))
        chunk[:, 6] = np.random.uniform(-0.1, 0.1, self._action_chunk_size)
        return chunk
