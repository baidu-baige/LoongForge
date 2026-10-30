# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark adapter interface.

Adapters live on the simulator side and must not import model frameworks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class BaseBenchmarkAdapter:
    """Provide BaseBenchmarkAdapter behavior."""

    def obs_to_canonical(self, env_obs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert benchmark-native observation to Canonical Observation."""
        raise NotImplementedError

    def action_from_canonical(self, canonical_action: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Convert Canonical Action to benchmark-native action."""
        raise NotImplementedError

    def get_eval_context(self) -> Dict[str, Any]:
        """Return benchmark metadata such as control_hz, max_steps, and action_scale."""
        raise NotImplementedError

    def judge_success(self, episode_replay: Dict[str, Any]) -> Tuple[Optional[bool], str]:
        """Real-robot success oracle hook. Sim adapters normally do not implement this."""
        raise NotImplementedError

    def wait_reset_complete(self) -> None:
        """Real-robot async reset hook. Sim adapters normally do not implement this."""
        raise NotImplementedError
