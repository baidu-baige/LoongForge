# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""PayloadBuilder base class.

The PayloadBuilder consumes a canonical dict (benchmark-neutral, model-aware)
and produces the ``kwargs`` for ``model.predict_action(**kwargs)``. Subclasses
declare their configurable fields via **class attributes with type
annotations** — the annotated names form the whitelist for YAML overrides in
``__init__``. Class attributes without a type annotation are treated as
internal data (not overridable from YAML).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


class PayloadBuilder:
    """Convert canonical eval dicts into model-specific predict_action kwargs.

    Subclasses declare capability defaults as annotated class attributes:

        class MyBuilder(PayloadBuilder):
            state_encoding: str = "ee6d"
            action_encoding: str = "ee6d"
            action_dim: int = 20
            action_horizon: int = 30

    ``__init__`` reads only annotated names from ``yaml_model`` and overrides
    them via ``setattr``. Anything unannotated stays as a pure class attribute.
    """

    # Fallback capability defaults (subclasses override; kept here so
    # resolve_action_decoder can rely on the attribute always existing).
    state_encoding: str = ""
    action_encoding: str = ""
    action_dim: int = 0
    action_horizon: int = 1
    # Set True by models whose inference is closed-loop *within* a chunk, i.e.
    # they must observe every env step rather than being called once per chunk.
    # Runners forward this to the RPC so ``GenericPredictActionPolicy`` skips its
    # chunk cache; the model then owns the action queue. Default False keeps the
    # existing chunk-cached behaviour for every other model.
    disable_action_cache: bool = False

    def __init__(
        self,
        yaml_model: Optional[Mapping[str, Any]] = None,
        yaml_server: Optional[Mapping[str, Any]] = None,
        yaml_benchmark: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Merge YAML overrides on top of annotated class-attribute defaults."""
        yaml_model = dict(yaml_model or {})
        yaml_server = dict(yaml_server or {})
        yaml_benchmark = dict(yaml_benchmark or {})
        annotated = self._collect_annotated_fields()
        for source in (yaml_model, yaml_benchmark, yaml_server):
            for key in annotated:
                if key in source:
                    setattr(self, key, source[key])
        self._yaml_model = yaml_model
        self._yaml_server = yaml_server
        self._yaml_benchmark = yaml_benchmark

    @classmethod
    def _collect_annotated_fields(cls) -> set:
        """Union of type-annotated class attributes across the MRO."""
        names: set = set()
        for klass in reversed(cls.__mro__):
            names.update(getattr(klass, "__annotations__", {}) or {})
        # Do not treat internal fields as YAML-overridable.
        names.discard("_yaml_model")
        names.discard("_yaml_server")
        names.discard("_yaml_benchmark")
        return names

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def build(self, canonical: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return the ``kwargs`` for ``model.predict_action(**kwargs)``.

        Args:
            canonical: Benchmark-neutral observation dict produced by an
                adapter (``instruction`` / ``images`` / ``state_raw`` /
                ``meta`` fields, plus any adapter-specific extras).
            ctx: Per-step ``RuntimeContext`` populated by the runner.

        Returns:
            A dict of kwargs that will be forwarded to the transport client's
            ``predict_action`` call. RPC-control fields (``episode_id`` etc)
            are added by the runner around this dict.
        """
        raise NotImplementedError

    # Optional stateful hooks (default no-ops).
    def reset(self, episode_id: str) -> None:
        """Clear per-episode caches (frame stacks, backfill buffers, ...)."""

    def update_from_response(self, response: Any) -> None:
        """Consume the raw model response for closed-loop / backfill logic."""

    def note_env_action(self, env_action: Any) -> None:
        """Consume the decoded env action for closed-loop proprio backfill.

        Used by benchmarks whose next-step proprio depends on the last
        *decoded* action (e.g. RoboTwin ``ee6d_dual`` overwrites the endpose
        with the last commanded ee action). Default no-op.
        """
