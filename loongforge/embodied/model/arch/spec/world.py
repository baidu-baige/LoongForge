# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``WorldModelSpec``, its construction-time checks, and ``build_world_model``.

Signatures follow ``docs/design/mot_spec_config.zh-CN.md`` §3.2 / §3.8.

The nine checks are grouped by the layer they belong to, one method per layer:
``_validate_visibility_layer`` (L1), ``_validate_stream_layer`` (L2),
``_validate_diffusion_layer`` (L3) and ``_validate_head_layer`` (L4). L0
(geometry and layout) has nothing to check here, so it has no method.

This file closes the config chain: once a ``WorldModelSpec`` is constructed
successfully, every layer check has passed and ``WorldModel.__init__`` no
longer has to validate any shape.

**Every check raises ``ValueError``; there is not a single ``assert``.**
``python -O`` strips asserts, and these checks are exactly the ones production
needs most.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from ..diffusion.loss import LossSpec, WeightRamp
from ..diffusion.spec import DiffusionSpec
from ..geometry import GeometrySpec
from ..heads.spec import HeadSpec
from ..streams.conditions import ConditionSpec, InjectionKind
from ..streams.engine import AttentionGroup
from ..streams.spec import StreamSpec
from ..visibility.combine import Visibility
from .core import get_provider

__all__ = ["WorldModelSpec", "build_world_model"]

_KNOWN_BACKENDS = frozenset({"dense", "flex", "sdpa", "flash"})
"""Known attention backends.

``_validate_visibility_layer`` validates against this set until
``CompiledMask.supports`` ships (Lane A, milestone S4).
"""


@dataclass(frozen=True)
class WorldModelSpec:
    """Complete declaration of one world model.

    The return type of a provider and the only input of ``WorldModel``.

    ``groups`` is a **tuple of attention groups**, not a flat list of streams:
    in EWAM the VL stream and the video stream do not share one joint
    attention, and that fact has to be visible in the type. A monolithic model
    is one group holding one stream (N=1 degenerates, it is not a special
    case).

    Attributes:
        model_type: Provider registry key for this model.
        geometry: Token geometry shared by all streams.
        groups: Attention groups; each group is one joint attention.
        diffusion: Timestep schedule and loss table.
        heads: Output head declarations.
        conditions: Condition declarations (channel concat, cross-attention,
            append-to-slot).
        visibility: Keyed by **stream name**, the visibility on that stream's
            query side. A missing key means fully bidirectional
            (``structural=None``), which is what monolithic models and the
            existing motus code do.
        attention_backend: Concrete backend resolved from YAML; ``"auto"`` is
            rejected by the L1 check.
        attention_block_size: Only meaningful for ``backend="flex"``; it must
            equal the ``Scenario.block_size`` used during negotiation.
    """

    model_type: str
    geometry: GeometrySpec
    groups: tuple[AttentionGroup, ...]
    diffusion: DiffusionSpec
    heads: tuple[HeadSpec, ...] = ()
    conditions: tuple[ConditionSpec, ...] = ()
    visibility: Mapping[str, Visibility] = field(default_factory=dict)
    # Keyed by **stream name**; value is that stream's query-side visibility.
    # Missing key => fully bidirectional (structural=None).
    attention_backend: str = "dense"    # [YAML] resolved backend, not "auto"
    attention_block_size: int | None = None
    # Only meaningful for backend="flex"; must match Scenario.block_size.

    def __post_init__(self) -> None:
        self._validate_stream_layer()
        self._validate_diffusion_layer()
        self._validate_head_layer()
        self._validate_visibility_layer()
        object.__setattr__(self, "visibility",
                           MappingProxyType(dict(self.visibility)))

    # ---- queries ----

    @property
    def streams(self) -> tuple[StreamSpec, ...]:
        """All streams, ordered by group then by position inside the group.

        Returns:
            A stable ordering, which is what optimizer parameter grouping
            relies on.
        """
        return tuple(spec for group in self.groups for spec in group.members)

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.streams)

    def stream(self, name: str) -> StreamSpec:
        for spec in self.streams:
            if spec.name == name:
                return spec
        raise KeyError(
            f"model {self.model_type!r} has no stream {name!r}; "
            f"it has {list(self.stream_names)}"
        )

    def group_of(self, stream: str) -> AttentionGroup:
        """Find the attention group holding ``stream``.

        A stream never spans two groups: ``_check_stream_identity`` makes
        stream names globally unique.

        Args:
            stream: Stream name.

        Returns:
            The group whose members include ``stream``.

        Raises:
            KeyError: No stream by that name; the message lists the existing
                names.
        """
        for group in self.groups:
            if stream in group.stream_names:
                return group
        raise KeyError(
            f"model {self.model_type!r} has no stream {stream!r}; "
            f"it has {list(self.stream_names)}"
        )

    def head(self, name: str) -> HeadSpec:
        for spec in self.heads:
            if spec.name == name:
                return spec
        raise KeyError(
            f"model {self.model_type!r} has no output head {name!r}; "
            f"it has {[h.name for h in self.heads]}"
        )

    def condition(self, name: str) -> ConditionSpec:
        """Look up one condition declaration by name.

        The consumers (a stream module building its cross-attention condition
        projector, an input convolution counting channels) need
        ``ConditionSpec.width``, and the condition table is held only by this
        class. Letting stream modules scan ``conditions`` themselves would
        give the "slot name -> width" mapping a second implementation.

        Args:
            name: Condition name.

        Returns:
            The matching ``ConditionSpec``.

        Raises:
            KeyError: No condition by that name; the message lists the
                existing names.
        """
        for spec in self.conditions:
            if spec.name == name:
                return spec
        raise KeyError(
            f"model {self.model_type!r} has no condition {name!r}; "
            f"it has {[c.name for c in self.conditions]}"
        )

    def captures(self) -> dict[str, tuple[int, ...]]:
        """Derive the engine's capture requests from the head table.

        This is the request side of interface I3: declaration to request,
        derived in exactly one place.

        Returns:
            ``{stream_name: (joint_layer_id, ...)}`` with layer ids sorted.
        """
        wanted: dict[str, set[int]] = {}
        for spec in self.heads:
            key = spec.capture_key
            if key is not None:
                wanted.setdefault(key[0], set()).add(key[1])
        return {stream: tuple(sorted(layers))
                for stream, layers in wanted.items()}

    def appended_to(self, slot: str, stream: str) -> tuple[str, ...]:
        """Condition names appended to slot ``slot`` for stream ``stream``.

        The order is the **declaration order** written by the provider, which
        is also the concatenation order along the token axis. It therefore
        needs a single source: relying on lexicographic order or on dict
        insertion order shows up as "renaming a condition moved the KV that
        attention reads" while every shape stays identical.

        Entries with ``enabled=False`` are skipped. That is the opposite of
        ``concat_channels``: here turning one off really removes a token, and
        a cross-attention KV length is data-dependent anyway, so it enters no
        weight shape.

        Args:
            slot: Host cross-attention slot name.
            stream: Target stream name.

        Returns:
            The appended condition names in declaration order.

        Raises:
            KeyError: ``stream`` is not a stream of this model.
        """
        self.stream(stream)
        return tuple(
            cond.name for cond in self.conditions
            if cond.kind is InjectionKind.APPEND_TO_SLOT
            and cond.enabled and cond.append_to == slot
            and stream in cond.targets
        )

    def concat_channels(self, stream: str) -> int:
        """Sum of condition channels concatenated onto ``stream``'s channels.

        This is where the existing ``in_dim=36`` comes from:
        ``wan_video_dit.py:484`` does ``torch.cat([x, y], dim=1)`` where ``y``
        is itself ``torch.cat([mask, y], dim=1)`` at ``:418``. So
        36 = 16 stream channels + 4 mask + 16 condition-frame latent. It is a
        **derived quantity** and does not deserve a field.

        It lives on ``WorldModelSpec`` rather than ``StreamSpec`` because the
        sum needs the stream name and the condition table at the same time,
        and only this class holds both. Conditions with ``enabled=False``
        still count: turning a condition off changes its value, not the shape
        of the input convolution. If it could change the shape, one YAML
        switch would make the same checkpoint fail to load.

        Args:
            stream: Target stream name.

        Returns:
            Total extra channel count, 0 when nothing concatenates.

        Raises:
            KeyError: ``stream`` is not a stream of this model.
        """
        self.stream(stream)             # raises and lists the existing names
        return sum(
            condition.width or 0
            for condition in self.conditions
            if condition.kind is InjectionKind.CHANNEL_CONCAT
            and stream in condition.targets
        )

    # ---- Per-layer checks. All raise ValueError; not a single assert ----
    # Message tags read (L<layer>/V<check>), so an error still points back at
    # the numbered check in the design doc.

    def _validate_stream_layer(self) -> None:
        """L2: stream identity, groups, joint widths, routes and conditions."""
        self._check_stream_identity()
        self._check_layer_mapping()
        self._check_joint_head_dim()
        self._check_kv_routes()
        self._check_condition_slots()
        self._check_appended_slots()

    def _check_stream_identity(self) -> None:
        """Stream names are non-empty and globally unique.

        Globally unique rather than unique within a group: a stream name is
        at the same time a ``TokenLayout`` label value, a capture hook key,
        and an optimizer grouping key. Those three share one namespace.

        Raises:
            ValueError: ``groups`` is empty, a name is the empty string, or
                two streams share a name.
        """
        if not self.groups:
            raise ValueError(
                f"WorldModelSpec({self.model_type!r}).groups must not be "
                f"empty (L2/V1)"
            )
        names = [spec.name for spec in self.streams]
        if any(not name for name in names):
            raise ValueError(
                f"stream names must not be the empty string (L2/V1): {names}"
            )
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"stream names must be globally unique (L2/V1), duplicates "
                f"are {duplicates}. A stream name is at once a layout label "
                "value, a capture key and an optimizer grouping key"
            )

    def _check_layer_mapping(self) -> None:
        """Every stream's layer mapping resolves under its group's depth.

        ``resolve_layer_index_map`` already asserts that the number of
        resolved participating layers equals the stream's ``num_layers``, so
        calling it once per stream covers the boundary conditions of the
        STRIDED and AT_LAYERS mapping modes.

        Raises:
            ValueError: A stream's layer mapping cannot be resolved against
                its group's joint layer count.
        """
        for group in self.groups:
            for spec in group.members:
                spec.resolve_layer_index_map(group.num_layers)

    def _check_joint_head_dim(self) -> None:
        """Joint members share one ``head_dim``.

        ``num_heads`` and the residual width may differ between members; only
        ``head_dim`` has to match, because the members' queries and keys meet
        in one attention.

        ``AttentionGroup.__post_init__`` checks this already. Re-checking here
        is **deliberate**: a provider may hand-build an ``AttentionGroup`` or
        receive one from elsewhere, and checking again at the spec layer puts
        ``model_type`` in the error message.

        Raises:
            ValueError: Members of one group disagree on ``head_dim``.
        """
        for group in self.groups:
            head_dims = {spec.name: spec.head_dim for spec in group.members}
            if len(set(head_dims.values())) != 1:
                raise ValueError(
                    f"model {self.model_type!r}: attention group "
                    f"{list(group.stream_names)} has unequal head_dim "
                    f"(L2/V3): {head_dims}"
                )

    def _check_kv_routes(self) -> None:
        """KV route endpoints are in one group with in-range layer ids.

        Cross-group routing is rejected: each group owns its own joint
        attention, so taking KV across groups couples two attentions in a way
        that visibility (L1) cannot see.

        ``AttentionGroup.__post_init__`` already checks that both endpoints
        belong to the group; what is re-checked here are the layer ids on both
        sides. Route layer ids are **local** to each stream (the ``KVRoute``
        convention), and mixing them up with joint layer ids is the easiest
        mistake to make here.

        Raises:
            ValueError: A ``src_layers`` or ``dst_layers`` entry is >= that
                stream's ``num_layers``.
        """
        for group in self.groups:
            for route in group.routes:
                src_spec = group.spec_of(route.src)
                dst_spec = group.spec_of(route.dst)
                for side, layers, spec in (
                        ("src_layers", route.src_layers, src_spec),
                        ("dst_layers", route.dst_layers, dst_spec)):
                    bad = [layer for layer in layers
                           if layer >= spec.num_layers]
                    if bad:
                        raise ValueError(
                            f"KVRoute({route.src!r} → {route.dst!r}).{side} "
                            f"contains {bad}, out of range for stream "
                            f"{spec.name!r} with num_layers="
                            f"{spec.num_layers} (L2/V4)"
                        )

    def _check_condition_slots(self) -> None:
        """Reconcile the condition table against each stream's slots.

        Both directions are checked:

          - a condition's ``targets`` must be real stream names;
          - a CROSS_ATTENTION condition's name must appear in the target
            stream's ``cross_kv_slots``, and conversely every slot a stream
            declares must be filled by some condition.

        This is the reconciliation point created by separating "the Nth
        condition input" from "the Nth output head". The existing code aligns
        them by index, so adding one condition silently shifts the rest onto
        the wrong slots.

        Raises:
            ValueError: Duplicate condition names, a target naming a
                non-existent stream, a declared slot with no condition, or a
                condition injecting a slot the stream never declared.
        """
        known = set(self.stream_names)
        names = [spec.name for spec in self.conditions]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate ConditionSpec names (L2/V7): {duplicates}"
            )
        cross_by_stream: dict[str, set[str]] = {
            name: set() for name in known
        }
        for cond in self.conditions:
            missing = sorted(set(cond.targets) - known)
            if missing:
                raise ValueError(
                    f"ConditionSpec({cond.name!r}).targets names streams that "
                    f"do not exist: {missing}; this model has "
                    f"{sorted(known)} (L2/V7)"
                )
            if cond.kind is InjectionKind.CROSS_ATTENTION and cond.enabled:
                for target in cond.targets:
                    cross_by_stream[target].add(cond.name)
        for spec in self.streams:
            declared = set(spec.cross_kv_slots)
            filled = cross_by_stream[spec.name]
            unfilled = sorted(declared - filled)
            if unfilled:
                raise ValueError(
                    f"stream {spec.name!r} declares cross_kv_slots "
                    f"{unfilled}, but no CROSS_ATTENTION condition injects "
                    "them (L2/V7)"
                )
            unused = sorted(filled - declared)
            if unused:
                raise ValueError(
                    f"conditions {unused} inject stream {spec.name!r} as "
                    f"CROSS_ATTENTION, but its "
                    f"cross_kv_slots={list(spec.cross_kv_slots)} "
                    "does not contain those slots (L2/V7)"
                )

    def _check_appended_slots(self) -> None:
        """The four checks on ``APPEND_TO_SLOT``.

        This kind of condition creates no KV weights of its own; it rides on a
        host slot. So what has to be checked is whether the ride exists and
        whether there is room on it:

          1. the host slot must be a real condition that uses
             CROSS_ATTENTION. Appending onto a channel-concat condition is
             meaningless, and the symptom would be a dimension mismatch in a
             ``cat`` during forward;
          2. the two ``width`` values must be equal, since both go through the
             same condition projector;
          3. ``targets`` must be a subset of the host's ``targets``, otherwise
             some stream receives a span of context it has no projector for;
          4. if the host is disabled, this condition must be disabled too:
             with no host there is nowhere to append.

        Raises:
            ValueError: Any of the four checks fails.
        """
        by_name = {cond.name: cond for cond in self.conditions}
        for cond in self.conditions:
            if cond.kind is not InjectionKind.APPEND_TO_SLOT:
                continue
            host = by_name.get(str(cond.append_to))
            if host is None:
                raise ValueError(
                    f"condition {cond.name!r} appends to slot "
                    f"{cond.append_to!r}, but this model has no such "
                    f"condition; it has {sorted(by_name)} (L2/V7)"
                )
            if host.kind is not InjectionKind.CROSS_ATTENTION:
                raise ValueError(
                    f"condition {cond.name!r} appends to {host.name!r}, whose "
                    f"kind={host.kind.value!r}; only a CROSS_ATTENTION slot "
                    "has a tail to append to (L2/V7)"
                )
            if cond.width != host.width:
                raise ValueError(
                    f"condition {cond.name!r} has width={cond.width} while "
                    f"host slot {host.name!r} has width={host.width} "
                    f"(L2/V7); "
                    "both go through the same condition projector, so "
                    "different widths cannot be concatenated"
                )
            outside = sorted(set(cond.targets) - set(host.targets))
            if outside:
                raise ValueError(
                    f"condition {cond.name!r} injects streams {outside}, but "
                    f"host slot {host.name!r} does not inject them (targets="
                    f"{list(host.targets)}); those streams have no projector "
                    "for that span of context (L2/V7)"
                )
            if cond.enabled and not host.enabled:
                raise ValueError(
                    f"condition {cond.name!r} is enabled, but its host slot "
                    f"{host.name!r} is disabled; the append has nowhere to go "
                    "(L2/V7)"
                )

    def _validate_diffusion_layer(self) -> None:
        """L3: timestep wiring and loss sources.

        V5: the timestep configuration matches each stream's
        ``needs_timestep`` one-to-one. It replaces the existing behaviour of
        treating a missing timestep in a stream as 0, which turns one
        forgotten config line into silent noise-free training that looks fine
        numerically.

        V6: every loss term's source (a stream or a head) exists.

        Raises:
            ValueError: V5 fails inside ``DiffusionSpec.validate_against``, or
                a ``LossSpec.head`` names a head this model does not have.
        """
        self.diffusion.validate_against(self.streams)
        head_names = {spec.name for spec in self.heads}
        for loss in self.diffusion.losses:
            if loss.head is not None and loss.head not in head_names:
                raise ValueError(
                    f"LossSpec({loss.name!r}).head={loss.head!r} is not a "
                    f"head of this model; its heads are "
                    f"{sorted(head_names)} (L3/V6)"
                )

    def _validate_head_layer(self) -> None:
        """L4: head names are unique and every head's source is reachable.

        Three checks: unique head names, the source stream exists, and when a
        head reads an intermediate layer, that joint layer id is one the
        stream actually participates in.

        The last one is the easiest to get wrong today: intermediate layers
        are taken with a hook, the layer id is written at the call site, and
        whether the stream participates in that layer can only be known by
        reading ``layer_mapping``. Once this passes, the requests derived by
        ``captures()`` are necessarily satisfiable.

        Raises:
            ValueError: Duplicate head names, an unknown ``source_stream``, or
                a capture layer the stream does not participate in.
        """
        names = [spec.name for spec in self.heads]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate output head names (L4/V8): {duplicates}"
            )
        known = set(self.stream_names)
        for spec in self.heads:
            if spec.source_stream not in known:
                raise ValueError(
                    f"HeadSpec({spec.name!r}).source points at stream "
                    f"{spec.source_stream!r}, which is not a stream of this "
                    f"model; its streams are {sorted(known)} (L4/V8)"
                )
            key = spec.capture_key
            if key is None:
                continue
            stream_name, layer_id = key
            group = self.group_of(stream_name)
            index_map = group.layer_index_maps()[stream_name]
            if layer_id not in index_map:
                raise ValueError(
                    f"HeadSpec({spec.name!r}) captures joint layer "
                    f"{layer_id} of stream {stream_name!r}, but that stream "
                    f"does not take part in the joint attention at that "
                    f"layer; capturable layers are {sorted(index_map)} "
                    f"(L4/V8)"
                )

    def _validate_visibility_layer(self) -> None:
        """L1: the backend is known, and ``flex`` carries a block size.

        The full form of this check is the three-way negotiation
        ``CompiledMask.supports(vis, scenario)``, run once for training and
        once for inference. That belongs to Lane A and ships with S4. Until
        then it degrades to the two single-field checks in this method. What
        degrades is the **strictness**, not the timing: the check still happens
        at construction time rather than raising once inference starts.

        Raises:
            ValueError: ``attention_backend`` is ``"auto"`` or unknown,
                ``"flex"`` without a positive ``attention_block_size``, or a
                block size given for a non-flex backend.
        """
        if self.attention_backend == "auto":
            raise ValueError(
                f"model {self.model_type!r}: attention_backend does not "
                "accept 'auto'; it must already be resolved to a concrete "
                "backend at build time (L1/V9)"
            )
        if self.attention_backend not in _KNOWN_BACKENDS:
            raise ValueError(
                f"unknown attention backend {self.attention_backend!r}; "
                f"known ones are {sorted(_KNOWN_BACKENDS)} (L1/V9)"
            )
        if self.attention_backend == "flex":
            if self.attention_block_size is None \
                    or self.attention_block_size <= 0:
                raise ValueError(
                    "attention_backend='flex' requires a positive "
                    f"attention_block_size (L1/V9), got "
                    f"{self.attention_block_size}. "
                    "It must equal the Scenario.block_size used during "
                    "negotiation: a different value is a different mask, and "
                    "the BlockMask gets captured by torch.compile"
                )
        elif self.attention_block_size is not None:
            raise ValueError(
                f"attention_block_size is only meaningful when "
                f"backend='flex'; current "
                f"backend={self.attention_backend!r} (L1/V9)"
            )


_SECTION_KEYS: tuple[str, ...] = (
    "geometry", "streams", "diffusion", "heads",
    "attention_backend", "attention_block_size",
)
"""The config fields ``build_world_model`` reads, besides ``model_type``.

Every arch model shares these; nothing here names a model or a topology (rule
R5). A per-model config subclass adds only *provider keyword arguments* on top,
which is what ``provider_kwargs()`` returns.
"""

_HEAD_KNOBS: frozenset[str] = frozenset({"weight", "warmup_steps", "enabled"})
"""Keys a YAML ``heads.<name>:`` block may carry.

All three land on the ``LossSpec`` that consumes the head, never on the
``HeadSpec``: weight ownership is defined in exactly one place, and a head that
also carried a weight would let two numbers disagree silently (``WorldModel``
has a matching check that a weight has a single owner).
"""


def _plain(value: Any) -> dict[str, Any]:
    """Shallow-convert a config section to a plain ``dict``.

    OmegaConf hands back ``DictConfig`` objects, which are ``MutableMapping``
    but are not ``dict`` and resolve interpolations lazily. One level of
    ``dict()`` per nesting level keeps the spec layer free of an OmegaConf
    import while still accepting both plain dicts and ``DictConfig``.

    Args:
        value: A mapping section, or ``None`` / an empty mapping.

    Returns:
        A plain ``dict``; empty when ``value`` is falsy.

    Raises:
        TypeError: ``value`` is neither falsy nor a mapping.
    """
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"a config section must be a mapping, got "
            f"{type(value).__name__}; geometry / streams / diffusion / heads "
            "are all key: value structures"
        )
    return dict(value)


def _flatten_per_stream(section: str, body: dict[str, Any],
                        into: dict[str, Any]) -> None:
    """Flatten a ``<stream>: {<key>: value}`` section into provider kwargs.

    The provider signature is flat (``video_num_layers=``, ``action_shift=``),
    while the YAML is nested per stream. Flattening here rather than giving the
    providers a nested parameter keeps one invariant worth having: a typo in a
    YAML key becomes an unexpected keyword argument at the provider call, which
    names both the model and the key.

    Args:
        section: Section name, used only in the collision message.
        body: The already-plain section, ``{stream_name: {key: value}}``.
        into: Accumulator for the provider keyword arguments; mutated in place.

    Raises:
        TypeError: A stream's body is not a mapping.
        ValueError: Two sections produce the same keyword argument. That can
            only happen when the same ``<stream>_<key>`` is written twice, and
            silently keeping one of them would make half the YAML inert.
    """
    for stream_name, knobs in body.items():
        if not isinstance(knobs, Mapping):
            raise TypeError(
                f"{section}.{stream_name} must be a mapping, "
                f"got {type(knobs).__name__}"
            )
        for key, value in knobs.items():
            name = f"{stream_name}_{key}"
            if name in into:
                raise ValueError(
                    f"the config gives {name!r} two sources "
                    f"({section}.{stream_name}.{key} and another section); "
                    "a provider can only take one value"
                )
            into[name] = value


def _reweight(loss: LossSpec, knobs: Mapping[str, Any]) -> LossSpec:
    """Apply one ``heads.<name>:`` block to the ``LossSpec`` that consumes it.

    ``weight`` and ``warmup_steps`` are two views of the same number, so they
    are resolved together: with a warmup the effective weight lives in
    ``WeightRamp.end_value`` and ``LossSpec.weight`` is ignored by
    ``weight_at``, which is why writing only ``weight`` onto a ramped loss has
    to move ``end_value`` rather than ``weight``.

    Args:
        loss: The term declared by the provider.
        knobs: The YAML block, already plain.

    Returns:
        A new ``LossSpec``; ``loss`` itself when the block is empty.

    Raises:
        ValueError: The block carries a key outside ``_HEAD_KNOBS``.
    """
    stray = sorted(set(knobs) - _HEAD_KNOBS)
    if stray:
        raise ValueError(
            f"heads.{loss.head} got unknown keys {stray}; "
            f"the writable ones are {sorted(_HEAD_KNOBS)}"
        )
    ramp = loss.ramp
    weight = float(knobs.get("weight",
                             ramp.end_value if ramp else loss.weight))
    warmup = int(knobs.get("warmup_steps", ramp.end_step if ramp else 0))
    if warmup > 0:
        ramp = WeightRamp(start_step=ramp.start_step if ramp else 0,
                          end_step=warmup,
                          start_value=ramp.start_value if ramp else 0.0,
                          end_value=weight)
    else:
        ramp = None
    out = replace(loss, weight=weight, ramp=ramp)
    if "enabled" in knobs:
        out = replace(out, enabled=bool(knobs["enabled"]))
    return out


def _apply_heads(spec: WorldModelSpec,
                 heads: dict[str, Any]) -> WorldModelSpec:
    """Rewrite the loss table from the ``heads:`` section.

    This one override is safe *after* the provider has run, unlike the
    geometry: no provider derives anything from a loss weight, so replacing the
    table cannot leave a stale derived value behind.

    Args:
        spec: The provider's output.
        heads: ``{head_name: {knob: value}}``, already plain.

    Returns:
        ``spec`` when the section is empty, otherwise a copy whose
        ``diffusion.losses`` carries the new weights.

    Raises:
        ValueError: A head name matches no loss term. The message lists the
            terms that do exist, because the usual cause is naming the head
            (``video_flow``) where the loss happens to use another name.
    """
    if not heads:
        return spec
    known = {loss.head: loss.name for loss in spec.diffusion.losses
             if loss.head is not None}
    stray = sorted(set(heads) - set(known))
    if stray:
        raise ValueError(
            f"{stray} in the heads section is consumed by no loss term; "
            f"the current head to loss term mapping is {known}"
        )
    losses = tuple(
        _reweight(loss, _plain(heads[loss.head]))
        if loss.head in heads else loss
        for loss in spec.diffusion.losses
    )
    return replace(spec, diffusion=replace(spec.diffusion, losses=losses))


def _apply_backend(spec: WorldModelSpec, backend: str | None,
                   block_size: int | None) -> WorldModelSpec:
    """Swap the attention backend, keeping ``attention_block_size`` consistent.

    The two fields are coupled by the L1 check: a block size is required
    exactly when the backend is ``"flex"`` and rejected otherwise. Overriding
    the backend alone would therefore fail in both directions, so the block
    size is carried along here instead of being a second, independently
    overridable knob.

    Args:
        spec: The provider's output.
        backend: Target backend, or ``None`` to keep the provider's choice.
        block_size: Block size for ``"flex"``. ``None`` reuses the provider's
            value, which is what a YAML that only names the backend wants.

    Returns:
        ``spec`` unchanged when nothing is overridden.

    Raises:
        ValueError: Switching to ``"flex"`` with no block size available from
            either the YAML or the provider.
    """
    if backend is None and block_size is None:
        return spec
    backend = backend or spec.attention_backend
    if backend == "flex":
        block_size = block_size or spec.attention_block_size
        if not block_size:
            raise ValueError(
                "attention_backend='flex' needs an attention_block_size, "
                "and neither the config nor the provider supplied one (L1/V9)"
            )
    else:
        # A non-flex backend carrying a block size is rejected outright by the
        # L1 check. Clear it here instead of raising, because "switch from
        # flex to dense" should take one config line.
        block_size = None
    return replace(spec, attention_backend=backend,
                   attention_block_size=block_size)


def build_world_model(config: Any) -> WorldModelSpec:
    """Turn one model config into a ``WorldModelSpec``.

    The four steps are ordered and the order is load-bearing:

    1. Look the provider up by ``config.model_type``. Model identity has a
       single source — the ``--model-name`` launch argument, which selects the
       config whose ``model_type`` is this key. No YAML line says "this is
       VPP2" beyond that one field.
    2. Translate the config sections into **keyword arguments**: ``geometry:``
       stays a mapping, ``streams:`` and ``diffusion:`` are flattened to
       ``<stream>_<key>``, and the per-model subclass fields are passed by
       name. The config object itself never reaches a provider (rule R5): a
       provider that could read it would be free to pull topology out of YAML.
    3. Call the provider. Everything structural — stream count, visibility,
       condition table, loss terms — is Python code there, so steps 1–2 can
       only move numbers, never shapes of the graph.
    4. Apply the two overrides that are safe afterwards: loss weights and the
       attention backend. Geometry is deliberately *not* here; it has to reach
       the provider as an argument because visibility and the flow heads are
       derived from it (see ``_resolve_geometry``).

    Args:
        config: A model config instance. Required: ``model_type``. Optional:
            the six names in ``_SECTION_KEYS`` plus a ``provider_kwargs()``
            method returning the per-model extras (absent ⇒ no extras).

    Returns:
        A validated ``WorldModelSpec``. The four layer checks run inside
        ``__post_init__``, both for this path and for a hand-written spec in a
        test, so no check can be bypassed by constructing one directly.

    Raises:
        ValueError: No ``model_type``, or a section is malformed.
        KeyError: ``model_type`` is not a registered provider.
        TypeError: The provider returned something other than a
            ``WorldModelSpec``, or a section has the wrong Python type.
    """
    model_type = getattr(config, "model_type", None)
    if not model_type:
        raise ValueError(
            "the config has no model_type; it is the provider registry key. "
            "Model identity has a single source, the --model-name launch "
            "argument"
        )
    provider = get_provider(model_type)

    kwargs: dict[str, Any] = {}
    geometry = _plain(getattr(config, "geometry", None))
    if geometry:
        kwargs["geometry"] = geometry
    _flatten_per_stream("streams", _plain(getattr(config, "streams", None)),
                        kwargs)
    _flatten_per_stream("diffusion",
                        _plain(getattr(config, "diffusion", None)), kwargs)
    extras = getattr(config, "provider_kwargs", None)
    if callable(extras):
        kwargs.update(extras())

    spec = provider(**kwargs)
    if not isinstance(spec, WorldModelSpec):
        raise TypeError(
            f"provider(model_type={model_type!r}) returned "
            f"{type(spec).__name__}, it must return a WorldModelSpec"
        )
    spec = _apply_heads(spec, _plain(getattr(config, "heads", None)))
    return _apply_backend(spec, getattr(config, "attention_backend", None),
                          getattr(config, "attention_block_size", None))






