# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Reader-centric, closed-world intra/read declarations and their shared compiler."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from ..layout import TokenLayout
from .combine import Visibility

__all__ = ["ReadGate", "ReadPolicy", "ReaderSpec", "VisibilityTopology"]


@dataclass(frozen=True)
class ReadGate:
    """A runtime label comparison scoped to one declared read rectangle."""

    label: str
    side: str = "query"
    op: str = "ge"
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("ReadGate.label must be a non-empty string")
        if self.side not in ("query", "key"):
            raise ValueError("ReadGate.side must be query or key")
        if self.op not in ("ge", "gt", "le", "lt", "eq"):
            raise ValueError("ReadGate.op must be ge, gt, le, lt or eq")
        if not isinstance(self.threshold, Real) or isinstance(self.threshold, bool):
            raise ValueError("ReadGate.threshold must be numeric")


@dataclass(frozen=True)
class ReadPolicy:
    """Temporal policy, optionally combined with an explicit runtime gate."""

    mode: str
    frames: int | None = None
    window: int | None = None
    group_size: int | None = None
    gate: ReadGate | None = None

    def __post_init__(self) -> None:
        modes = (
            "bidirectional", "prefix_frames", "frame_causal",
            "first_frame_causal", "chunk_causal", "group_causal", "group_diagonal",
        )
        if self.mode not in modes:
            raise ValueError(f"Unknown read policy mode {self.mode!r}")
        allowed = {
            "frames": self.mode == "prefix_frames",
            "window": self.mode in ("frame_causal", "chunk_causal"),
            "group_size": self.mode in ("group_causal", "group_diagonal"),
        }
        for name, supported in allowed.items():
            value = getattr(self, name)
            required = supported and name != "window"
            if value is None:
                if required:
                    raise ValueError(f"{self.mode} requires {name}")
            elif not supported:
                raise ValueError(f"{self.mode} does not support {name}")
            elif not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"ReadPolicy.{name} must be a positive integer")
        if self.gate is not None and not isinstance(self.gate, ReadGate):
            raise TypeError("ReadPolicy.gate must be a ReadGate")

    @classmethod
    def parse(cls, value: Any) -> ReadPolicy:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, Mapping):
            raise TypeError("read policy must be a string, mapping or ReadPolicy")
        unknown = set(value) - {"mode", "frames", "window", "group_size", "gate"}
        if unknown or "mode" not in value:
            raise ValueError(f"Invalid read policy fields: {sorted(unknown)}; mode is required")
        fields = dict(value)
        gate = fields.get("gate")
        if isinstance(gate, Mapping):
            fields["gate"] = ReadGate(**gate)
        return cls(**fields)


@dataclass(frozen=True)
class ReaderSpec:
    """Either an explicit leaf reader or a stream's query-region partition."""

    intra: ReadPolicy | None = None
    reads: Mapping[str, ReadPolicy] = field(default_factory=dict)
    regions: Mapping[str, ReaderSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.regions:
            if self.intra is not None or self.reads:
                raise ValueError("regions cannot inherit intra or reads")
            children = {name: ReaderSpec.parse(spec) for name, spec in self.regions.items()}
            if any(child.regions for child in children.values()):
                raise ValueError("Nested query regions are not supported")
            object.__setattr__(self, "regions", MappingProxyType(children))
        else:
            if self.intra is None:
                raise ValueError("A reader requires explicit intra")
            object.__setattr__(self, "intra", ReadPolicy.parse(self.intra))
            object.__setattr__(self, "regions", MappingProxyType({}))
        normalized = {}
        for name, policy in self.reads.items():
            _validate_name(name)
            normalized[name] = ReadPolicy.parse(policy)
        for name in self.regions:
            _validate_name(name)
        object.__setattr__(self, "reads", MappingProxyType(normalized))

    @classmethod
    def parse(cls, value: Any) -> ReaderSpec:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("reader specification must be a mapping or ReaderSpec")
        if set(value) - {"intra", "reads", "regions"}:
            raise ValueError("Unknown reader specification fields")
        if "regions" in value:
            if set(value) != {"regions"} or not value["regions"]:
                raise ValueError("A region reader must declare only non-empty regions")
        elif "intra" not in value:
            raise ValueError("A reader requires explicit intra")
        return cls(**value)


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Declaration names must be non-empty strings")


class VisibilityTopology(Mapping[str, Visibility]):
    """Readable declarations with legacy query-row adapters and one compiler.

    ``declarations[name]`` is a ReaderSpec. Mapping access returns a Visibility
    selecting that stream's query rows; new consumers use ``compile(layout)``.
    An empty topology deliberately means unrestricted attention.
    """

    def __init__(self, declarations: Mapping[str, Any]) -> None:
        parsed = {}
        for name, value in declarations.items():
            _validate_name(name)
            parsed[name] = ReaderSpec.parse(value)
        self.declarations = MappingProxyType(parsed)

    def __iter__(self) -> Iterator[str]:
        return iter(self.declarations)

    def __len__(self) -> int:
        return len(self.declarations)

    def __getitem__(self, name: str) -> Visibility:
        self.declarations[name]
        return _DeclarativeVisibility(self, name)

    def validate_streams(self, stream_names: Sequence[str]) -> None:
        """Validate stream coverage at spec construction, without a layout."""
        if self and set(self.declarations) != set(stream_names):
            raise ValueError(
                "A nonempty topology must declare every stream exactly"
            )

    def as_visibility(self, *, reader: str | None = None) -> Visibility | None:
        if reader is not None:
            self.declarations[reader]
        return _DeclarativeVisibility(self, reader) if self else None

    def compile(self, layout: TokenLayout, *, key_valid: Tensor | None = None):
        from .mask import AttentionPlan

        return AttentionPlan.from_visibility(self.as_visibility(), layout, key_valid=key_valid)

    def _rules(self, layout: TokenLayout):
        """Resolve scopes and validate rectangles before applying any policies."""
        self.validate_streams(layout.stream_names)
        device = (next(iter(layout.labels.values())).device if layout.labels else
                  next(iter(layout.regions.values())).device if layout.regions else None)
        positions = torch.arange(layout.total, device=device)
        streams = {
            seg.stream: (positions >= seg.start) & (positions < seg.stop)
            for seg in layout.segments
        }
        sources = {**streams, **{
            name: members.to(device) for name, members in layout.regions.items()
        }}
        leaves = []
        claimed_regions = set()
        for owner, spec in self.declarations.items():
            if not spec.regions:
                leaves.append((owner, owner, streams[owner], spec))
                continue
            covered = torch.zeros(layout.total, dtype=torch.bool, device=device)
            for name, child in spec.regions.items():
                if name not in layout.regions:
                    raise ValueError(f"Query region {name!r} is missing from layout")
                members = sources[name]
                if (members & ~streams[owner]).any():
                    raise ValueError(f"Query region {name!r} contains tokens outside stream {owner!r}")
                if name in claimed_regions or (covered & members).any():
                    raise ValueError("Query regions must partition their stream without overlap")
                claimed_regions.add(name)
                covered |= members
                leaves.append((owner, name, members, child))
            if not torch.equal(covered, streams[owner]):
                raise ValueError("Query regions must partition their stream without overlap")
        rules = []
        for owner, name, query, spec in leaves:
            used = query.clone()
            local = [(query, spec.intra)]
            for source, policy in spec.reads.items():
                if source not in sources:
                    raise ValueError(f"Unknown read source {source!r}")
                keys = sources[source]
                if query.any() and (used & keys).any():
                    raise ValueError(f"Overlapping read rules for query {name!r}")
                used |= keys
                local.append((keys, policy))
            for keys, policy in local:
                rules.append((owner, query, keys, policy))
        return rules

    def _evaluate(self, q, k, *, reader=None, gates=True) -> Tensor:
        layout = q.layout
        out = torch.zeros((layout.total, layout.total), dtype=torch.bool,
                          device=q.positions().device)
        for owner, query, keys, policy in self._rules(layout):
            # Validate every rule even for a single-stream compatibility adapter.
            relation = _temporal_relation(layout, query, keys, policy)
            if reader is not None and owner != reader:
                continue
            edge = query[:, None] & keys[None, :] & relation
            if gates and policy.gate is not None:
                gate = policy.gate
                view = q if gate.side == "query" else k
                if not view.has(gate.label):
                    raise ValueError(f"visibility requires runtime labels {[gate.label]}")
                edge = edge & getattr(torch, gate.op)(view.get(gate.label), gate.threshold)
            out = out | edge
        return out


def _temporal_relation(layout, query, keys, policy) -> Tensor:
    """Static policy support; only labels in the participating scopes matter."""
    device = query.device
    if policy.mode == "bidirectional":
        return torch.ones((layout.total, layout.total), dtype=torch.bool, device=device)
    label = "chunk" if policy.mode == "chunk_causal" else "frame"
    if not layout.has_label(label):
        raise ValueError(f"{policy.mode} requires layout label {label!r}")
    values = layout.labels_of(label).to(device)
    if values.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise ValueError(
            f"{policy.mode} requires integer dtype for layout label {label!r}"
        )
    scope = keys if policy.mode == "prefix_frames" else query | keys
    if not torch.isfinite(values[scope]).all():
        raise ValueError(
            f"{policy.mode} requires finite {label!r} labels in its scope"
        )
    if (values[scope] < 0).any():
        raise ValueError(f"{policy.mode} requires non-negative {label!r} labels in its scope")
    if not query.any() or not keys.any():
        return torch.zeros((layout.total, layout.total), dtype=torch.bool, device=device)
    qvalues, kvalues = values[:, None], values[None, :]
    if policy.mode == "prefix_frames":
        selected = torch.unique(values[keys], sorted=True)[:policy.frames]
        return torch.isin(values, selected)[None, :].expand(layout.total, -1)
    if policy.mode == "first_frame_causal":
        first = values[keys].min()
        return (qvalues != first) | (kvalues == first)
    if policy.mode in ("group_causal", "group_diagonal"):
        qvalues = torch.div(qvalues, policy.group_size, rounding_mode="floor")
        kvalues = torch.div(kvalues, policy.group_size, rounding_mode="floor")
        return qvalues == kvalues if policy.mode == "group_diagonal" else qvalues >= kvalues
    relation = qvalues >= kvalues
    if policy.window is not None:
        relation &= (qvalues - kvalues) < policy.window
    return relation


class _DeclarativeVisibility(Visibility):
    """Compatibility expression retaining an ungated structural evaluation."""

    def __init__(self, topology: VisibilityTopology, reader: str | None) -> None:
        labels = set()
        for owner, spec in topology.declarations.items():
            if reader is not None and owner != reader:
                continue
            for leaf in spec.regions.values() if spec.regions else (spec,):
                for policy in (leaf.intra, *leaf.reads.values()):
                    if policy.gate is not None:
                        labels.add(policy.gate.label)
        super().__init__(
            lambda q, k: topology._evaluate(q, k, reader=reader),
            f"intra+reads({reader or 'all'})",
            frozenset(labels),
        )
        object.__setattr__(self, "_topology", topology)
        object.__setattr__(self, "_reader", reader)

    def evaluate_structural(self, layout: TokenLayout) -> Tensor:
        q, k = layout.label_views()
        return self._topology._evaluate(q, k, reader=self._reader, gates=False)
