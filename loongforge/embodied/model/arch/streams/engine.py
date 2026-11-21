# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 joint attention engine: ``AttentionGroup`` / ``EngineOutput`` /
``StreamEngine``.

Signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.6.

Two hard conventions carry the properties of all of L2:

1. The engine is not an ``nn.Module``. It holds no parameters and never
   enters state_dict (INV-3). All parameters live in the stream modules. If
   the engine read parameters, the FSDP2 / ZeRO-3 all-gather would be
   triggered at the engine level, and the engine is not FSDP-wrapped.
2. A single stream is the N=1 degenerate case, not a special case. The
   engine must contain no ``if len(streams) == 1``. The degeneracy shows up
   only in two pieces of data: ``AttentionPlan.structural is None`` and
   ``routes == ()``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint

from ..layout import SegmentRange, TokenLayout
from ..visibility.mask import AttentionPlan, LayerAttentionOverride
from .conditions import ConditionBus
from .routing import KVRoute
from .spec import StreamSpec
from .state import QKV, StreamState

__all__ = [
    "AttentionGroup", "EngineOutput", "StreamEngine", "JointAttentionEngine",
    "assert_shape_signature",
]


@dataclass(frozen=True)
class AttentionGroup:
    """A set of streams sharing one joint attention, plus the layer count.

    Attributes:
        streams: Specs of the streams in this group, in token order.
        num_layers: Number of joint layers. It is unrelated to any single
            stream's ``num_layers``; this independent number is exactly what
            the relaxation of "all three num_layers equal" requires. Each
            stream lines up through
            ``StreamSpec.resolve_layer_index_map(num_layers)``.
        routes: Cross-stream KV routes inside this group.
    """

    streams: tuple[StreamSpec, ...]
    num_layers: int
    routes: tuple[KVRoute, ...] = ()

    def __post_init__(self) -> None:
        if not self.streams:
            raise ValueError("AttentionGroup.streams must not be empty")
        if self.num_layers <= 0:
            raise ValueError(
                f"AttentionGroup.num_layers must be positive, got "
                f"{self.num_layers}"
            )
        names = [spec.name for spec in self.streams]
        if len(set(names)) != len(names):
            raise ValueError(
                f"AttentionGroup.streams has duplicate stream names: {names}"
            )
        # V3: joint streams must share head_dim. num_heads and
        # residual_width may differ. This is the only width constraint left
        # after relaxing the "all three num_layers equal" rule.
        head_dims = {spec.name: spec.head_dim for spec in self.streams}
        if len(set(head_dims.values())) != 1:
            raise ValueError(
                f"AttentionGroup streams must share head_dim (L2/V3), got "
                f"{head_dims}"
            )
        known = set(names)
        for route in self.routes:
            missing = {route.src, route.dst} - known
            if missing:
                raise ValueError(
                    f"KVRoute({route.src!r} -> {route.dst!r}) points at "
                    f"streams outside this group: {sorted(missing)}; group "
                    f"streams are {names}"
                )
            if max(route.dst_layers) >= self.spec_of(route.dst).num_layers:
                raise ValueError(
                    f"KVRoute({route.src!r} -> {route.dst!r}).dst_layers max "
                    f"layer index {max(route.dst_layers)} exceeds stream "
                    f"{route.dst!r} num_layers="
                    f"{self.spec_of(route.dst).num_layers}. Route layer "
                    "indices are **local** layer indices, not joint ones"
                )

    @property
    def head_dim(self) -> int:
        """Shared head_dim of the group.

        Returns:
            The first stream's ``head_dim``; ``__post_init__`` has already
            checked that all streams agree.
        """
        return self.streams[0].head_dim

    @property
    def stream_names(self) -> tuple[str, ...]:
        """Stream names, in stream order."""
        return tuple(spec.name for spec in self.streams)

    def spec_of(self, name: str) -> StreamSpec:
        """Look up one stream's spec.

        Args:
            name: Stream name.

        Returns:
            That stream's ``StreamSpec``.

        Raises:
            KeyError: If the stream is not a member of this group.
        """
        for spec in self.streams:
            if spec.name == name:
                return spec
        raise KeyError(
            f"AttentionGroup has no stream {name!r}; streams are "
            f"{list(self.stream_names)}"
        )

    def layer_index_maps(self) -> dict[str, dict[int, int]]:
        """Build every stream's joint-to-local layer index map.

        Returns:
            Stream name to {joint layer index: local layer index}. Computed
            once at construction time; the forward pass only reads it.
        """
        return {spec.name: spec.resolve_layer_index_map(self.num_layers)
                for spec in self.streams}

    def routes_into(self, dst: str) -> tuple[KVRoute, ...]:
        """Collect the routes ending at one stream.

        Args:
            dst: Destination stream name.

        Returns:
            Matching routes. An empty tuple is normal and is always the case
            for a single-stream model.
        """
        return tuple(route for route in self.routes if route.dst == dst)


@dataclass
class EngineOutput:
    """What the engine produces after running one group.

    This is the carrier of interface **I3** (Lane B to Lane C).

    Attributes:
        states: Stream name to final ``StreamState``.
        captured: Keyed by ``(stream name, joint layer index)``, holding
            intermediate hidden states for L3/L4 to read through
            ``HiddenStateRef``. The request side is
            ``captures: Mapping[str, tuple[int, ...]]``, i.e. which joint
            layers each stream should capture. The layer index is part of the
            key because the current code takes intermediate layers through a
            hook plus a global variable, which scatters layer indices across
            call sites. An explicit key makes "which layer was captured"
            visible in the type, and lets validator V8 check at construction
            time that the requested layer exists.
    """

    states: dict[str, StreamState]
    captured: dict[tuple[str, int], Tensor] = field(default_factory=dict)

    def require(self, stream: str) -> StreamState:
        """Fetch one stream's final state.

        Args:
            stream: Stream name.

        Returns:
            That stream's ``StreamState``.

        Raises:
            KeyError: If absent; the message lists the available streams.
        """
        try:
            return self.states[stream]
        except KeyError:
            raise KeyError(
                f"EngineOutput has no state for stream {stream!r}; "
                f"available: {sorted(self.states)}"
            ) from None

    def require_capture(self, stream: str, layer_id: int) -> Tensor:
        """Fetch one capture point.

        Args:
            stream: Stream name.
            layer_id: Joint layer index.

        Returns:
            The captured hidden state tensor.

        Raises:
            KeyError: If absent; the message lists every captured key.
        """
        key = (stream, layer_id)
        try:
            return self.captured[key]
        except KeyError:
            raise KeyError(
                f"EngineOutput has no capture point {key}; "
                f"captured: {sorted(self.captured)}"
            ) from None


class StreamEngine(Protocol):
    """Protocol of the joint attention engine. ``run`` is all of L2's entry.

    Call order (interfaces §7; these three steps must not be reordered):
      1. ``prepare`` on every stream, taking each stream's raw input tensor
         (patchify happens inside this step);
      2. per joint layer: ``pre_attn`` on every participating stream, then
         exactly one joint attention, then ``post_attn`` on each stream;
      3. ``finalize`` on every stream.
    "All streams compute QKV before attention runs" is the part of step 2
    that cannot be reordered.
    """

    def run(self, group: AttentionGroup,
            inputs: Mapping[str, Tensor],
            modules: Mapping[str, Any],
            plan: AttentionPlan,
            layout: TokenLayout,
            *,
            conditions: ConditionBus,
            timesteps: Mapping[str, Tensor | None],
            captures: Mapping[str, Sequence[int]] | None = None,
            override: LayerAttentionOverride | None = None) -> EngineOutput:
        """Run every joint layer of one attention group.

        Args:
            group: Streams and joint layer count.
            inputs: Stream name to its untokenized raw tensor (video
                ``[B, C, F, H, W]``, actions ``[B, T, D]``). Tokenization is
                done by each stream's ``prepare``; the engine recognizes no
                modality's raw shape.
            modules: Stream name to ``StreamModule``.
            plan: Visibility plan for the joint sequence.
            layout: Token layout whose segment order equals the member order.
            conditions: The whole condition bus, passed to every stream.
            timesteps: Stream name to timestep; None for streams with
                ``needs_timestep=False``.
            captures: Stream name to the joint layers to capture.
            override: When not None, the layers it takes over skip the
                concat-plus-single-attention path, which also makes them
                ineligible for CUDA graph capture.

        Returns:
            An ``EngineOutput`` with the final states and the captures.
        """
        ...


def _assert_prepared_against_layout(live: Mapping[str, StreamState],
                                    layout: TokenLayout) -> None:
    """Check each prepared token count against the layout segment length.

    This check replaces the old "token count unchanged across prepare" rule:
    ``prepare`` now takes a raw tensor and is supposed to change the token
    count (patchify turns ``[B, C, F, H, W]`` into ``[B, L, W]``), so the
    layout is the only thing left to reconcile against. The segment lengths
    ultimately come from ``GeometrySpec.padded_seq_len``, the single source
    of static shapes (INV-2). A failure here means geometry and patchify
    disagree; continuing would raise a shape error at the concat site, far
    from the cause.

    Args:
        live: Stream name to the state returned by ``prepare``.
        layout: Token layout for the joint sequence.

    Raises:
        ValueError: If any stream's token count differs from its segment
            length.
    """
    for name, state in live.items():
        want = layout.range_of(name).length
        if state.num_tokens != want:
            raise ValueError(
                f"stream {name!r} prepare produced {state.num_tokens} "
                f"tokens, but the layout gives it a segment length of "
                f"{want}. Segment lengths come from "
                "GeometrySpec.padded_seq_len (the single source of static "
                "shapes), so a mismatch means geometry and patchify disagree"
            )


def assert_shape_signature(before: Mapping[str, StreamState],
                           after: Mapping[str, StreamState]) -> None:
    """Check C8 / C9 across one layer.

    C8: the set of streams must be identical before and after the layer.
    C9: each stream's token count must be unchanged.

    This lives at module level rather than inside the engine so that the
    Lane B implementation and its tests share one copy of the decision.

    Args:
        before: Stream name to state before the layer.
        after: Stream name to state after the layer.

    Raises:
        ValueError: If the stream set changed (C8) or any token count
            changed (C9).
    """
    if set(before) != set(after):
        raise ValueError(
            f"stream set changed across the layer: before {sorted(before)}, "
            f"after {sorted(after)} (C8)"
        )
    for name, state in before.items():
        got = after[name].num_tokens
        if got != state.num_tokens:
            raise ValueError(
                f"stream {name!r} token count went from {state.num_tokens} "
                f"to {got}; a layer must not change a stream's token count "
                "(C9)"
            )


def restrict_to_streams(layout: TokenLayout, plan: AttentionPlan,
                        names: tuple[str, ...]
                        ) -> tuple[TokenLayout, AttentionPlan]:
    """Restrict a layout and a plan to the streams active at one layer.

    Streams that do not participate are cut out whole, and the remaining
    segments are renumbered contiguously because ``TokenLayout`` requires
    the segments to cover ``[0, total)``. Both plan channels are
    index_select'ed with the same index table, so after the cut the rows and
    columns of ``structural``, the columns of ``key_valid``, and the
    frame/chunk/stream labels still refer to the same token.

    Args:
        layout: Full joint token layout.
        plan: Full visibility plan.
        names: Stream names to keep, in layout order.

    Returns:
        The restricted ``(layout, plan)`` pair. When
        ``names == layout.stream_names`` the inputs are returned unchanged;
        that is the normal case for the first two models (two equal-depth
        streams participating at every layer), which should not pay for an
        index_select.

    Raises:
        KeyError: If ``names`` contains a stream absent from the layout.
    """
    if names == layout.stream_names:
        return layout, plan
    stray = sorted(set(names) - set(layout.stream_names))
    if stray:
        raise KeyError(
            f"restrict_to_streams: the layout has no streams {stray}; "
            f"it has {list(layout.stream_names)}"
        )
    keep = [seg for seg in layout.segments if seg.stream in names]
    index = torch.cat([torch.arange(seg.start, seg.stop, dtype=torch.long)
                       for seg in keep])
    segments: list[SegmentRange] = []
    cursor = 0
    for seg in keep:
        segments.append(SegmentRange(stream=seg.stream, start=cursor,
                                     stop=cursor + seg.length))
        cursor += seg.length
    sub_layout = TokenLayout(
        segments=tuple(segments),
        labels={key: value.index_select(0, index.to(value.device))
                for key, value in layout.labels.items()},
        regions={name: membership.index_select(0, index.to(membership.device))
                 for name, membership in layout.regions.items()},
    )
    structural = plan.structural
    if structural is not None:
        idx = index.to(structural.device)
        structural = structural.index_select(0, idx).index_select(1, idx)
    key_valid = plan.key_valid
    if key_valid is not None:
        key_valid = key_valid.index_select(1, index.to(key_valid.device))
    if plan.visibility is not None:
        raise ValueError("bind runtime visibility before restricting streams")
    pairwise = plan.pairwise
    if pairwise is not None:
        idx = index.to(pairwise.device)
        pairwise = pairwise.index_select(1, idx).index_select(2, idx)
    return sub_layout, AttentionPlan(structural=structural,
                                     key_valid=key_valid, layout=sub_layout,
                                     pairwise=pairwise)


def _joint_attention(packed: QKV, mask: Any, *, backend: str) -> Tensor:
    """Run one joint attention call.

    Args:
        packed: Concatenated Q/K/V, each ``[B, L, num_heads, head_dim]``.
        mask: Backend-specific mask: a ``BlockMask`` for flex, a dense
            boolean tensor for dense/sdpa.
        backend: One of ``"flex"``, ``"dense"``, ``"sdpa"``.

    Returns:
        ``[B, L, num_heads, head_dim]``. ``QKV`` stores heads and head_dim
        on separate axes while both kernels want
        ``[B, num_heads, L, head_dim]``, so the transpose appears only in
        this function. A transpose scattered across call sites raises
        nothing when one is missed; it just treats heads as tokens.

    Raises:
        ValueError: If the backend is unknown. varlen needs the segment
            table from ``CompiledMask.as_segments`` wired up first, which is
            rule R3 (cosmos3).
    """
    q = packed.q.transpose(1, 2)
    k = packed.k.transpose(1, 2)
    v = packed.v.transpose(1, 2)
    if backend == "flex":
        from torch.nn.attention.flex_attention import flex_attention
        out = flex_attention(q, k, v, block_mask=mask)
    elif backend in ("dense", "sdpa"):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        raise ValueError(
            f"_joint_attention: backend {backend!r} is not implemented. "
            "varlen needs the segment table from CompiledMask.as_segments "
            "wired up first, which is rule R3 (cosmos3)"
        )
    return out.transpose(1, 2)


def _copy_state(state: StreamState, *, detach: bool = False) -> StreamState:
    """Copy state containers for replay, optionally cutting tensor graphs.

    Do not deepcopy tensors: trainable modulation and cached context must keep
    their connection to prepare/condition encoders. Stream methods must not
    modify tensor storage in place when checkpointing is enabled.
    """
    def copy(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.detach() if detach else value
        if isinstance(value, dict):
            return {key: copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [copy(item) for item in value]
        if isinstance(value, tuple):
            items = tuple(copy(item) for item in value)
            return type(value)(*items) if hasattr(value, "_fields") else items
        return value

    return StreamState(stream=state.stream, hidden=copy(state.hidden),
                       modulation=copy(state.modulation),
                       extras=copy(state.extras))


class JointAttentionEngine:
    """``StreamEngine`` implementation: concat, one joint attention, split.

    Not an ``nn.Module`` (convention 1 at the top of this module): it holds
    no parameters and never enters state_dict. It contains no
    ``if len(streams) == 1`` (convention 2): a single stream takes the same
    path, with one-item concat/split and ``plan.structural`` set to None.

    Masks are materialized once per ``run`` call and cached in a dict local
    to ``run``, keyed by the set of streams active at that layer. They are
    not cached across runs because ``plan.key_valid`` is per batch, so reuse
    would carry the previous batch's padding over, and that kind of defect
    is invisible in the loss. With the flex backend this means
    ``create_block_mask`` runs once per forward; hoisting it to construction
    time requires settling who owns key_valid first (master plan §10-5).

    Attributes:
        backend: ``"dense"``, ``"sdpa"`` or ``"flex"``.
        block_size: BlockMask block size, required by the flex backend.
        frozen_streams: Stream names executed without autograd. Their states,
            QKV and attention outputs are detached, not the condition bus.
        checkpoint_layers: Recompute each joint layer during backward.
    """

    def __init__(self, *, backend: str = "dense",
                 block_size: int | None = None,
                 frozen_streams: Iterable[str] = (),
                 checkpoint_layers: bool = False) -> None:
        """Configure the attention backend.

        Args:
            backend: ``"dense"``, ``"sdpa"`` or ``"flex"``.
            block_size: BlockMask block size. Required when
                ``backend == "flex"`` because it changes the BlockMask shape
                and the BlockMask is captured by ``torch.compile``.
            frozen_streams: Names whose prepare/pre_attn/post_attn/finalize
                run under no_grad, regardless of parameter requires_grad.
                Freezing is a stop-gradient execution policy, not a change to
                parameters or visibility. Other streams can attend their
                detached KV. The caller owns choosing a suitable topology.
            checkpoint_layers: Use non-reentrant checkpointing for each joint
                layer when autograd is enabled, preserving RNG and conditions.

        Raises:
            ValueError: If the backend is unsupported, or if flex is chosen
                without a ``block_size``.
        """
        if backend not in ("dense", "sdpa", "flex"):
            raise ValueError(
                f"JointAttentionEngine: backend {backend!r} is not "
                "supported; choose 'dense', 'sdpa' or 'flex'"
            )
        if backend == "flex" and block_size is None:
            raise ValueError(
                "JointAttentionEngine(backend='flex') requires block_size; "
                "it changes the BlockMask shape and the BlockMask is "
                "captured by torch.compile"
            )
        self.backend = backend
        self.block_size = block_size
        self.frozen_streams = frozenset(frozen_streams)
        self.checkpoint_layers = checkpoint_layers

    def _stream_context(self, name: str) -> Any:
        return torch.no_grad() if name in self.frozen_streams else nullcontext()

    def _stream_state(self, name: str, state: StreamState) -> StreamState:
        return (_copy_state(state, detach=True)
                if name in self.frozen_streams else state)

    def run(self, group: AttentionGroup,
            inputs: Mapping[str, Tensor],
            modules: Mapping[str, Any],
            plan: AttentionPlan,
            layout: TokenLayout,
            *,
            conditions: ConditionBus,
            timesteps: Mapping[str, Tensor | None],
            captures: Mapping[str, Sequence[int]] | None = None,
            override: LayerAttentionOverride | None = None) -> EngineOutput:
        """Run every joint layer of one attention group.

        See ``StreamEngine.run`` for the argument contract and the fixed
        three-step call order.

        Args:
            group: Streams and joint layer count.
            inputs: Stream name to its untokenized raw tensor.
            modules: Stream name to ``StreamModule``.
            plan: Visibility plan for the joint sequence.
            layout: Token layout; its segment order must equal the member
                order (C8).
            conditions: The whole condition bus.
            timesteps: Stream name to timestep, or None per stream.
            captures: Stream name to the joint layers to capture.
            override: Optional per-layer attention override.

        Returns:
            An ``EngineOutput`` with the finalized states and the captures.

        Raises:
            ValueError: If the input or module stream sets differ from the
                group streams, if the layout segment order differs, if a
                joint layer has no participating stream, or if an override
                returns the wrong stream set.
        """

        names = group.stream_names
        for label, got in (("inputs", set(inputs)), ("modules", set(modules))):
            if got != set(names):
                raise ValueError(
                    f"JointAttentionEngine.run: {label} stream set "
                    f"{sorted(got)} != group streams {list(names)}"
                )
        if layout.stream_names != names:
            raise ValueError(
                f"JointAttentionEngine.run: layout segment order "
                f"{list(layout.stream_names)} != group member order "
                f"{list(names)} (C8)"
            )
        index_maps = group.layer_index_maps()
        wanted = _resolve_captures(captures, index_maps)

        # Step 1: prepare every stream; each takes its raw tensor and
        # tokenizes it itself.
        # The whole condition bus goes to the stream module, which picks the
        # conditions it wants by name (§4.4): the engine interprets no
        # condition's semantics but must pass the bus down, otherwise
        # cross-attention KV and channel-concat conditions would have no
        # path into L2 at all.
        live = {}
        for name in names:
            with self._stream_context(name):
                state = modules[name].prepare(
                    inputs[name], timestep=timesteps.get(name),
                    conditions=conditions)
            live[name] = self._stream_state(name, state)
        _assert_prepared_against_layout(live, layout)

        # Step 2: per joint layer, pre_attn on all participating streams,
        # one joint attention, then post_attn on each stream.
        masks: dict[tuple[str, ...], tuple[TokenLayout, Any]] = {}
        captured: dict[tuple[str, int], Tensor] = {}
        for layer_id in range(group.num_layers):
            active = tuple(name for name in names
                           if layer_id in index_maps[name])
            if not active:
                raise ValueError(
                    f"joint layer {layer_id} has no participating stream, "
                    f"which makes it a dead layer. Participating layers per "
                    f"stream: "
                    f"{ {n: sorted(m) for n, m in index_maps.items()} }"
                )
            # Bind IDs now, not through loop-variable closures that would
            # replay every checkpoint using the final layer's IDs/state.
            run_layer = partial(
                self._run_layer, active=active,
                local_ids=tuple(index_maps[name][layer_id] for name in active),
                layer_id=layer_id, modules=modules, plan=plan, layout=layout,
                masks=masks, override=override)
            before = dict(live)
            states = tuple(live[name] for name in active)
            if self.checkpoint_layers and torch.is_grad_enabled():
                # Expose tensors in state dataclasses as checkpoint inputs:
                # checkpoint needs their devices to save accelerator RNG
                # state (a dataclass itself is not a registered pytree).
                values, structure = tree_flatten(tuple(
                    (state.stream, state.hidden, state.modulation, state.extras)
                    for state in states))
                after = checkpoint(
                    partial(self._checkpoint_layer, run_layer=run_layer,
                            structure=structure),
                    *values, use_reentrant=False)
            else:
                after = run_layer(*states)
            live.update(zip(active, after))
            assert_shape_signature(before, live)
            for name in active:
                if layer_id in wanted.get(name, ()):
                    captured[(name, layer_id)] = live[name].hidden

        # Step 3: finalize every stream
        final = {}
        for name in names:
            with self._stream_context(name):
                state = modules[name].finalize(live[name])
            final[name] = self._stream_state(name, state)
        assert_shape_signature(live, final)
        return EngineOutput(states=final, captured=captured)

    @staticmethod
    def _checkpoint_layer(*values: Any, run_layer: Any,
                          structure: Any) -> tuple[StreamState, ...]:
        states = tuple(StreamState(stream=stream, hidden=hidden,
                                   modulation=modulation, extras=extras)
                       for stream, hidden, modulation, extras
                       in tree_unflatten(list(values), structure))
        return run_layer(*states)

    def _run_layer(self, *states: StreamState,
                   active: tuple[str, ...], local_ids: tuple[int, ...],
                   layer_id: int, modules: Mapping[str, Any],
                   plan: AttentionPlan, layout: TokenLayout,
                   masks: dict[tuple[str, ...], tuple[TokenLayout, Any]],
                   override: LayerAttentionOverride | None
                   ) -> tuple[StreamState, ...]:
        # Each execution owns fresh containers; post_attn may update a state
        # or nested cache without mutating the checkpoint's input snapshot.
        live = tuple(_copy_state(state) for state in states)
        qkvs = {}
        for name, state, local_id in zip(active, live, local_ids):
            with self._stream_context(name):
                packed = modules[name].pre_attn(state, local_layer_id=local_id)
            if name in self.frozen_streams:
                packed = QKV(q=packed.q.detach(), k=packed.k.detach(),
                             v=packed.v.detach())
            qkvs[name] = packed
        if override is not None and override.handles_layer(layer_id):
            outs = override.run_attention(layer_id, qkvs, plan, layout)
            if set(outs) != set(active):
                raise ValueError(
                    f"LayerAttentionOverride returned stream set "
                    f"{sorted(outs)} at layer {layer_id} != "
                    f"participating streams {list(active)}"
                )
        else:
            # Never put joint attention under a frozen stream's no_grad:
            # trainable queries still need gradients through detached KV.
            outs = self._attend(active, qkvs, plan, layout, masks)
        after = []
        for name, state, local_id in zip(active, live, local_ids):
            out = outs[name]
            if name in self.frozen_streams:
                out = out.detach()
            with self._stream_context(name):
                state = modules[name].post_attn(
                    state, out, local_layer_id=local_id)
            after.append(self._stream_state(name, state))
        return tuple(after)

    def _attend(self, active: tuple[str, ...],
                qkvs: Mapping[str, QKV],
                plan: AttentionPlan,
                layout: TokenLayout,
                masks: dict[tuple[str, ...], tuple[TokenLayout, Any]]
                ) -> dict[str, Tensor]:
        """Concat QKV, run one masked attention, split back per stream.

        Args:
            active: Stream names participating at this layer, in layout
                order.
            qkvs: Per-stream ``QKV`` from ``pre_attn``.
            plan: Full visibility plan.
            layout: Full joint token layout.
            masks: Cache local to one ``run`` call, keyed by ``active``,
                holding the restricted layout and the materialized mask.

        Returns:
            Stream name to its slice of the attention output,
            ``[B, L_self, num_heads, head_dim]``.

        Raises:
            ValueError: If the participating streams disagree on
                ``num_heads``, or if the concatenated token count differs
                from the restricted layout total.
        """

        heads = {name: int(qkvs[name].q.shape[2]) for name in active}
        if len(set(heads.values())) != 1:
            raise ValueError(
                f"joint attention requires participating streams to share "
                f"num_heads (otherwise they cannot be concatenated along the "
                f"token axis), got {heads}. V3 only constrains head_dim; a "
                "topology with unequal num_heads needs per-stream separate "
                "attention, not one joint mask"
            )
        q = torch.cat([qkvs[name].q for name in active], dim=1)
        k = torch.cat([qkvs[name].k for name in active], dim=1)
        v = torch.cat([qkvs[name].v for name in active], dim=1)
        if active not in masks:
            sub_layout, sub_plan = restrict_to_streams(layout, plan, active)
            sub_plan = sub_plan.to(q.device)
            for channel in (sub_plan.key_valid, sub_plan.pairwise):
                if channel is not None and channel.shape[0] != q.shape[0]:
                    raise ValueError("attention mask batch size disagrees with QKV")
            masks[active] = (
                sub_layout,
                sub_plan.materialize(self.backend, block_size=self.block_size),
            )
        sub_layout, mask = masks[active]
        if q.shape[1] != sub_layout.total:
            raise ValueError(
                f"token count after concat {q.shape[1]} != layout total "
                f"{sub_layout.total}; segment lengths of participating "
                f"streams {list(active)} are "
                f"{[(s.stream, s.length) for s in sub_layout.segments]}"
            )
        out = _joint_attention(QKV(q=q, k=k, v=v), mask, backend=self.backend)
        return sub_layout.split(out, dim=1)


def _resolve_captures(captures: Mapping[str, Sequence[int]] | None,
                      index_maps: Mapping[str, Mapping[int, int]]
                      ) -> dict[str, tuple[int, ...]]:
    """Validate capture requests and normalize them.

    Args:
        captures: Stream name to the joint layers to capture, or None.
        index_maps: Stream name to its joint-to-local layer index map.

    Returns:
        Stream name to a tuple of joint layer indices. An empty dict when
        ``captures`` is falsy.

    Raises:
        KeyError: If a requested stream is not in the group.
        ValueError: If a requested joint layer is one the stream does not
            participate in. Raising beats skipping silently: the symptom of
            a silent skip is a KeyError from
            ``EngineOutput.require_capture`` over in L4, far from where the
            request was declared (validator V8 is meant to stop it at
            construction time).
    """
    if not captures:
        return {}
    stray = sorted(set(captures) - set(index_maps))
    if stray:
        raise KeyError(
            f"captures requested streams outside this group: {stray}; "
            f"group streams are {sorted(index_maps)}"
        )
    resolved: dict[str, tuple[int, ...]] = {}
    for name, layer_ids in captures.items():
        for layer_id in layer_ids:
            if layer_id not in index_maps[name]:
                raise ValueError(
                    f"captures asked to capture joint layer {layer_id} of "
                    f"stream {name!r}, but that stream does not participate "
                    f"in this layer; it participates in joint layers "
                    f"{sorted(index_maps[name])}"
                )
        resolved[name] = tuple(layer_ids)
    return resolved
