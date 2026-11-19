# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 per-stream state and the ``StreamModule`` protocol (Lane B).

Signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.3.

The four ``StreamModule`` methods cut "what one layer does" into four
stages. The engine only calls them in order and never touches the tensor
semantics inside a stream. That is what lets a single-stream model degrade
to N=1 instead of taking a separate code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:       # typing only; not imported at runtime so that
                        # state and conditions do not form an import cycle
    from .conditions import ConditionBus

__all__ = ["StreamState", "QKV", "StreamModule"]


@dataclass
class QKV:
    """Q/K/V produced by one stream at one layer, not yet concatenated.

    Shapes may differ per stream. The convention is
    ``[B, L, num_heads, head_dim]``. The last two axes stay separate instead
    of being flattened into attn_width because joint attention allows each
    stream a different ``num_heads`` but requires the same ``head_dim``
    (validator V3); flattening would hide that check.

    Attributes:
        q: Query tensor, ``[B, L_self, num_heads, head_dim]``.
        k: Key tensor, ``[B, L_self, num_heads, head_dim]``.
        v: Value tensor, same token count as ``k``.
    """

    q: Tensor
    k: Tensor
    v: Tensor

    def __post_init__(self) -> None:
        for name in ("q", "k", "v"):
            tensor = getattr(self, name)
            if tensor.ndim != 4:
                raise ValueError(
                    f"QKV.{name} must be 4-D [B, L, num_heads, head_dim], "
                    f"got {tuple(tensor.shape)}"
                )
        if self.k.shape[1] != self.v.shape[1]:
            raise ValueError(
                f"QKV: k has {self.k.shape[1]} tokens != v has "
                f"{self.v.shape[1]}"
            )
        if self.q.shape[-1] != self.k.shape[-1]:
            raise ValueError(
                f"QKV: q head_dim {self.q.shape[-1]} != k head_dim "
                f"{self.k.shape[-1]} (V3 requires joint streams to share "
                "head_dim)"
            )

    @property
    def head_dim(self) -> int:
        """Per-head width, read off the last axis of ``q``."""
        return int(self.q.shape[-1])

    @property
    def num_query_tokens(self) -> int:
        """Number of query tokens this stream contributes at this layer."""
        return int(self.q.shape[1])


@dataclass
class StreamState:
    """Everything one stream carries from layer to layer.

    The engine owns the instance; the stream module reads and writes it.

    Attributes:
        stream: Stream name, equal to ``StreamSpec.name``.
        hidden: Residual tensor ``[B, L, residual_width]``. The width is
            ``StreamSpec.residual_width``, NOT attn_width.
        modulation: Packed modulation tensor, or None. Slot order is given by
            ``StreamSpec.modulation_slots``.
        extras: Anything the stream itself needs across layers, for example
            RoPE frequencies or a per-layer modulation cache. The engine does
            not interpret the contents and only passes them through.
    """

    stream: str
    hidden: Tensor
    modulation: Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hidden.ndim != 3:
            raise ValueError(
                f"StreamState({self.stream!r}).hidden must be 3-D "
                f"[B, L, residual_width], got {tuple(self.hidden.shape)}"
            )

    @property
    def num_tokens(self) -> int:
        """Token count of this stream.

        Returns:
            ``hidden.shape[1]``. Checks C8/C9 use it to assert the token
            count is unchanged across a layer.
        """
        return int(self.hidden.shape[1])

    @property
    def width(self) -> int:
        """Residual width of this stream, read off ``hidden.shape[-1]``."""
        return int(self.hidden.shape[-1])

    def replace_hidden(self, hidden: Tensor) -> StreamState:
        """Return a copy with a new residual and the other fields kept.

        The engine uses this to write each layer's output back.
        ``dataclasses.replace`` would not run the token-count check, which is
        why this is an explicit method rather than an alias for it.

        Args:
            hidden: New residual tensor; its token count must match.

        Returns:
            A new ``StreamState`` sharing ``modulation`` and a shallow copy
            of ``extras``.

        Raises:
            ValueError: If the token count changed (check C9: a layer must
                not change a stream's token count).
        """
        if hidden.shape[1] != self.num_tokens:
            raise ValueError(
                f"StreamState({self.stream!r}).replace_hidden: token count "
                f"went from {self.num_tokens} to {hidden.shape[1]}; a layer "
                "must not change a stream's token count (C9)"
            )
        return StreamState(stream=self.stream, hidden=hidden,
                           modulation=self.modulation, extras=dict(self.extras))


class StreamModule(Protocol):
    """Protocol a stream's block must satisfy.

    It is what ``build(StreamSpec.impl, ...)`` produces.

    Why four stages: joint attention has to happen after every stream has
    computed its QKV and before any stream runs its post-attention part.
    Writing a layer as one ``forward`` cannot express that, which is why the
    current code interleaves two streams' code inside ``MoT.forward``.

    ``local_layer_id`` is this stream's own local layer index, resolved at
    construction time by ``StreamSpec.resolve_layer_index_map``. A stream
    module never sees the joint layer index; if it did, streams would start
    deciding "do I take part" on their own.
    """

    def prepare(self, x: Tensor, *,
                timestep: Tensor | None = None,
                conditions: ConditionBus | None = None) -> StreamState:
        """Turn the raw input tensor into this stream's layer-0 state.

        Called once per forward pass.

        The argument is a bare tensor rather than a ``StreamState`` because
        patchify and tokenization belong to the stream (L2 detailed design
        §1.1 assigns them to ``prepare``), and before patchify the raw video
        is ``[B, C, F, H, W]`` while actions are ``[B, T, D]``, neither of
        which is the ``[B, L, W]`` that ``StreamState`` requires. Making
        ``prepare`` take a ``StreamState`` would force tokenization to happen
        outside it, and that outside could only be the engine or the builder,
        so every new modality would mean a framework code change.

        Modulation computation and RoPE frequency preparation also happen
        here; their products hang off ``modulation`` and ``extras``.

        Args:
            x: Raw input tensor for this stream, taken from
                ``batch[StreamSpec.batch_key]``.
            timestep: Diffusion timestep, or None when
                ``StreamSpec.needs_timestep`` is False (a condition-only
                stream such as VL).
            conditions: The whole condition bus. The stream module picks the
                conditions it wants by name (§4.4). It appears only in this
                method: CROSS_ATTENTION KV and CHANNEL_CONCAT channels are
                layer-invariant, so projecting once and storing the result in
                ``extras`` is cheaper than reprojecting per layer, and it
                keeps the ``pre_attn`` / ``post_attn`` signatures independent
                of how many conditions exist.

        Returns:
            A ``StreamState`` whose ``hidden`` is ``[B, L, residual_width]``,
            where ``L`` must equal the segment length the layout assigns to
            this stream (the engine reconciles this in ``run``).
        """
        ...

    def pre_attn(self, state: StreamState, *, local_layer_id: int) -> QKV:
        """Compute this layer's Q/K/V without performing attention.

        Args:
            state: Current state of this stream.
            local_layer_id: Index into this stream's own layer list.

        Returns:
            The ``QKV`` triple, each ``[B, L_self, num_heads, head_dim]``.
        """
        ...

    def post_attn(self, state: StreamState, attn_out: Tensor, *,
                  local_layer_id: int) -> StreamState:
        """Consume this stream's slice of the joint attention output.

        Performs the output projection, the residual add, and the FFN.

        Args:
            state: State as passed to ``pre_attn`` for the same layer.
            attn_out: ``[B, L_self, num_heads, head_dim]``. Its token count
                must equal the token count of the ``q`` returned by
                ``pre_attn``.
            local_layer_id: Index into this stream's own layer list.

        Returns:
            The updated state, with the same token count.
        """
        ...

    def finalize(self, state: StreamState) -> StreamState:
        """Run once after all layers, for example a final norm.

        This excludes output heads: heads belong to L4 (``heads/``) and read
        this method's product through ``HiddenStateRef``.

        Args:
            state: State after the last layer.

        Returns:
            The finalized state.
        """
        ...
