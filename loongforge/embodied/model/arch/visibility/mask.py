# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 materialization: the two mask channels and capability negotiation (Lane A).

``AttentionPlan`` carries the two channels and ``CompiledMask`` is the
materialized result. The signatures follow
``docs/design/mot_layer_interfaces.zh-CN.md`` §3.4 / §3.5 / §3.7.
``AttentionPlan`` plus the four methods of ``CompiledMask`` make up interface
**I1**: Lane B may call only those four methods and may not read
``TokenLayout``'s labels to decide visibility on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from ..layout import SegmentRange, TokenLayout
from .combine import Visibility

__all__ = [
    "AttentionPlan", "Capability", "Scenario", "CompiledMask",
    "LayerAttentionOverride", "KNOWN_BACKENDS",
]

KNOWN_BACKENDS = ("dense", "sdpa", "flex", "varlen")
"""Registered attention backends.

``attention_backend`` may only take a value from this tuple.

``auto`` is deliberately absent: it must be resolved away at build time
(``Scenario.__post_init__`` already blocks it), otherwise the negotiation result
would depend on something only known at run time.
"""


def _block_mask(structural: Tensor, *, block_size: int,
                key_valid: Tensor | None) -> Any:
    """Build a FlexAttention ``BlockMask`` from the structural channel.

    The two channels are **composed into one mask_mod** here, instead of folding
    key_valid into structural: the structural part is a lookup on (q, kv) only
    and is independent of b, so flex can skip whole all-False blocks; the
    per-sample part costs just one more ``key_valid[b, kv]`` lookup. Doing it the
    other way round (``&``-ing them into one ``[B, Q, K]`` tensor first) defeats
    block-level skipping, which is the reason I1 has two channels.

    Args:
        structural: ``[total, total]`` bool, shared across the batch.
        block_size: BlockMask block size.
        key_valid: ``[B, total]`` bool per-sample valid bits, or None.

    Returns:
        A FlexAttention ``BlockMask``.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    device = structural.device
    valid = None if key_valid is None else key_valid.to(device)

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor,
                 kv_idx: Tensor) -> Tensor:
        del h
        keep = structural[q_idx, kv_idx]
        if valid is None:
            return keep
        return keep & valid[b, kv_idx]

    total = int(structural.shape[0])
    batch = None if valid is None else int(valid.shape[0])
    return create_block_mask(mask_mod, B=batch, H=None, Q_LEN=total,
                             KV_LEN=total, device=device,
                             BLOCK_SIZE=block_size)


@dataclass(frozen=True)
class AttentionPlan:
    """Carrier of I1. The two-channel split is what makes this refactor fast.

    Why two channels: structural visibility (who may see whom) is shared across
    the batch, so it is compilable, cacheable, and can generate a BlockMask;
    per-sample valid bits (where this sample's padding is) necessarily carry a
    batch dimension. Mixing them into one tensor gives the whole mask a batch
    dimension, which kills BlockMask reuse and forces everything back to dense.

    Hard invariant (master plan §1 rule 1 / INV-1):
        ``structural.shape == (total, total)``, **exactly 2 dims, no batch
        dimension**. That assertion lives in ``__post_init__``, and CI has a
        separate shape-assertion case for it.

    Attributes:
        structural: Batch-shared structural visibility, or None for all visible.
        key_valid: Per-sample valid bits, or None for all valid.
        layout: The layout these channels came from.
    """

    structural: Tensor | None = None    # [total, total] bool, shared; None = all visible
    key_valid: Tensor | None = None     # [B, total] bool, per-sample; None = all valid
    layout: TokenLayout | None = None   # Source layout; lets B assert order (C8)
    # structural=None is **the normal case for a monolithic model**
    # (interfaces §2.3): one stream and no causality means all visible, and then
    # materializing an all-True [L, L] tensor is pure wasted memory and blocks
    # flash's fast path. In a multi-stream model structural=None shows up only
    # for "fully bidirectional" (the existing motus is such a case).

    def __post_init__(self) -> None:
        if self.structural is not None:
            shape = tuple(self.structural.shape)
            if self.structural.ndim != 2:
                raise ValueError(
                    f"AttentionPlan.structural must be exactly 2-D (no batch "
                    f"dim, INV-1), got {shape}"
                )
            if shape[0] != shape[1]:
                raise ValueError(
                    f"AttentionPlan.structural must be square, got {shape}"
                )
            if self.layout is not None and shape[0] != self.layout.total:
                raise ValueError(
                    f"AttentionPlan.structural shape {shape} disagrees with "
                    f"layout.total {self.layout.total}"
                )
        if self.key_valid is not None:
            if self.key_valid.ndim != 2:
                raise ValueError(
                    f"AttentionPlan.key_valid must be 2-D [B, total], "
                    f"got {tuple(self.key_valid.shape)}"
                )
            if self.layout is not None \
                    and self.key_valid.shape[1] != self.layout.total:
                raise ValueError(
                    f"AttentionPlan.key_valid total dim "
                    f"{self.key_valid.shape[1]} != layout.total {self.layout.total}"
                )

    @property
    def total(self) -> int:
        if self.layout is not None:
            return self.layout.total
        if self.structural is not None:
            return int(self.structural.shape[0])
        if self.key_valid is not None:
            return int(self.key_valid.shape[1])
        raise ValueError(
            "AttentionPlan has all three fields None, cannot report total")

    @classmethod
    def from_visibility(cls, visibility: Visibility | None,
                        layout: TokenLayout,
                        *, key_valid: Tensor | None = None) -> AttentionPlan:
        """Step 2 of ``WorldModel.__init__``: materialize the structural channel once.

        ``visibility is None`` means fully bidirectional, and ``structural``
        stays None (absent means unconstrained; single-stream models and the
        existing motus take this path).

        Args:
            visibility: The visibility expression, or None for bidirectional.
            layout: Token layout to evaluate against.
            key_valid: Per-sample valid bits, or None.

        Returns:
            The constructed ``AttentionPlan``.
        """
        structural = None if visibility is None else visibility.evaluate(layout)
        return cls(structural=structural, key_valid=key_valid, layout=layout)

    def dense_bool(self) -> Tensor:
        """Both channels composed into a ``[B or 1, 1, Lq, Lk]`` bool tensor.

        The shared base of the dense and sdpa paths. When ``key_valid`` is None
        the batch dimension stays 1, so it broadcasts instead of being
        materialized; that is the entire memory benefit of keeping the
        structural channel batch-shared.

        Returns:
            Bool mask of shape ``[B or 1, 1, Lq, Lk]``.
        """
        total = self.total
        if self.structural is None:
            device = None if self.key_valid is None else self.key_valid.device
            base = torch.ones((1, 1, total, total), dtype=torch.bool,
                              device=device)
        else:
            base = self.structural.unsqueeze(0).unsqueeze(0)
        if self.key_valid is None:
            return base
        return base & self.key_valid.to(base.device)[:, None, None, :]

    def materialize(self, backend: str, *,
                    block_size: int | None = None) -> Any:
        """Produce the object the attention implementation actually wants.

        dense → ``[B or 1, 1, Lq, Lk]`` bool (batch dim broadcast when possible)
        flex  → BlockMask (structural lookup, key_valid folded into the same
                mask_mod); **only this path reads block_size**, taken from
                ``WorldModelSpec.attention_block_size``
        sdpa  → additive float bias (-inf where invisible)
        both channels None → returns None, so the caller takes the maskless
                fast path

        The return type is deliberately Any: each backend wants a different
        object type.

        Args:
            backend: One of the names in ``KNOWN_BACKENDS``.
            block_size: BlockMask block size; required for and used only by
                ``flex``.

        Returns:
            The backend-specific mask object, or None when nothing is masked.

        Raises:
            ValueError: On an unknown backend, or on ``flex`` without a
                structural channel or without ``block_size``.
        """
        if self.structural is None and self.key_valid is None:
            return None
        if backend == "dense":
            return self.dense_bool()
        if backend == "sdpa":
            dense = self.dense_bool()
            bias = torch.zeros(dense.shape, dtype=torch.float32,
                               device=dense.device)
            return bias.masked_fill(~dense, float("-inf"))
        if backend == "flex":
            if self.structural is None:
                raise ValueError(
                    "AttentionPlan.materialize(flex): structural channel is "
                    "None (all visible) and only key_valid is set. BlockMask "
                    "block-level skipping buys nothing on such input; use "
                    "dense, or let the caller apply key_valid as padding"
                )
            if block_size is None:
                raise ValueError(
                    "AttentionPlan.materialize(flex) requires block_size, and "
                    "it must be the same value Scenario.block_size used during "
                    "negotiation: a different value is a different mask, and "
                    "the BlockMask gets captured by torch.compile"
                )
            return _block_mask(self.structural, block_size=block_size,
                               key_valid=self.key_valid)
        raise ValueError(
            f"AttentionPlan.materialize: unknown backend {backend!r}; "
            f"registered backends are {list(KNOWN_BACKENDS)}"
        )


class Capability(Enum):
    """The three possible outcomes of capability negotiation."""

    EXACT = "exact"              # The backend expresses this visibility exactly
    DEGRADED = "degraded"        # Expressible but degraded (e.g. flash → SDPA)
    UNSUPPORTED = "unsupported"  # Not expressible


@dataclass(frozen=True)
class Scenario:
    """Description of a negotiation scenario.

    Training and inference are negotiated separately, which is why this object
    exists.

    The five fields come from three places (backend/block_size from L2,
    cuda_graph from the trainer, has_kv_cache from the inference path); build
    time collects and injects them, and L1 only consumes, never constructs.
    Ownership is stated in master plan §10-5 (due 12/04).

    Attributes:
        phase: Which phase is being negotiated.
        backend: Resolved backend name.
        has_kv_cache: Whether a KV cache is in play.
        cuda_graph: Whether CUDA graph capture is in play.
        block_size: BlockMask block size, meaningful for ``flex`` only.
    """

    phase: Literal["train", "inference"]
    backend: str                     # The **resolved** value of
                                     # WorldModelSpec.attention_backend
    has_kv_cache: bool = False
    cuda_graph: bool = False
    block_size: int | None = None    # Meaningful for backend="flex" only. It is
                                     # part of Scenario and not merely an
                                     # as_block_mask argument because it changes
                                     # the negotiation result: a visibility whose
                                     # boundaries do not land on multiples of
                                     # block_size can only get DEGRADED from
                                     # BlockMask

    def __post_init__(self) -> None:
        if self.backend == "auto":
            raise ValueError(
                "Scenario.backend does not accept 'auto': build time must "
                "already have resolved it to a concrete backend"
            )


@dataclass(frozen=True)
class CompiledMask:
    """A materialized mask. Lane B **may only call** these four methods.

    That restriction is the I1 agreement. B may not read ``TokenLayout``'s
    labels and decide for itself; otherwise the same rule would have two
    implementations, and they would diverge on boundary frames and padding rows.

    Attributes:
        structural: ``[total, total]`` bool from ``Visibility.evaluate``.
        layout: The layout it was evaluated on.
        backend: The backend this mask was materialized for.
    """

    structural: Tensor          # [total, total] bool, from Visibility.evaluate
    layout: TokenLayout
    backend: str

    def as_dense(self) -> Tensor:
        """``[1, 1, Lq, Lk]`` bool.

        The fallback path of every backend, and also the reference for the
        equivalence tests.

        Returns:
            Bool mask of shape ``[1, 1, Lq, Lk]``.
        """
        return self.structural.unsqueeze(0).unsqueeze(0)

    def as_block_mask(self, *, block_size: int) -> Any:
        """FlexAttention's BlockMask. Requires structural to have no batch dim.

        ``block_size`` must be the same value used by ``Scenario.block_size``
        during negotiation: changing the value means a different mask, and the
        BlockMask gets captured by ``torch.compile``, so changing it at run time
        triggers a recompile (hard constraint: static shapes).

        Args:
            block_size: BlockMask block size; must match the negotiated value.

        Returns:
            A FlexAttention ``BlockMask``.

        Raises:
            ValueError: If structural is not exactly 2-D, or block_size is not
                positive.
        """
        if self.structural.ndim != 2:
            raise ValueError(
                f"CompiledMask.as_block_mask: structural must be exactly 2-D "
                f"(INV-1), got {tuple(self.structural.shape)}"
            )
        if block_size <= 0:
            raise ValueError(
                f"CompiledMask.as_block_mask: block_size must be positive, "
                f"got {block_size}"
            )
        return _block_mask(self.structural, block_size=block_size,
                           key_valid=None)

    def as_segments(self) -> tuple[SegmentRange, ...]:
        """Materialize a segment table (for the varlen path).

        Neither of the first two models needs it, so it is the first thing that
        can be dropped when degrading.

        The structural channel is first verified to **actually** be
        block-diagonal by segment, and only then is the segment table returned.
        Returning ``layout.segments`` directly would assume that visibility
        happens to be block-diagonal, and the cross-stream edges of fastwam and
        VPP2 are exactly what breaks that assumption.

        Returns:
            The layout's segment ranges.

        Raises:
            ValueError: If the visibility is not block-diagonal by segment; the
                message counts the extra and missing edges.
        """
        total = self.layout.total
        expected = torch.zeros((total, total), dtype=torch.bool,
                               device=self.structural.device)
        for seg in self.layout.segments:
            expected[seg.start:seg.stop, seg.start:seg.stop] = True
        if not torch.equal(self.structural, expected):
            extra = int((self.structural & ~expected).sum())
            lost = int((~self.structural & expected).sum())
            raise ValueError(
                f"CompiledMask.as_segments: this visibility is not "
                f"block-diagonal by segment ({extra} extra edges outside "
                f"segments, {lost} missing edges inside segments), varlen "
                "cannot express it. Use the dense or flex backend instead"
            )
        return self.layout.segments

    @staticmethod
    def negotiate_capability(vis: Visibility, scenario: Scenario) -> Capability:
        """Capability negotiation, done once at construction time.

        **Must be called from build_world_model_spec, not from forward.**

        The class of defect this is meant to catch: EXACT during training,
        UNSUPPORTED during inference. Today that defect shows up as "training
        works, inference raises NotImplementedError", and you only find out by
        actually reaching inference. Negotiating early turns it into a
        configuration error.

        This implementation **looks only at the backend and the scenario**, not
        at the layout, because the layout is not in the signature (that is
        precisely the open item in master plan §10-5). So the varlen tier gets a
        **conservative verdict based on the expression**: only the always-true
        and the purely intra-segment expressions get EXACT, everything else gets
        UNSUPPORTED. Too strict beats too loose: judging it too loose would let
        varlen see tokens it should not, while the numbers still converge and
        nobody notices.

        Args:
            vis: The visibility expression to negotiate.
            scenario: The scenario (phase, backend, cache, graph, block size).

        Returns:
            The negotiated ``Capability``.
        """
        if scenario.backend not in KNOWN_BACKENDS:
            return Capability.UNSUPPORTED
        if scenario.backend in ("dense", "sdpa"):
            # The full [Q, K] is materialized, so any elementwise predicate
            # can be expressed exactly
            return Capability.EXACT
        if scenario.backend == "flex":
            if scenario.block_size is None:
                # Without a block size there is no BlockMask to build, so the
                # only option is falling back to dense
                return Capability.DEGRADED
            if scenario.has_kv_cache:
                # With a KV cache KV_LEN changes every step, so the BlockMask
                # has to be rebuilt step by step, which is mutually exclusive
                # with cuda_graph capture; that counts as degraded, not
                # unsupported
                return Capability.DEGRADED
            return Capability.EXACT
        if vis.expr in ("always_visible", "same_stream()"):
            return Capability.EXACT
        return Capability.UNSUPPORTED


class LayerAttentionOverride(Protocol):
    """Per-layer override / escape hatch for layers that truncate keys inside.

    Some layers physically truncate the key set **inside** the layer, so the
    sequence length is not a constant and an [L, L] mask cannot express it.

    Conflict with INV-2 (static shapes), and the resulting boundaries:
      - a layer that goes through an override **cannot** be captured into a
        CUDA graph;
      - therefore an override must be declared in the configuration so that
        per_microbatch_cuda_graph skips those layers at capture time, rather
        than discovering it at run time;
      - neither of the first two models (VPP2, InternW0-Δ) needs it, so the
        first batch **defines the signature only and implements no consumer**,
        and the static-shape rule holds throughout.

    Agreement: an override must not change any stream's token count (otherwise
    the engine's C8/C9 will raise at the next layer). It only changes "how this
    layer computes internally", not "how long this layer's output is".
    """

    def handles_layer(self, layer_id: int) -> bool:
        """Whether this override takes over the given joint layer index.

        Queryable at construction time, so capture logic can exclude the layer.

        Args:
            layer_id: Joint layer index.

        Returns:
            True if this override handles that layer.
        """
        ...

    def run_attention(self, layer_id: int,
                      qkvs: Mapping[str, Any],
                      plan: AttentionPlan,
                      layout: TokenLayout) -> dict[str, Tensor]:
        """Run this layer's attention itself and return per-stream outputs.

        What it receives is the **unconcatenated** per-stream QKV
        (``streams.state.QKV``), because the whole point of an override is not
        taking the "concatenate, then one masked attention" path.

        Args:
            layer_id: Joint layer index.
            qkvs: Per-stream QKV bundles, not concatenated.
            plan: The attention plan for this forward.
            layout: The token layout.

        Returns:
            Per-stream outputs. The key set must equal that of ``qkvs``, and
            each tensor's token count must be unchanged.
        """
        ...
