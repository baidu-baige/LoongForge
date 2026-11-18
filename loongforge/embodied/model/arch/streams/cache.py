# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 KV cache signatures: ``CacheSignature`` / ``CacheScope`` (Lane B).

Signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.6.

Neither of the first two models uses a KV cache, so this module declares the
signatures and implements nothing. It exists so that "does inference need a
cache" can be negotiated at construction time, through
``Scenario.has_kv_cache`` in ``CompiledMask.supports``, instead of
discovering at inference time that the question cannot be expressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor

__all__ = ["CacheSignature", "CacheScope"]


@dataclass(frozen=True)
class CacheSignature:
    """Cache shape signature for one stream at one layer.

    This is what guarantees static shapes (INV-2).

    Attributes:
        stream: Stream name.
        layer_id: Joint layer index.
        num_heads: Number of attention heads for this stream.
        head_dim: Per-head width.
        max_tokens: Maximum cached token count. Must be a construction-time
            constant: once the cache is captured in a CUDA graph, changing
            the length at runtime forces a recapture.
    """

    stream: str
    layer_id: int
    num_heads: int
    head_dim: int
    max_tokens: int

    def __post_init__(self) -> None:
        for name in ("num_heads", "head_dim", "max_tokens"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(
                    f"CacheSignature({self.stream!r}, L{self.layer_id})."
                    f"{name} must be positive, got {value}"
                )
        if self.layer_id < 0:
            raise ValueError(
                f"CacheSignature({self.stream!r}).layer_id must be "
                f"non-negative, got {self.layer_id}"
            )

    @property
    def key(self) -> tuple[str, int]:
        """Key for the cache dict.

        Returns:
            ``(stream, layer_id)``, deliberately the same shape of key as
            ``EngineOutput.captured``.
        """
        return (self.stream, self.layer_id)

    def shape(self, batch_size: int) -> tuple[int, int, int, int]:
        """Cache tensor shape for a given batch size.

        Args:
            batch_size: Number of sequences.

        Returns:
            ``[B, max_tokens, num_heads, head_dim]``, matching the ``QKV``
            layout.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return (batch_size, self.max_tokens, self.num_heads, self.head_dim)


class CacheScope(Protocol):
    """Cache lifetime within one inference run. Signature only for now.

    ``reset`` exists instead of rebuilding the object because rebuilding
    reallocates device memory, while a CUDA graph captures addresses.
    """

    def signatures(self) -> tuple[CacheSignature, ...]:
        """List every cache slot this scope manages.

        Returns:
            All slots, enumerable at construction time.
        """
        ...

    def read(self, signature: CacheSignature) -> tuple[Tensor, Tensor]:
        """Read one cache slot.

        Args:
            signature: Slot to read.

        Returns:
            The ``(key, value)`` pair, each of shape
            ``signature.shape(batch_size)``.
        """
        ...

    def write(self, signature: CacheSignature, *, key: Tensor, value: Tensor,
              at: int) -> None:
        """Write tokens into one cache slot in place.

        Args:
            signature: Slot to write.
            key: Key tensor to store.
            value: Value tensor to store.
            at: Token offset at which the write starts. The cache shape must
                not change.
        """
        ...

    def reset(self) -> None:
        """Clear the valid length without reallocating device memory."""
        ...

    def state(self) -> dict[str, Any]:
        """Return a debugging snapshot.

        Returns:
            Plain Python values. This never enters state_dict (INV-3).
        """
        ...
