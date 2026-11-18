# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 cross-stream KV routing: ``KVRoute`` (Lane B).

Signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.5;
configurability follows ``mot_spec_config.zh-CN.md`` §3.5, which keeps the
stream names out of YAML because they are topology.

Division of labour with visibility (L1), the easiest pair to confuse:
  - visibility says which queries may attend to which keys inside one joint
    attention call;
  - KV routing says which layer of which stream a given layer takes its KV
    from.
The first is a mask, the second is a tensor lookup. The EWAM VL stream feeds
the KV of its layer k to video-stream layer j, and a mask cannot express
that: it changes which tensor is read, not which position is hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["KVRoute"]


@dataclass(frozen=True)
class KVRoute:
    """One directed KV route from a ``src`` stream layer to a ``dst`` layer.

    ``src_layers`` and ``dst_layers`` are two parallel lists paired by index
    and must have equal length. Two lists rather than a dict because in
    config files these two columns are usually contiguous ranges, and
    parallel lists read as a reconciliation table.

    An empty route set (``routes=()``) is the normal case for a
    single-stream model: there is no peer stream to route to. The engine must
    not special-case ``if not routes``; the empty set takes the same
    "iterate over routes" path and happens to be an empty loop.

    Attributes:
        src: Name of the stream providing KV. Topology, not YAML-visible.
        dst: Name of the stream consuming KV. Topology, not YAML-visible.
        src_layers: Local layer indices on the src side.
        dst_layers: Local layer indices on the dst side.
    """

    src: str
    dst: str
    src_layers: tuple[int, ...]
    dst_layers: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.src == self.dst:
            raise ValueError(
                f"KVRoute src and dst must differ (both {self.src!r}); "
                "taking KV from the same stream is not routing, it is the "
                "stream's own business"
            )
        if len(self.src_layers) != len(self.dst_layers):
            raise ValueError(
                f"KVRoute({self.src!r} -> {self.dst!r}): src_layers has "
                f"{len(self.src_layers)} entries, dst_layers has "
                f"{len(self.dst_layers)}; the two parallel lists must have "
                "equal length"
            )
        if not self.src_layers:
            raise ValueError(
                f"KVRoute({self.src!r} -> {self.dst!r}) layer lists must not "
                "be empty; do not build a route that routes no layer"
            )
        for name in ("src_layers", "dst_layers"):
            layers = getattr(self, name)
            if any(layer < 0 for layer in layers):
                raise ValueError(
                    f"KVRoute({self.src!r} -> {self.dst!r}).{name} layer "
                    f"indices must be non-negative: {layers}"
                )
        if len(set(self.dst_layers)) != len(self.dst_layers):
            raise ValueError(
                f"KVRoute({self.src!r} -> {self.dst!r}).dst_layers has "
                f"duplicates: {self.dst_layers}. One dst layer taking two KV "
                "sources is an ambiguous configuration"
            )

    def resolve(self, dst_layer: int) -> int | None:
        """Map a dst local layer index to the src local layer to read.

        Args:
            dst_layer: Local layer index on the dst side.

        Returns:
            The src local layer index, or None when this route covers no
            such layer. None rather than an exception because most layers
            have no cross-stream route, so "absent" is normal, not an error.
        """
        try:
            index = self.dst_layers.index(dst_layer)
        except ValueError:
            return None
        return self.src_layers[index]

    @classmethod
    def per_layer(cls, src: str, dst: str, *, num_layers: int) -> KVRoute:
        """Build a one-to-one route where dst layer i reads src layer i.

        Args:
            src: Stream providing KV.
            dst: Stream consuming KV.
            num_layers: Number of paired layers.

        Returns:
            The route with both layer lists equal to ``range(num_layers)``.

        Raises:
            ValueError: If ``num_layers`` is not positive.
        """
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        layers = tuple(range(num_layers))
        return cls(src=src, dst=dst, src_layers=layers, dst_layers=layers)

    @classmethod
    def fixed(cls, src: str, dst: str, *, src_layer: int,
              dst_layers: tuple[int, ...]) -> KVRoute:
        """Build a route where many dst layers read the same src layer.

        This is the common "feed the last VL encoder layer's KV to every
        video-stream layer" pattern, which appears in the current code as a
        cached variable indexed repeatedly.

        Args:
            src: Stream providing KV.
            dst: Stream consuming KV.
            src_layer: The single src local layer every dst layer reads.
            dst_layers: Local dst layer indices to cover.

        Returns:
            The route with ``src_layers`` repeating ``src_layer``.
        """
        return cls(src=src, dst=dst,
                   src_layers=(src_layer,) * len(dst_layers),
                   dst_layers=dst_layers)
