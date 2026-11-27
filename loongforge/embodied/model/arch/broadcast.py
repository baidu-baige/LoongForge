# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 broadcast spec ``BroadcastSpec`` (Lane C).

The signature follows ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.4.

This module is where interface **I2** (A -> C) lands on the C side: which axes a
timestep / noise value varies along must be decided by the label names of
``TokenLayout``, and **never guessed from the tensor's ndim**.

The practice being replaced is ``has_seq = len(t.shape) == 4``: the moment a
shape changes, that test silently picks the wrong branch. Once ``axes`` names
the labels explicitly, a missing label is a construction-time KeyError.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .layout import TokenLayout

__all__ = ["BroadcastSpec", "DEFAULT_PER_SAMPLE"]


@dataclass(frozen=True)
class BroadcastSpec:
    """Which **label axes** a timestep / noise value varies along.

    ``axes = ()`` means one scalar per sample (the common case; every existing
    model sits here). ``axes = ("frame",)`` means one timestep per frame (VPP2's
    per-frame noise). ``axes = ("frame", "chunk")`` means frame x action chunk.

    Axis names must be labels that actually exist in ``TokenLayout`` (I2);
    ``validate`` checks this once at construction time. A name that is not found
    raises instead of degrading to per-sample, because that degradation is
    exactly the entry point of the ndim-guessing path described above.

    Attributes:
        axes: Declared label names, in the order the value's axes follow.
    """

    axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.axes)) != len(self.axes):
            raise ValueError(
                f"BroadcastSpec.axes has duplicate axes: {self.axes}")

    @property
    def is_per_sample(self) -> bool:
        """No axes means a single scalar per sample."""
        return not self.axes

    def validate(self, layout: TokenLayout) -> None:
        """Check I2: every axis name must be a label present in the layout.

        **Must be called at construction time** (V5 / V6 of ``WorldModelSpec``).
        Checking during forward is too late: by then the error shows up as a
        broadcast tensor whose shape happens to multiply cleanly.

        Args:
            layout: The token layout whose labels the axis names must exist in.

        Raises:
            ValueError: If any axis name is not a label of ``layout``.
        """
        missing = [axis for axis in self.axes if not layout.has_label(axis)]
        if missing:
            raise ValueError(
                f"BroadcastSpec.axes labels {missing} are not in the layout; "
                f"the layout provides labels {sorted(layout.labels)}. "
                "A missing label does not degrade to per-sample, that is "
                "exactly the ndim-guessing path this replaces"
            )

    def expand(self, value: Tensor, layout: TokenLayout) -> Tensor:
        """Broadcast a value organised by ``axes`` out to ``[B, total, 1]``.

        ``axes = ()`` expects ``value`` of shape [B] and returns [B, 1, 1] (a
        pure view, no copy). ``axes = ("frame",)`` expects ``value`` of shape
        [B, num_frames] and gathers along ``layout.labels_of("frame")``.

        With several axes the labels are folded into one C-order flat index in
        the **declared order** of ``axes``, matching the axis order of
        ``value``; so ``axes=("frame", "chunk")`` requires ``value`` to be
        ``[B, num_frames, num_chunks]``.

        Tokens whose label value is **-1** (alignment padding, and ``row`` /
        ``col`` on action tokens) have **no position** on that axis and always
        get 0. They are not gathered from a neighbour: -1 used as an index wraps
        around to the tail and would silently pick up the last frame's
        timestep with an exactly correct shape.

        Args:
            value: Values laid out as ``[B, *axis_sizes]`` (or ``[B]`` when
                ``axes`` is empty).
            layout: Layout supplying the per-token label vectors.

        Returns:
            A ``[B, total, 1]`` tensor aligned with the layout's token order.

        Raises:
            ValueError: If ``value`` has the wrong rank, or a label exceeds the
                corresponding axis length of ``value``.
        """
        self.validate(layout)
        if not self.axes:
            if value.ndim != 1:
                raise ValueError(
                    f"BroadcastSpec(axes=()).expand expects a per-sample "
                    f"scalar of shape [B], got {tuple(value.shape)}"
                )
            return value.view(-1, 1, 1)
        if value.ndim != 1 + len(self.axes):
            raise ValueError(
                f"BroadcastSpec(axes={self.axes}).expand expects "
                f"[B, {', '.join(self.axes)}], {1 + len(self.axes)} dims in "
                f"total, got {tuple(value.shape)}"
            )
        flat = None
        undefined = None
        for position, axis in enumerate(self.axes):
            label = layout.labels_of(axis).to(value.device)
            size = value.shape[1 + position]
            index = label.clamp(min=0)
            highest = int(index.max())
            if highest >= size:
                raise ValueError(
                    f"BroadcastSpec.expand: axis {axis!r} has largest label "
                    f"{highest}, which exceeds length {size} of value dim "
                    f"{1 + position}"
                )
            flat = index if flat is None else flat * size + index
            missing = label < 0
            undefined = missing if undefined is None else undefined | missing
        picked = value.reshape(value.shape[0], -1).index_select(1, flat)
        assert undefined is not None            # axes non-empty => one pass ran
        return picked.masked_fill(undefined.unsqueeze(0), 0.0).unsqueeze(-1)


DEFAULT_PER_SAMPLE = BroadcastSpec(axes=())
"""One scalar per sample. **The setting every existing model uses**, and the
default."""
