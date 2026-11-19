# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""**Algorithm implementations** of the L4 output heads (Lane C).

Split of work against the sibling modules: ``spec.py`` holds the declaration
(``HeadSpec``), ``head.py`` holds the protocol (``OutputHead``),
``hidden_ref.py`` holds "which layer of which stream do I read", and this
module holds "how the hidden state actually becomes the output".

Four forms, all taken from existing code and **not interchangeable** (each has
a different number of parameters):

====================  ==================================================
``patch_flow``        ``norm`` + two-slot modulation + ``head`` Linear +
                      patch unflatten. The existing ``Head`` at
                      ``wan_video_dit.py:279-298`` plus ``unpatchify``
                      at ``:405-416``. The video flow head.
``linear``            A bare ``Linear``. The existing ``self.head`` at
                      ``action_dit.py:150``; ``post_dit`` at ``:423``
                      calls it exactly once.
``norm_linear``       ``LayerNorm`` + ``Linear``. InternW0-Δ's existing
                      ``future_delta`` (``wan_video_dit.py:557-563``).
``identity``          **No parameters**; returns the hidden state as is.
                      The existing ``semantic_future`` computes
                      ``1 - cos`` directly on the 3072-dim raw hidden
                      state (``losses.py:58-72``); it has no projection.
====================  ==================================================

The form name is a **pure implementation detail**, so it travels in the
``form`` key of ``Spec.params`` rather than as a ``HeadSpec`` field: it changes
no externally visible shape (the exit width is already
``HeadSpec.output_width``), it only decides how these few layers are arranged.

One convention runs through all four forms: **a head's output shape equals the
shape of that stream's raw input tensor**. The video head emits
``[B, C, F, H, W]`` (the unflatten happens inside the head), the action head
emits ``[B, T, D]``. L3's regression target therefore has the same shape as the
head's output, and the loss step needs no further shape shuffling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ..geometry import GeometrySpec
from ..streams.blocks.wan_dit import FP32LayerNorm, modulate
from .spec import HeadSpec

if TYPE_CHECKING:          # typing only; not imported at runtime, to avoid a
                           # heads <-> spec import cycle
    from ..spec.world import WorldModelSpec

__all__ = ["IdentityHead", "LinearHead", "NormLinearHead", "PatchFlowHead",
           "build_flow_head"]


class _HeadBase(nn.Module):
    """Shared parts of the four forms: exit width, entry width, timestep check.

    ``input_width`` is not a field but is **looked up**: it equals the source
    stream's ``residual_width`` (the head reads exactly that stream's hidden
    state). Making it a field would create a second source of truth, and when
    the stream width changes only one of the two would get updated.
    """

    def __init__(self, *, spec: HeadSpec, geometry: GeometrySpec,
                 world: WorldModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.geometry = geometry
        self.input_width = world.stream(spec.source_stream).residual_width

    @property
    def output_width(self) -> int:
        """Attribute required by the ``OutputHead`` protocol.

        ``HeadSpec`` is its single source.
        """
        return self.spec.output_width

    def _check_timestep_embedding(self, timestep_embedding: Tensor | None) -> Tensor | None:
        """``needs_timestep_embedding`` and the embedding actually received must agree.

        What arrives is the timestep **embedding** -- the ``[B, residual_width]``
        tensor that the source stream's ``prepare`` stored into
        ``extras["timestep_embedding"]`` -- not the scalar in ``[0, 1]``. The ``t`` of the
        existing ``Head.forward(x, t)`` is that embedding too
        (``wan_video_dit.py:289-298`` adds it straight onto ``modulation``).
        The two have different widths, and mixing them up broadcasts to a wrong
        shape at the addition instead of raising.

        Args:
            timestep_embedding: The timestep embedding, or None.

        Returns:
            ``timestep_embedding`` unchanged.

        Raises:
            ValueError: If presence or shape disagrees with ``needs_timestep_embedding``.
        """
        if self.spec.needs_timestep_embedding:
            if timestep_embedding is None:
                raise ValueError(
                    f"head {self.spec.name!r} has needs_timestep_embedding=True, "
                    "but forward received no timestep embedding"
                )
            if (timestep_embedding.ndim != 2
                    or timestep_embedding.shape[-1] != self.input_width):
                raise ValueError(
                    f"head {self.spec.name!r} consumes the timestep "
                    f"**embedding** [B, {self.input_width}] (the source "
                    f"stream's extras['timestep_embedding']), got "
                    f"{tuple(timestep_embedding.shape)}"
                )
        elif timestep_embedding is not None:
            raise ValueError(
                f"head {self.spec.name!r} has needs_timestep_embedding=False, but got a "
                "timestep; it has no modulation slot to consume it"
            )
        return timestep_embedding

    def _check_hidden(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.input_width:
            raise ValueError(
                f"head {self.spec.name!r} consumes the hidden state of source "
                f"stream {self.spec.source_stream!r} "
                f"[B, L, {self.input_width}], got {tuple(hidden.shape)}"
            )
        return hidden


class IdentityHead(_HeadBase):
    """Parameter-free head: returns the hidden state as is.

    The existing ``semantic_future`` is exactly this -- that self-distillation
    term computes ``1 - cos`` directly on the 3072-dim raw hidden state
    (``losses.py:58-72``), with **no** projection layer. ``output_width`` must
    therefore equal the source stream's ``residual_width``; if it does not the
    declaration is wrong and construction fails.
    """

    def __init__(self, *, spec: HeadSpec, geometry: GeometrySpec,
                 world: WorldModelSpec) -> None:
        super().__init__(spec=spec, geometry=geometry, world=world)
        if spec.output_width != self.input_width:
            raise ValueError(
                f"head {spec.name!r} has form 'identity' (no projection "
                f"layer), so output_width must equal source stream "
                f"{spec.source_stream!r} residual_width="
                f"{self.input_width}, got {spec.output_width}"
            )

    def forward(self, hidden: Tensor, *, timestep_embedding: Tensor | None = None,
                conditions: Mapping[str, Any] | None = None) -> Tensor:
        del conditions
        self._check_timestep_embedding(timestep_embedding)
        return self._check_hidden(hidden)


class LinearHead(_HeadBase):
    """A bare ``Linear``. The existing ``self.head`` at ``action_dit.py:150``.

    The action head has **no modulation**: ``post_dit`` (``:423``) is the single
    line ``self.head(tokens)``. The ``ActionHead`` at ``:20-33`` of the same
    file, which does carry a norm and two modulation slots, is never
    instantiated (``ActionDiT.__init__`` builds an ``nn.Linear``), so it is dead
    code and not the reference for this head.
    """

    def __init__(self, *, spec: HeadSpec, geometry: GeometrySpec,
                 world: WorldModelSpec) -> None:
        super().__init__(spec=spec, geometry=geometry, world=world)
        self.head = nn.Linear(self.input_width, spec.output_width)

    def forward(self, hidden: Tensor, *, timestep_embedding: Tensor | None = None,
                conditions: Mapping[str, Any] | None = None) -> Tensor:
        del conditions
        self._check_timestep_embedding(timestep_embedding)
        return self.head(self._check_hidden(hidden))


class NormLinearHead(_HeadBase):
    """``LayerNorm`` + ``Linear``. InternW0-Δ's existing ``future_delta``.

    The existing code writes it as one ``nn.Sequential``
    (``wan_video_dit.py:557-563``), so the keys are ``0.*`` / ``1.*``. Here it
    is split into two named members, ``norm`` and ``head``: once each stage has
    a name, porting weights is one definite rename instead of "is index 0 the
    norm or the linear", which can only be settled by reading the source.
    """

    def __init__(self, *, spec: HeadSpec, geometry: GeometrySpec,
                 world: WorldModelSpec, eps: float = 1e-6) -> None:
        super().__init__(spec=spec, geometry=geometry, world=world)
        self.norm = nn.LayerNorm(self.input_width, eps=eps)
        self.head = nn.Linear(self.input_width, spec.output_width)

    def forward(self, hidden: Tensor, *, timestep_embedding: Tensor | None = None,
                conditions: Mapping[str, Any] | None = None) -> Tensor:
        del conditions
        self._check_timestep_embedding(timestep_embedding)
        return self.head(self.norm(self._check_hidden(hidden)))


class PatchFlowHead(_HeadBase):
    """Video flow head: ``norm`` + two-slot modulation + ``head`` + unflatten.

    The existing ``Head`` at ``wan_video_dit.py:279-298`` plus ``unpatchify`` at
    ``:405-416``. Three details each worth a note:

    1. ``norm`` has **elementwise_affine=False** (``:285``): the affine part is
       carried by the two modulation slots, and enabling both would give two
       equivalent degrees of freedom.
    2. ``head``'s output width is ``output_width × prod(patch_size)``, while
       ``HeadSpec.output_width`` records only the channel count **after** the
       unflatten. The ``× prod(patch_size)`` stays inside the head; otherwise
       one field would carry two meanings.
    3. ``modulation`` is initialized as ``randn / dim**0.5`` (``:287``), not
       zero. Switching to zero init would make the first step's output always
       ``head(norm(x))``, which differs from the behavior of the existing
       weights by a constant term.
    """

    def __init__(self, *, spec: HeadSpec, geometry: GeometrySpec,
                 world: WorldModelSpec, eps: float = 1e-6) -> None:
        super().__init__(spec=spec, geometry=geometry, world=world)
        slots = spec.modulation_slots
        if slots.names != ("shift_out", "scale_out"):
            raise ValueError(
                f"head {spec.name!r} has form 'patch_flow', which accepts "
                f"only two-slot modulation (shift_out, scale_out), got "
                f"{list(slots.names)}"
            )
        self.patch_size = tuple(geometry.patch_size)
        self.norm = FP32LayerNorm(self.input_width, eps=eps,
                                  elementwise_affine=False)
        self.head = nn.Linear(self.input_width,
                              spec.output_width * math.prod(self.patch_size))
        self.modulation = nn.Parameter(
            torch.randn(1, slots.width, self.input_width)
            / self.input_width ** 0.5)

    def forward(self, hidden: Tensor, *, timestep_embedding: Tensor | None = None,
                conditions: Mapping[str, Any] | None = None) -> Tensor:
        del conditions
        timestep_embedding = self._check_timestep_embedding(timestep_embedding)
        assert timestep_embedding is not None            # guaranteed by needs_timestep_embedding=True
        self._check_hidden(hidden)
        slots = self.spec.modulation_slots.unpack(
            self.modulation + timestep_embedding.unsqueeze(1), slot_axis=1)
        tokens = self.head(modulate(self.norm(hidden),
                                    slots["shift_out"].unsqueeze(1),
                                    slots["scale_out"].unsqueeze(1)))
        return self._unpatchify(tokens)

    def _unpatchify(self, tokens: Tensor) -> Tensor:
        """``[B, L, out × prod(patch)]`` → ``[B, out, F, H, W]``.

        Alignment padding is cut off before the unflatten: the tokens that
        ``padded_seq_len`` adds on top of ``video_seq_len`` have no place in the
        grid (``build_labels`` labels them ``frame=row=col=-1``).

        The flattening order is the strict inverse of the ``flatten(2)`` at the
        entry: the last dim runs as (pt, ph, pw, C) and the token dim runs as
        (f, h, w) in C order. The existing code uses one einops call
        (``:407-416``, ``b (f h w) (x y z c) -> b c (f x) (h y) (w z)``); here
        it is explicit view/permute instead. One more dependency is not worth
        it, and getting the order wrong here is **silent**: the shape matches
        exactly, it just stirs the time axis into the spatial axes.

        Args:
            tokens: Head output of shape ``[B, L, out × prod(patch)]``.

        Returns:
            Tensor of shape ``[B, out, F, H, W]``.

        Raises:
            ValueError: If fewer tokens arrive than ``video_seq_len``.
        """
        frames, rows, cols = self.geometry.patch_grid
        real = self.geometry.video_seq_len
        if tokens.shape[1] < real:
            raise ValueError(
                f"head {self.spec.name!r} got {tokens.shape[1]} tokens, "
                f"fewer than video_seq_len={real}"
            )
        patch_t, patch_h, patch_w = self.patch_size
        batch = tokens.shape[0]
        grid = tokens[:, :real].view(batch, frames, rows, cols,
                                     patch_t, patch_h, patch_w,
                                     self.spec.output_width)
        grid = grid.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return grid.reshape(batch, self.spec.output_width,
                            frames * patch_t, rows * patch_h, cols * patch_w)


_FORMS: dict[str, type[_HeadBase]] = {
    "identity": IdentityHead,
    "linear": LinearHead,
    "norm_linear": NormLinearHead,
    "patch_flow": PatchFlowHead,
}


def build_flow_head(*, spec: HeadSpec, geometry: GeometrySpec,
                    world: WorldModelSpec, form: str,
                    **params: Any) -> _HeadBase:
    """Build one output head by form name. ``form`` comes from ``Spec.params``.

    There is deliberately no default: the four forms take different numbers of
    parameters, and the symptom of guessing wrong is missing or unexpected keys
    during a strict load, by which point you are far away from the place that
    declared it.

    The output width is **not** read from ``params``: it is already
    ``HeadSpec.output_width``, and a second copy here could disagree with the
    one the L4 check (V8) validates against the regression target.

    Args:
        spec: The head declaration.
        geometry: Geometry, needed by ``patch_flow`` for the patch grid.
        world: The world-model spec, used to look up the source stream's width.
        form: One of ``identity`` / ``linear`` / ``norm_linear`` /
            ``patch_flow``.
        **params: Extra keyword arguments forwarded to the form's constructor.

    Returns:
        The constructed head module.

    Raises:
        ValueError: If ``form`` is not a known form name.
    """
    try:
        cls = _FORMS[form]
    except KeyError:
        raise ValueError(
            f"head {spec.name!r} has unknown params['form']={form!r}; "
            f"available forms are {sorted(_FORMS)}"
        ) from None
    return cls(spec=spec, geometry=geometry, world=world, **params)
