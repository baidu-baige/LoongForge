# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Condition encoders (L2): implementations and builders for
``ConditionSpec.encoder``.

Why they are a separate parameter group instead of living inside a stream:
one condition can be injected into several streams at once (VPP2 proprio
feeds both streams), while the existing checkpoint holds exactly one encoder
weight set (``vpp2.py:117-121`` builds one ``nn.Linear``, and
``vpp2.py:1800-1801`` stores a single top-level payload key that is not under
the ``mot`` subtree). Hanging it under a stream would force one copy per
stream, which is a different model; naming an arbitrary stream as the "host"
would break the whole chain whenever that stream is switched off.

So L2 has two parameter namespaces rather than one: ``blocks.<stream>.*`` and
``condition_encoders.<condition>.*``. The L4 namespace ``heads.<head>.*`` is
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from ..geometry import GeometrySpec
from .conditions import ConditionSpec

if TYPE_CHECKING:          # typing only; not imported at runtime so that
                           # spec and streams do not form an import cycle
    from ..spec.world import WorldModelSpec

__all__ = ["TokenEncoder", "linear_token_encoder"]


class TokenEncoder(nn.Linear):
    """A ``Linear`` encoding ``[B, D]`` or ``[B, T, D]`` input into tokens.

    It subclasses ``nn.Linear`` deliberately instead of wrapping one in a
    submodule: the existing weights live in ``payload["proprio_encoder"]``
    whose keys are exactly ``{weight, bias}``. Wrapping would turn them into
    ``encode.weight`` and need an extra remapping rule whose only reason for
    existing would be the extra wrapper.

    A 2-D input gets a token axis added, matching the existing
    ``proprio.unsqueeze(1)`` at ``vpp2.py:531-533``. The axis is inserted at
    position 1, not last: inserting last would also pass through Linear when
    D == 1 and would silently produce a wrong token count.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Encode raw condition input into tokens.

        Args:
            x: ``[B, D]`` or ``[B, T, D]``, last axis equal to
                ``in_features`` (that is, ``ConditionSpec.source_width``).

        Returns:
            ``[B, T, out_features]``, with T == 1 for a 2-D input.

        Raises:
            ValueError: If the rank is not 2 or 3, or if the last axis does
                not match ``in_features``.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(
                f"TokenEncoder takes [B, D] or [B, T, D], got "
                f"{tuple(x.shape)}"
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"TokenEncoder input width is {self.in_features} "
                f"(ConditionSpec.source_width), got {x.shape[-1]}"
            )
        return super().forward(x)


def linear_token_encoder(*, spec: ConditionSpec, geometry: GeometrySpec,
                         world: WorldModelSpec) -> nn.Module:
    """Build a Linear from ``source_width`` to the host slot's ``width``.

    The output width is not configurable: it must equal the input width of
    the host slot's condition projector, and that number already lives at
    ``world.condition(spec.append_to).width``. Configuring it again would
    create a second source of truth, and the symptom of a mismatch would be
    a cat shape error during forward, far away from the declaration.

    Args:
        spec: Condition spec being built. Must be APPEND_TO_SLOT and must
            declare ``source_width``.
        geometry: Unused; the parameter exists so that encoder builders have
            the same signature as stream and head builders.
        world: World model spec, used to read the host slot's width.

    Returns:
        A ``TokenEncoder`` of shape ``source_width -> host.width``.

    Raises:
        NotImplementedError: If the condition has an encoder but is not
            APPEND_TO_SLOT, since no other injection kind has a defined
            output width source yet.
        ValueError: If ``source_width`` is missing, or if the host slot
            declares no ``width``.
    """
    del geometry
    if spec.append_to is None:
        raise NotImplementedError(
            f"condition {spec.name!r} has an encoder but is not "
            "APPEND_TO_SLOT; linear_token_encoder derives its output width "
            "only from the host slot, and no other injection kind has a "
            "defined output width source"
        )
    if spec.source_width is None:
        raise ValueError(
            f"condition {spec.name!r} has an encoder and must give "
            "source_width, the input width of this Linear; it cannot be "
            "derived from anything else"
        )
    host = world.condition(spec.append_to)
    if host.width is None:
        raise ValueError(
            f"condition {spec.name!r} appends to slot {spec.append_to!r}, "
            "but that condition declares no width; the encoder's output "
            "width has nowhere to come from"
        )
    return TokenEncoder(spec.source_width, host.width)
