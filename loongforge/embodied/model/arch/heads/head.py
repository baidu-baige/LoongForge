# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L4 output head protocol ``OutputHead`` (Lane C).

The signature follows ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.6.

A head is an ``nn.Module`` (it owns parameters), but the engine never sees it:
heads are called in step 6 of ``WorldModel.forward``, with the tensor pulled out
by ``HiddenStateRef.resolve`` as input. That is exactly how heads and streams
stay decoupled: **a head does not take part in the joint attention**.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from torch import Tensor

__all__ = ["OutputHead"]


class OutputHead(Protocol):
    """One output head. Two responsibilities: compute the output, declare its width.

    ``forward``'s ``conditions`` is a read-only mapping, there for heads that
    need conditions (action heads usually want state / proprio). Heads that do
    not need it can just ignore it: an empty mapping is the normal case, so no
    branching is required.
    """

    def forward(self, hidden: Tensor, *,
                timestep_embedding: Tensor | None = None,
                conditions: Mapping[str, Any] | None = None) -> Tensor:
        """Map a hidden state to an output.

        Args:
            hidden: Hidden state taken from the source stream.
            timestep_embedding: Non-None only when the head modulates on its own (the
                action head's two-slot modulation, see
                ``streams.spec.HEAD_SLOTS``).
            conditions: Read-only condition mapping; heads may ignore it.

        Returns:
            The head's output tensor.
        """
        ...

    @property
    def output_width(self) -> int:
        """Width of the output's last dimension.

        A construction-time constant; V8 reconciles it against ``HeadSpec``.
        """
        ...
