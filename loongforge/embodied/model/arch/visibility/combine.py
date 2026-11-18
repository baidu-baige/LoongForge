# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 visibility expressions and the four combinators (Lane A).

The signature follows ``docs/design/mot_layer_interfaces.zh-CN.md`` §3.3.

The set of combinators is closed at four, and each implementation **must be an
elementwise operation on bool tensors**. That is the precondition for
materializing the same predicate onto both the dense backend and FlexAttention's
score_mod. Any non-elementwise operator (sort, cumsum, scatter) kills the
score_mod path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..layout import LabelView, TokenLayout

if TYPE_CHECKING:  # avoid a combine <-> mask import cycle
    from .mask import CompiledMask

__all__ = ["Predicate", "Visibility", "from_predicate", "ALWAYS_VISIBLE"]


# Type of a predicate function: given the label views of the query side and the
# key side, return an [Lq, Lk] bool tensor.
# Key design point: it is a **predicate over per-token labels**, not interval
# arithmetic. The whole reason for choosing label composition over an interval
# algebra: an interval algebra has no slot to put "noise level" in.
Predicate = Callable[[LabelView, LabelView], Tensor]


@dataclass(frozen=True)
class Visibility:
    """A visibility expression. Holds no tensor; materializes only in ``compile``.

    ``expr`` exists only for error messages and test readability: a composed
    expression has to be able to tell a human **which rule** made a backend
    unable to express it, when V9 negotiation fails.

    Attributes:
        predicate: The elementwise predicate over label views.
        expr: Human-readable rendering of the expression.
    """

    predicate: Predicate = field(repr=False)
    expr: str = "<anonymous>"

    # ---- the four combinators, all elementwise ----

    def __and__(self, other: Visibility) -> Visibility:
        """Visible only where both rules hold."""
        return Visibility(
            lambda q, k: self.predicate(q, k) & other.predicate(q, k),
            f"({self.expr} & {other.expr})",
        )

    def __or__(self, other: Visibility) -> Visibility:
        """Visible where either rule holds."""
        return Visibility(
            lambda q, k: self.predicate(q, k) | other.predicate(q, k),
            f"({self.expr} | {other.expr})",
        )

    def __invert__(self) -> Visibility:
        """Negation."""
        return Visibility(
            lambda q, k: ~self.predicate(q, k),
            f"~{self.expr}",
        )

    def restrict(self, *, where: Visibility) -> Visibility:
        """Apply this rule only on the (q, kv) region ``where`` selects.

        Outside that region the result is **left as it was**. Difference from
        ``& where``: ``&`` turns the unselected region invisible, ``restrict``
        does not change it. This semantics is what the T7 composition-law test
        checks.

        Args:
            where: Region selector; the rule applies only where it holds.

        Returns:
            A new ``Visibility``.
        """
        return Visibility(
            lambda q, k: (~where.predicate(q, k)) | self.predicate(q, k),
            f"{self.expr}.restrict({where.expr})",
        )

    # ---- materialization ----

    def evaluate(self, layout: TokenLayout) -> Tensor:
        """Evaluate on a layout, giving a ``[total, total]`` bool tensor.

        This is where "composition is closed" is implemented: ``&`` / ``|`` /
        ``~`` / ``restrict`` only compose two predicates into one predicate, and
        evaluation runs the whole thing in one pass without materializing any
        intermediate tensor.

        Hard invariant (master plan INV-1): the result is exactly 2-D and
        carries no batch dimension. Per-sample valid bits travel on the separate
        ``AttentionPlan.key_valid`` channel.

        Args:
            layout: The token layout to evaluate against.

        Returns:
            ``[total, total]`` bool tensor.

        Raises:
            ValueError: If the predicate returns a non-bool dtype or a shape
                other than ``(total, total)``.
        """
        q_view, k_view = layout.label_views()
        out = self.predicate(q_view, k_view)
        expected = (layout.total, layout.total)
        if out.dtype != torch.bool:
            raise ValueError(
                f"visibility {self.expr} evaluated to dtype={out.dtype}, "
                "must be bool"
            )
        if tuple(out.shape) != expected:
            raise ValueError(
                f"visibility {self.expr} evaluated to shape {tuple(out.shape)} "
                f"!= {expected} (structural carries no batch dim, INV-1)"
            )
        return out

    def compile(self, layout: TokenLayout, *, backend: str) -> CompiledMask:
        """Materialize on a layout. ``backend`` picks the materialization path.

        Args:
            layout: The token layout to evaluate against.
            backend: Target attention backend name.

        Returns:
            A ``CompiledMask`` holding the evaluated structural channel.
        """
        from .mask import CompiledMask

        return CompiledMask(structural=self.evaluate(layout),
                            layout=layout, backend=backend)


def from_predicate(expr: str) -> Callable[[Predicate], Visibility]:
    """Wrap a predicate function into a ``Visibility``.

    Used by the nine rules in ``rules.py``.

    Args:
        expr: Human-readable rendering of the rule.

    Returns:
        A decorator turning a ``Predicate`` into a ``Visibility``.
    """

    def wrap(predicate: Predicate) -> Visibility:
        return Visibility(predicate, expr)

    return wrap


ALWAYS_VISIBLE = Visibility(
    lambda q, k: torch.ones((q.layout.total, k.layout.total), dtype=torch.bool),
    "always_visible",
)
"""Fully visible. Rules degrade to this when a label is absent, rather than
raising (the default semantics of §3.2)."""
