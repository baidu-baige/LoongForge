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

from collections.abc import Callable, Mapping
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
    runtime_labels: frozenset[str] = frozenset()

    # ---- the four combinators, all elementwise ----

    def __and__(self, other: Visibility) -> Visibility:
        """Visible only where both rules hold."""
        return Visibility(
            lambda q, k: self.predicate(q, k) & other.predicate(q, k),
            f"({self.expr} & {other.expr})",
            self.runtime_labels | other.runtime_labels,
        )

    def __or__(self, other: Visibility) -> Visibility:
        """Visible where either rule holds."""
        return Visibility(
            lambda q, k: self.predicate(q, k) | other.predicate(q, k),
            f"({self.expr} | {other.expr})",
            self.runtime_labels | other.runtime_labels,
        )

    def __invert__(self) -> Visibility:
        """Negation."""
        return Visibility(
            lambda q, k: ~self.predicate(q, k),
            f"~{self.expr}",
            self.runtime_labels,
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
            self.runtime_labels | where.runtime_labels,
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

    def evaluate_runtime(self, layout: TokenLayout,
                         labels: Mapping[str, Tensor]) -> Tensor:
        """Evaluate batch labels without changing the static layout.

        Runtime values are [B, L]. Boolean composition is evaluated as a whole:
        a noise gate scoped to a cross-stream edge must not mask unrelated
        self-attention edges. The result is [B, L, L], never a structural mask.
        """
        missing = self.runtime_labels - labels.keys() - layout.labels.keys()
        if missing:
            raise ValueError(f"visibility {self.expr} requires runtime labels {sorted(missing)}")
        batch = None
        device = None
        for name, value in labels.items():
            if value.ndim != 2 or value.shape[1] != layout.total:
                raise ValueError(f"runtime label {name!r} must be [B, {layout.total}]")
            if batch is not None and (value.shape[0] != batch or value.device != device):
                raise ValueError("runtime labels must share batch size and device")
            batch, device = value.shape[0], value.device
        if batch is None or batch == 0:
            raise ValueError("runtime labels must contain a non-empty batch")
        # Static predicates also participate in the expression, on the same
        # device as the batch labels.
        local = TokenLayout(
            layout.segments,
            {key: value.to(device) for key, value in layout.labels.items()},
            regions={name: membership.to(device)
                     for name, membership in layout.regions.items()},
        )
        out = self.predicate(_RuntimeLabelView(local, "query", labels),
                             _RuntimeLabelView(local, "key", labels))
        if out.dtype != torch.bool:
            raise ValueError("runtime visibility must evaluate to bool")
        return torch.broadcast_to(out, (batch, layout.total, layout.total))

    def compile(self, layout: TokenLayout, *, backend: str) -> CompiledMask:
        """Materialize on a layout. ``backend`` picks the materialization path.

        Args:
            layout: The token layout to evaluate against.
            backend: Target attention backend name.

        Returns:
            A ``CompiledMask`` holding the evaluated structural channel.
        """
        from .mask import CompiledMask

        missing = self.runtime_labels - layout.labels.keys()
        if missing:
            raise ValueError(
                f"cannot compile runtime labels {sorted(missing)} as a static mask; "
                "use AttentionPlan.from_visibility(...).bind(labels=...)"
            )
        return CompiledMask(structural=self.evaluate(layout),
                            layout=layout, backend=backend)


class _RuntimeLabelView:
    """LabelView-compatible lookup with batched runtime overrides."""

    def __init__(self, layout, side, labels):
        self.layout, self.side, self.labels = layout, side, labels

    def has(self, key):
        return key in self.labels or self.layout.has_label(key)

    def get(self, key):
        if key in self.labels:
            value = self.labels[key]
            return value.unsqueeze(2 if self.side == "query" else 1)
        value = self.layout.labels_of(key)
        return value.unsqueeze(1 if self.side == "query" else 0)

    def positions(self):
        device = next(iter(self.labels.values())).device
        value = torch.arange(self.layout.total, device=device)
        return value.unsqueeze(1 if self.side == "query" else 0)


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
    lambda q, k: torch.ones((q.layout.total, k.layout.total), dtype=torch.bool,
                            device=q.positions().device),
    "always_visible",
)
"""Fully visible. Rules degrade to this when a label is absent, rather than
raising (the default semantics of §3.2)."""
