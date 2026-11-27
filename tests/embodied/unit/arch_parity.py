# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pieces the three ``test_arch_*_parity.py`` harnesses share.

Not a test module: it holds no cases and is never collected. It exists because
each harness has to skip for its own reason (one needs an external checkout,
one needs a GPU), so they cannot import each other -- a module-level skip in
one would take the others with it.

Only what is genuinely model independent lives here: how a mismatch is
reported, one spec surgery two harnesses need, and the two spec readings the
mask comparison takes. Building the legacy side stays in each harness, because
the three legacy sources agree on nothing.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import torch

from loongforge.embodied.model.arch.builder import join_stream_visibility
from loongforge.embodied.model.arch.spec.world import WorldModelSpec
from loongforge.embodied.model.arch.visibility.mask import AttentionPlan
from loongforge.embodied.model.arch.visibility.phases import CONDITION


def assert_identical(got: torch.Tensor, want: torch.Tensor,
                     what: str) -> None:
    """Assert bit equality, reporting the largest deviation when it fails.

    Equality rather than ``allclose`` because both sides issue the same
    primitives in the same order; the magnitude in the message is what tells a
    reordering (around 1e-7) from a wrong tensor (order 1).

    Args:
        got: The tensor produced by this architecture.
        want: The tensor produced by the legacy source.
        what: Name of the quantity, used in the failure message.

    Raises:
        AssertionError: Shapes differ, or any element differs.
    """
    assert tuple(got.shape) == tuple(want.shape), \
        f"{what}: shape {tuple(got.shape)} vs {tuple(want.shape)}"
    if torch.equal(got, want):
        return
    raise AssertionError(
        f"{what} differs, max abs "
        f"{(got - want).abs().max().item():.3e}")


def deviation(got: torch.Tensor, want: torch.Tensor) -> float:
    """Largest absolute deviation, as a float, for messages and thresholds."""
    return (got.detach().float() - want.detach().float()).abs().max().item()


def assert_close(got: torch.Tensor, want: torch.Tensor, what: str, *,
                 atol: float) -> None:
    """Assert agreement up to ``atol``, reporting the deviation either way.

    The tolerance form is for the side that cannot be bit exact: a fused or
    kernel-level implementation computes the same mathematical expression in a
    different order, so the two agree only to rounding. Every caller has to
    name the kernel that forces it.

    Args:
        got: The tensor produced by this architecture.
        want: The tensor produced by the legacy source.
        what: Name of the quantity, used in the failure message.
        atol: Largest absolute deviation accepted.

    Raises:
        AssertionError: Shapes differ, or the deviation exceeds ``atol``.
    """
    assert tuple(got.shape) == tuple(want.shape), \
        f"{what}: shape {tuple(got.shape)} vs {tuple(want.shape)}"
    worst = deviation(got, want)
    if worst > atol:
        raise AssertionError(
            f"{what} differs by {worst:.3e}, over the {atol:.1e} allowed")


def assert_same_mask(got: torch.Tensor, want: torch.Tensor,
                     what: str) -> None:
    """Assert two bool visibility matrices agree cell for cell.

    Separate from ``assert_identical`` for two reasons. Bool tensors cannot be
    subtracted, so the magnitude that function reports is unavailable; and the
    useful report for a mask is not a magnitude but which way it went. Too
    strict (closed here, open there) and too permissive are different bugs:
    the first only costs accuracy, the second leaks information the model is
    not supposed to have, and both leave every shape intact.

    Args:
        got: The matrix this architecture compiled.
        want: The matrix the legacy builder produced.
        what: Name of the comparison, used in the failure message.

    Raises:
        AssertionError: Shapes differ, or any cell differs.
    """
    assert tuple(got.shape) == tuple(want.shape), \
        f"{what}: shape {tuple(got.shape)} vs {tuple(want.shape)}"
    if torch.equal(got, want):
        return
    rows = torch.nonzero((got != want).any(1)).flatten().tolist()
    raise AssertionError(
        f"{what}: of {got.numel()} cells, {int((~got & want).sum())} are too "
        f"strict here and {int((got & ~want).sum())} too permissive, in "
        f"query rows {rows[:8]}{' ...' if len(rows) > 8 else ''}")


def structural_mask(spec: WorldModelSpec) -> torch.Tensor:
    """Compile a spec's own visibility declaration to its ``[S, S]`` matrix.

    The same two calls step 2 of ``WorldModel.__init__`` makes, without
    allocating any stream: a mask comparison needs the declaration and the
    layout and nothing else, which is what lets the parametrized cases afford
    one spec per configuration. Each harness also has one case tying this back
    to the ``model.plans[0]`` its forwards really run under.

    Args:
        spec: The spec whose ``visibility`` and ``token_labels`` to compile.

    Returns:
        The structural visibility, ``[total, total]`` bool and batch free.
    """
    return AttentionPlan.from_visibility(
        join_stream_visibility(dict(spec.visibility)),
        spec.build_layout()).structural


def condition_frames(spec: WorldModelSpec) -> int:
    """Recover the condition-frame count from the ``phase`` label.

    A provider takes the count as an argument and turns it into the label; the
    legacy mask builders take it back as an argument. Reading it off the label
    keeps the comparison on one number, rather than restating a provider
    default beside the builder call where the two could drift apart.

    Args:
        spec: A spec whose provider declares ``token_labels``.

    Returns:
        How many leading video frames carry a condition phase.

    Raises:
        ValueError: The spec declares no ``phase`` label.
    """
    layout = spec.build_layout()
    if not layout.has_label("phase"):
        raise ValueError(
            f"{spec.model_type} declares no phase label, so the condition "
            "frame count cannot be read back off its layout")
    video = layout.range_of("video")
    phase = layout.labels_of("phase")[video.start:video.stop]
    wanted = torch.tensor([int(value) for value in CONDITION])
    hits = int((phase[:, None] == wanted[None, :]).any(1).sum())
    return hits // spec.geometry.tokens_per_frame


def drop_heads(spec: WorldModelSpec, names: Iterable[str]) -> WorldModelSpec:
    """Remove heads from a spec, together with the loss terms that name them.

    The loss terms have to go too: V6 requires every ``LossSpec.head`` to name
    a head that exists, so dropping only the head produces a spec that fails
    validation rather than a smaller model.

    Args:
        spec: The provider spec to shrink.
        names: Head names to remove. Absent names are ignored.

    Returns:
        A copy with those heads and their loss terms gone.
    """
    dropped = frozenset(names)
    return replace(
        spec,
        heads=tuple(head for head in spec.heads if head.name not in dropped),
        losses=tuple(loss for loss in spec.losses if loss.head not in dropped),
    )
