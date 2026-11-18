# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The two assembly-layer facts that need no ``WorldModel`` to check (the free
functions in ``builder.py``).

Joint visibility composition is checked against the three real providers in
``test_arch_providers.py``. This file checks condition-bus assembly plus the two
paths that **raise on purpose**: they raise because a design decision has not
landed yet, not because an implementation is missing, so they need tests to pin
them down as well.
"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.builder import materialize_conditions
from loongforge.embodied.model.arch.spec.core import Spec
from loongforge.embodied.model.arch.streams.conditions import (
    ConditionSpec, InjectionKind)


def text(**extra) -> ConditionSpec:
    """A cross-attention text condition read from the ``context`` batch key."""
    kwargs = dict(name="text", kind=InjectionKind.CROSS_ATTENTION,
                  targets=("video", "action"), source_key="context")
    kwargs.update(extra)
    return ConditionSpec(**kwargs)


def batch(**extra) -> dict:
    """A batch holding one ``[2, 7, 16]`` context tensor."""
    out = {"context": torch.zeros((2, 7, 16))}
    out.update(extra)
    return out


def test_batch_sourced_condition_lands_on_its_targets():
    """A condition with no encoder passes the batch tensor through unchanged and
    appears on exactly the streams listed in ``targets``."""
    bus = materialize_conditions([text()], batch(), training=True)
    assert len(bus) == 1
    condition = bus.require("text")
    assert condition.kind is InjectionKind.CROSS_ATTENTION
    assert torch.equal(condition.value, batch()["context"])
    assert set(bus.for_stream("video")) == {"text"}
    assert set(bus.for_stream("action")) == {"text"}
    # A stream absent from ``targets`` gets an empty dict rather than an error,
    # which is the normal case for a single-stream model
    assert bus.for_stream("vl") == {}


def test_disabled_condition_takes_no_name_on_the_bus():
    """A disabled condition is skipped entirely.

    Leaving a zero tensor behind would make the V7 name reconciliation count it
    as present.
    """
    bus = materialize_conditions([text(enabled=False)], batch(), training=True)
    assert len(bus) == 0
    assert bus.declared_names() == frozenset()


def test_missing_source_key_lists_the_batch_keys():
    """A ``source_key`` absent from the batch raises and names the keys that are
    present, which is what tells a data-pipeline mismatch from a typo."""
    with pytest.raises(KeyError, match="but batch only has"):
        materialize_conditions([text()], {"video": torch.zeros(1)},
                               training=True)


def test_two_conditions_sharing_a_name_are_rejected():
    """Condition names are the bus keys, so a duplicate name is rejected instead
    of silently overwriting the earlier entry."""
    with pytest.raises(ValueError, match="same-name conditions must not overwrite"):
        materialize_conditions([text(), text()], batch(), training=True)


def test_encoder_bearing_condition_runs_its_encoder_on_the_raw_material():
    """A condition carrying an ``encoder`` yields
    ``value = encoders[name](batch[source_key])``.

    The encoder is **not** built inside ``materialize_conditions``: it holds
    parameters, so it has to live in ``WorldModel``'s submodule tree to be seen
    by the optimizer and the checkpoint. That is why it is passed in by name
    here, turning a 16-wide source into the 5-wide value asserted below.
    """
    spec = text(encoder=Spec(impl=torch.nn.Identity), source_width=16)
    encoder = torch.nn.Linear(16, 5, bias=False)
    bus = materialize_conditions([spec], batch(), training=True,
                                 encoders={"text": encoder})
    assert tuple(bus.require("text").value.shape) == (2, 7, 5)


def test_encoder_bearing_condition_without_its_module_raises():
    """Declaring an encoder but not passing one raises, instead of degrading to
    "use the raw source tensor as the value".

    The degraded form shows up as a shape error in some later ``cat`` or
    ``Linear`` (source width != host slot width), far away from the question of
    who was supposed to build that encoder.
    """
    spec = text(encoder=Spec(impl=torch.nn.Identity), source_width=16)
    with pytest.raises(KeyError, match="but encoders does not hold it"):
        materialize_conditions([spec], batch(), training=True)


def test_dropout_is_an_open_ruling_only_while_training():
    """Condition dropout is not implemented: a non-zero ``dropout_prob`` raises
    in training and is a no-op in eval.

    Raising rather than ignoring the field keeps a config that asks for dropout
    from training as if it had been applied.
    """
    spec = text(dropout_prob=0.1)
    with pytest.raises(NotImplementedError, match="the semantics of CFG condition dropout"):
        materialize_conditions([spec], batch(), training=True)
    # No dropout at eval time, so the same declaration materialises as usual
    assert len(materialize_conditions([spec], batch(), training=False)) == 1
