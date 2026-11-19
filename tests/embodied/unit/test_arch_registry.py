# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 legacy intake: the stream registry and two-channel split of old masks.

What matters about ``plan_from_legacy_mask`` is **which shapes it rejects**:
its entire value is keeping a per-sample ``[B, 1, L, L]`` mask out of the
batch-shared ``structural`` channel. Letting one in raises nothing; it only
kills BlockMask reuse across the batch, drops the whole path back to dense,
and the loss keeps going down.
"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.layout import SegmentRange, TokenLayout
from loongforge.embodied.model.arch.streams.registry import (
    StreamRegistry, plan_from_legacy_mask)

TOTAL = 6


def layout() -> TokenLayout:
    """Six tokens: four video then two action, with a stream_id label."""
    return TokenLayout(
        segments=(SegmentRange("video", 0, 4), SegmentRange("action", 4, 6)),
        labels={"stream_id": torch.tensor([0, 0, 0, 0, 1, 1])})


def test_registry_rejects_duplicate_names():
    """Registering one name twice raises instead of overwriting the first.

    A silent overwrite would make the stream implementation depend on module
    import order.
    """
    registry = StreamRegistry()

    @registry.register("toy")
    def build_toy():  # pragma: no cover - only the registration is checked
        return None

    assert registry.names() == ("toy",)
    assert registry.get("toy") is build_toy
    with pytest.raises(ValueError, match="duplicate stream implementation name"):
        registry.register("toy")(build_toy)


def test_registry_lists_names_on_miss():
    """A lookup miss raises ``KeyError`` listing the names that are registered.

    The listing is what turns a typo into a one-line diagnosis.
    """
    registry = StreamRegistry()
    registry.register("toy")(lambda: None)
    with pytest.raises(KeyError, match="no stream implementation"):
        registry.get("typo")


@pytest.mark.parametrize("shape", [(TOTAL, TOTAL), (1, TOTAL, TOTAL),
                                   (1, 1, TOTAL, TOTAL)])
def test_batch_shared_shapes_squeeze_into_structural(shape):
    """Leading size-1 axes are squeezed away to the ``[L, L]`` structural form.

    Legacy code passes the same batch-shared mask as 2-D, 3-D or 4-D, and all
    three mean one thing, so all three must land in one canonical channel.
    """
    lay = layout()
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[..., 0] = True
    plan = plan_from_legacy_mask(mask, lay)
    assert plan.structural.shape == (TOTAL, TOTAL)
    assert plan.key_valid is None          # A legacy mask never fills channel 2
    assert plan.structural[:, 0].all()


def test_per_sample_mask_is_rejected_not_degraded():
    """A genuinely per-sample ``[2, 1, L, L]`` mask is rejected, not accepted.

    INV-1 keeps the structural channel batch free. Accepting it would silently
    disable BlockMask reuse rather than report a problem.
    """
    lay = layout()
    mask = torch.ones((2, 1, TOTAL, TOTAL), dtype=torch.bool)
    with pytest.raises(ValueError, match="INV-1"):
        plan_from_legacy_mask(mask, lay)


def test_float_bias_is_rejected():
    """A float additive bias is rejected; this entry point takes bool masks.

    A float tensor would be truthy almost everywhere, turning a -inf bias into
    an all-visible mask.
    """
    with pytest.raises(ValueError, match="must be bool"):
        plan_from_legacy_mask(torch.zeros((TOTAL, TOTAL)), layout())


def test_shape_must_match_layout_total():
    """The mask side length must equal ``layout.total``, here 6.

    An 8x8 mask over a 6-token layout would otherwise index past the sequence
    inside the attention kernel.
    """
    with pytest.raises(ValueError, match="layout.total=6"):
        plan_from_legacy_mask(torch.ones((8, 8), dtype=torch.bool), layout())
