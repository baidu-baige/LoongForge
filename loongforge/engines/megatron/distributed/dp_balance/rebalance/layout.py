# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Batch layout registry for DP load balancing.

``dp_balance`` owns the sample reordering algorithm; a model family owns the
batch layout that algorithm operates on.  A layout supplies two callbacks:

* ``reorder``: redistribute one packed micro-batch across DP ranks.
* ``cumulative_lengths``: cumulative sequence offsets used by the warm-up
  statistics.

The default layout for ``VisionLanguageModelFamilies`` lives in
``balance.py``.  Families with a different packed format register their own
layout before training starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from loongforge.engines.megatron.distributed.dp_balance.rebalance.balance import RebalanceContext


@dataclass(frozen=True)
class BatchLayout:
    reorder: Callable[[dict, RebalanceContext], dict]
    cumulative_lengths: Callable[[Dict[str, torch.Tensor]], torch.Tensor]
    caller: str


_BATCH_LAYOUTS: Dict[str, BatchLayout] = {}


def set_batch_layout(model_family: str, layout: BatchLayout) -> None:
    """Register the batch layout used by *model_family*."""
    _BATCH_LAYOUTS[model_family] = layout


def get_batch_layout(model_family: str) -> BatchLayout:
    """Return the registered layout, or the VLM default for VLM families."""
    layout = _BATCH_LAYOUTS.get(model_family)
    if layout is not None:
        return layout

    from loongforge import constants

    if model_family in constants.VisionLanguageModelFamilies.names():
        from loongforge.engines.megatron.distributed.dp_balance.rebalance.balance import (
            VLM_BATCH_LAYOUT,
        )

        return VLM_BATCH_LAYOUT

    supported = sorted(
        [*_BATCH_LAYOUTS, *constants.VisionLanguageModelFamilies.names()]
    )
    raise ValueError(
        f"Unsupported model_family '{model_family}'. Expected one of: {supported}"
    )
