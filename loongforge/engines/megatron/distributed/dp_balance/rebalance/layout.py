# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Batch layout registry for DP load balancing.

``dp_balance`` owns the sample reordering algorithm. Each packed batch format
defines its layout in ``loongforge/data`` (for example
``data/vlm/dp_balance_layout.py``), and the training method installs the layout
for its model family before the loop starts. A layout supplies two callbacks:

* ``reorder``: redistribute one packed micro-batch across DP ranks.
* ``cumulative_lengths``: cumulative sequence offsets used by the warm-up
  statistics.

The engine holds no default: a missing registration is an error, not a silent
VLM assumption.
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


def install_batch_layout(model_family: str, layout: BatchLayout) -> None:
    """Register the batch layout used by *model_family*.

    Called by the training method. Registering a second, different layout for the same
    family is a programming error.
    """
    existing = _BATCH_LAYOUTS.get(model_family)
    if existing is not None and existing is not layout:
        raise ValueError(
            f"Batch layout already installed for model_family '{model_family}' "
            f"(caller '{existing.caller}', new caller '{layout.caller}')"
        )
    _BATCH_LAYOUTS[model_family] = layout


def get_batch_layout(model_family: str) -> BatchLayout:
    """Return the layout registered for *model_family*."""
    layout = _BATCH_LAYOUTS.get(model_family)
    if layout is None:
        raise ValueError(
            f"No batch layout registered for model_family '{model_family}'. "
            f"Registered: {sorted(_BATCH_LAYOUTS)}. The training method must call "
            f"install_batch_layout before DP load balancing runs."
        )
    return layout
