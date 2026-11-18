# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The five-layer contract of the multi-stream world model.

Layers and their owning lane (``docs/design/mot_layer_interfaces.zh-CN.md`` §2):

==============  =========================  ======
Layer           Module                     Lane
==============  =========================  ======
L0 geometry     ``geometry`` / ``layout``  A
L1 visibility   ``visibility/``            A
L2 streams      ``streams/``               B
L3 supervision  ``diffusion/``             C
L4 heads        ``heads/``                 C
config layer    ``spec/``                  -
assembly        ``builder``                -
==============  =========================  ======

The four cross-lane interfaces (I1-I4) are frozen at M0; changing one goes
through the master plan §9 amendment process:
  - **I1** A to B: ``AttentionPlan`` plus the four ``CompiledMask`` methods;
  - **I2** A to C: ``TokenLayout.labels_of``;
  - **I3** B to C: ``EngineOutput.captured``;
  - **I4** C to trainer: ``forward(batch) -> (loss, metrics)``.

Three hard invariants:
  - **INV-1** ``AttentionPlan.structural`` is exactly 2-D, with no batch
    dimension, so the structural channel stays batch-shared and compilable;
  - **INV-2** shapes are static, which is what CUDA graph capture requires;
  - **INV-3** no new state_dict prefixes, so existing checkpoints keep loading.
"""

from __future__ import annotations

from .geometry import GeometrySpec
from .layout import LabelView, SegmentRange, TokenLayout

__all__ = ["GeometrySpec", "TokenLayout", "SegmentRange", "LabelView"]
