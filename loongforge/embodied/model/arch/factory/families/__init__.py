# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Per-family layer between ``ArchModelBase`` and one model package.

A family fixes the three things a model package does not get to choose: the
attention topology (which engine runs it), the batch path (what
``encode_batch`` turns the trainer's sample into), and the frozen towers the
shell owns. ``factory/<model_name>/`` then adds only its registry key, its
``config_cls`` and its ``transfer`` table.
"""

from .dual_stream_wan import DualStreamWanBase, DualStreamWanConfig

__all__ = ["DualStreamWanBase", "DualStreamWanConfig"]
