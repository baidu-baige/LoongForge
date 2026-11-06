# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-specific training components.

Importing the package registers cosmos-framework's capturable FusedAdam as the
``CapturableFusedAdamW`` optimizer.
"""

from loongforge.embodied.train.trainers.custom.cosmos3.cosmos3_optimizer import CapturableFusedAdam

__all__ = ["CapturableFusedAdam"]
