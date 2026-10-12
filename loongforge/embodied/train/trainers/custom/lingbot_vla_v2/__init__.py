# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Custom trainers for LingBot VLA v2."""

from .lingbot_vla_v2_replicated_sharded_trainer import LingbotVlaV2ReplicatedShardedTrainer

__all__ = ["LingbotVlaV2ReplicatedShardedTrainer"]
