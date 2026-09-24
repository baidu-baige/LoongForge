# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Muon for the replicated-sharded optimizer.

``DistributedMuon`` updates the fp32 master parameters the shard owns and
reports each finished update through ``param_update_callback`` so the shard can
publish it back to the replicas. ``build_muon_optimizer`` splits parameters
between that Muon and an AdamW fallback.
"""

from .muon import DistributedMuon, split_muon_adamw_params
from .optimizer import build_muon_optimizer

__all__ = [
    "DistributedMuon",
    "build_muon_optimizer",
    "split_muon_adamw_params",
]
