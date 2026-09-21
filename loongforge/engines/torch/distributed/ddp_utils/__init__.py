# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DDP utilities package: gradient communication hook resolution."""

from .ddp_comm_hook import resolve_comm_hook

__all__ = ["resolve_comm_hook"]
