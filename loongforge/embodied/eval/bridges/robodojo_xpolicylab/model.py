# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""XPolicyLab ``Model`` entry: the LoongForge RoboDojo bridge (runs in the policy-server process)."""

from loongforge.embodied.eval.bridges.robodojo_policy import RoboDojoPolicyModel as Model

__all__ = ["Model"]
