# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dual-arm morphology retargeting package."""

import os


# MuJoCo reads its rendering backend during import. Keep the legacy selection
# at package initialization so every retarget submodule observes the same value.
os.environ["MUJOCO_GL"] = os.environ.get(
    "EGO2ROBOT_GL", os.environ.get("MUJOCO_GL", "osmesa")
)
