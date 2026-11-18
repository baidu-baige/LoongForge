# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""train.trainers.custom — trainer extensions outside the generic supervised/ paradigm.

Each subpackage hosts trainers + runners that are too model-specific to live in
the generic supervised/ paradigm. ``tensor_wise_trainer.py`` holds the
model-agnostic base for the tensor_wise distributed optimizer, which the
model-specific trainers subclass. Mirrors the layout of ``loongforge/train``.
"""
