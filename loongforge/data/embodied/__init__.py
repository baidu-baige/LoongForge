# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Embodied (VLA) data.

``registry.py`` maps a model type to its transform builder, collator, sampler,
and dataset strategy. ``dataloader.py`` builds the DataLoader. ``shared/`` holds
everything used by more than one model: ``collator_base.py``,
``sampler_factory.py``, ``formats/`` (generic dataset readers), and
``transforms/`` (per-sample transforms). ``<model>/`` holds one package per
model.
"""
