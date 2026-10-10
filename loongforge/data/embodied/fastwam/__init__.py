# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM model-specific data transforms and collator."""

from loongforge.data.embodied.fastwam import fastwam_collator, fastwam_transform

__all__ = ["fastwam_collator", "fastwam_transform"]
