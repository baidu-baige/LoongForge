# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Motus model-specific trainer (DeepSpeed/torch.compile + VAE side-stream prefetch)."""

from loongforge.embodied.train.trainers.custom.motus.motus_trainer import MotusTrainer

__all__ = ["MotusTrainer"]
