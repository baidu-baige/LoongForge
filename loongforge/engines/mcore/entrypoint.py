# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MCore training entry."""

from loongforge.engines.mcore.parser import parse_train_args
from loongforge.engines.mcore.trainer_builder import build_model_trainer


def main():
    args = parse_train_args()
    build_model_trainer(args).train()
