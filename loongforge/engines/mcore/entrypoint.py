# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MCore training entry."""

from loongforge.engines.mcore import build_model_trainer, parse_train_args


def main():
    args = parse_train_args()
    build_model_trainer(args).train()
