# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MCore training entry."""

from loongforge.engine.mcore.parser import parse_train_args
from loongforge.engine.mcore.trainer_builder import build_model_trainer


def main():
    """Run the Megatron training engine."""
    args = parse_train_args()
    build_model_trainer(args).train()


if __name__ == "__main__":
    main()
