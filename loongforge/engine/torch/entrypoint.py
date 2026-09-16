# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Torch training entry."""

from loongforge.engine.torch.parser import parse_train_args
from loongforge.engine.torch.trainers import build_model_trainer


def main():
    """Parse configs, build the trainer, and start the training loop."""
    training_args, model_cfg, data_cfg = parse_train_args()
    trainer = build_model_trainer(training_args, model_cfg, data_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
