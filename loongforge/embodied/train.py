# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Embodied training entry."""

from loongforge.embodied.train.parser import parse_train_args
from loongforge.embodied.train.config_map import get_model_schema
from loongforge.embodied.train.trainers import build_model_trainer


def main():
    """Parse configs, build the trainer, and start the training loop."""
    training_args, model_cfg, data_cfg = parse_train_args()
    trainer_cls = get_model_schema(training_args.model_name).trainer_cls
    if trainer_cls is None:
        trainer = build_model_trainer(training_args, model_cfg, data_cfg)
    else:
        trainer = trainer_cls(training_args, model_cfg, data_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
