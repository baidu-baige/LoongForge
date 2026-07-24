# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Trainer construction via --trainer-type argument."""

import logging

from loongforge.embodied.train.trainers.custom.groot_n1_6 import GrootN1d6Trainer
from loongforge.embodied.train.trainers.custom.lingbot_va import LingBotFinetuneTrainer
from loongforge.embodied.train.trainers.custom.groot_n1_7 import GrootN1d7Trainer
from loongforge.embodied.train.trainers.supervised.finetune_trainer import FinetuneTrainer

logger = logging.getLogger(__name__)

_TRAINER_CLASSES = {
    "FinetuneTrainer": FinetuneTrainer,
    "GrootN1d6Trainer": GrootN1d6Trainer,
    "LingBotFinetuneTrainer": LingBotFinetuneTrainer,
    "GrootN1d7Trainer": GrootN1d7Trainer,
}


def build_model_trainer(training_args, model_cfg, data_cfg):
    """Build Trainer from ``training_args.trainer_type``.

    Resolves the trainer class from training_args.trainer_type (e.g.
    "FinetuneTrainer") via _TRAINER_CLASSES, instantiates with the three configs.
    """
    trainer_type = training_args.trainer_type

    if not trainer_type:
        raise ValueError(
            "--trainer-type is required. "
            f"Available: {list(_TRAINER_CLASSES.keys())}"
        )

    trainer_cls = _TRAINER_CLASSES.get(trainer_type)
    if trainer_cls is None:
        raise ValueError(
            f"Unknown --trainer-type '{trainer_type}'. "
            f"Available: {list(_TRAINER_CLASSES.keys())}"
        )

    return trainer_cls(training_args, model_cfg, data_cfg)
