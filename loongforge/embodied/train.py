# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge Embodied training entry."""

import logging

from loongforge.embodied.train.parser import parse_train_args
from loongforge.embodied.train.config_map import get_model_schema
from loongforge.embodied.train.trainers import build_model_trainer

logger = logging.getLogger(__name__)


def main():
    """Parse configs, build the trainer, and start the training loop.

    Trainer selection has a single, explicit precedence:
      1. If the model's schema pins a ``trainer_cls`` (see
         ``config_map.MODEL_SCHEMA``), that class is authoritative and
         ``--trainer-type`` is not consulted.
      2. Otherwise the ``--trainer-type`` registry (``build_model_trainer``)
         resolves the trainer.
    """
    training_args, model_cfg, data_cfg = parse_train_args()
    schema = get_model_schema(training_args.model_name)
    trainer_cls = schema.trainer_cls
    if trainer_cls is None:
        trainer = build_model_trainer(training_args, model_cfg, data_cfg)
    else:
        # The schema pins the trainer; surface (rather than silently drop) a
        # conflicting --trainer-type so the override is never a surprise. The
        # default "FinetuneTrainer" means "unset", so only a deliberate,
        # non-matching value warns.
        if training_args.trainer_type not in ("FinetuneTrainer", trainer_cls.__name__):
            logger.warning(
                "Model '%s' pins trainer %s via its schema; "
                "ignoring --trainer-type=%s.",
                training_args.model_name,
                trainer_cls.__name__,
                training_args.trainer_type,
            )
        trainer = trainer_cls(training_args, model_cfg, data_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
