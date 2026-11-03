# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training entry for Megatron and Torch model families."""

import argparse
import importlib
import logging
import sys

from loongforge.models.catalog import TORCH_MODEL_NAMES


TORCH_TRAINER_TYPES = {
    "FinetuneTrainer": "loongforge.training.embodied.sft.FinetuneTrainer",
    "TensorWiseTrainer": "loongforge.training.embodied.custom.tensor_wise_trainer.TensorWiseTrainer",
    "VLDAFinetuneTrainer": "loongforge.training.embodied.custom.vlda.VLDAFinetuneTrainer",
    "GrootN1d6Trainer": "loongforge.training.embodied.custom.gr00t_n1_6.GrootN1d6Trainer",
    "LingBotFinetuneTrainer": "loongforge.training.embodied.custom.lingbot_va.LingBotFinetuneTrainer",
    "GrootN1d7Trainer": "loongforge.training.embodied.custom.gr00t_n1_7.GrootN1d7Trainer",
    "LingbotVlaV2Trainer": "loongforge.training.embodied.custom.lingbot_vla_v2.LingbotVlaV2Trainer",
}


def resolve_torch_trainer(trainer_type: str):
    """Load the Torch recipe selected by ``--trainer-type``."""
    if not trainer_type:
        raise ValueError(
            "--trainer-type is required. "
            f"Available: {list(TORCH_TRAINER_TYPES.keys())}"
        )

    entry = TORCH_TRAINER_TYPES.get(trainer_type)
    if entry is None:
        raise ValueError(
            f"Unknown --trainer-type '{trainer_type}'. "
            f"Available: {list(TORCH_TRAINER_TYPES.keys())}"
        )

    module_path, class_name = entry.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _run_megatron():
    """Parse the Megatron arguments and run the recipe for their family and phase."""
    logging.basicConfig(level=logging.WARNING)
    from loongforge.engines.megatron.parser import parse_train_args
    from loongforge.training import MODEL_FAMILY_TRAINER_FACTORY, _load_builtin_trainers

    _load_builtin_trainers()
    args = parse_train_args()
    model_family = args.model_family

    if model_family not in MODEL_FAMILY_TRAINER_FACTORY:
        raise ValueError(f"Not found trainer for family: {model_family}")
    if args.training_phase not in MODEL_FAMILY_TRAINER_FACTORY[model_family]:
        raise ValueError(
            f"Loongforge not support {args.training_phase} phase for {args.model_name} (family: {model_family})"
        )

    trainer = MODEL_FAMILY_TRAINER_FACTORY[model_family][args.training_phase](args)
    trainer.train()


def _run_torch():
    """Parse the Torch arguments and run the trainer selected by ``--trainer-type``."""
    from loongforge.training.torch_args import parse_train_args

    training_args, model_cfg, data_cfg = parse_train_args()
    trainer_type = resolve_torch_trainer(training_args.trainer_type)
    trainer = trainer_type(training_args, model_cfg, data_cfg)
    trainer.train()


def main():
    """Pick the engine for ``--model-name`` and run the selected recipe."""
    argv = sys.argv[1:]
    for index, arg in enumerate(argv):
        if arg in ("-h", "--help", "--"):
            argv = argv[:index]
            break

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    # Let the selected engine report missing values.
    parser.add_argument("--model-name", nargs="?")
    args, _ = parser.parse_known_args(argv)
    if args.model_name is None:
        # Only Torch accepts model-name abbreviations.
        parser.allow_abbrev = True
        args, _ = parser.parse_known_args(argv)

    # Keep validation and help in the selected engine's original parser.
    if args.model_name and args.model_name.lower().replace("-", "_") in TORCH_MODEL_NAMES:
        _run_torch()
    else:
        _run_megatron()


if __name__ == "__main__":
    main()
