# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training entry for Megatron and Torch model families."""

import argparse
import logging
import sys

from loongforge.models.catalog import TORCH_MODEL_NAMES


def _run_megatron():
    """Parse the Megatron arguments and run the recipe for their family and phase."""
    logging.basicConfig(level=logging.WARNING)
    from loongforge.engines.megatron.parser import parse_train_args
    from loongforge.training.trainer_builder import build_model_trainer

    args = parse_train_args()
    trainer = build_model_trainer(args)
    trainer.train()


def _run_torch():
    """Parse the Torch arguments and run the trainer selected by ``--trainer-type``."""
    from loongforge.training.torch_args import parse_train_args
    from loongforge.training.embodied.trainer_builder import build_embodied_trainer

    training_args, model_cfg, data_cfg = parse_train_args()
    trainer = build_embodied_trainer(training_args, model_cfg, data_cfg)
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
