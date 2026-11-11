# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training entry that picks the engine for a model and runs its registered recipe."""

import argparse
import sys

from loongforge.models.catalog import EMBODIED_MODEL_NAMES
from loongforge.training.trainer_builder import build_model_trainer


def main():
    """Pick the engine for ``--model-name`` and run the registered recipe."""
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
    model_name = (args.model_name or "").lower().replace("-", "_")
    engine = "torch" if model_name in EMBODIED_MODEL_NAMES else "megatron"

    build_model_trainer(engine).train()


if __name__ == "__main__":
    main()
