# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training entry that picks the engine for a model and runs its registered recipe."""

import argparse
import sys

from loongforge.models.catalog import get_model_spec
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

    # Let the selected engine keep ownership of full argument validation and help.
    if args.model_name is None:
        engine = "megatron"
    else:
        engine = get_model_spec(args.model_name).engine

    build_model_trainer(engine).train()


if __name__ == "__main__":
    main()
