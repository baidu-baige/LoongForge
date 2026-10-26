# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Composition root: resolve the engine, parse its arguments, and run the recipe."""

import argparse
from importlib import import_module
import logging

logging.basicConfig(level=logging.WARNING)


def _parse_args(engine):
    """Parse the engine's training arguments with the current module layout."""
    if engine == "megatron":
        from loongforge.training.megatron_args import parse as parse_megatron

        return parse_megatron()

    from loongforge.training.torch_args import parse as parse_torch

    return parse_torch()


def main():
    """Resolve the engine selected by ``--model-name`` and run its recipe."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-name")
    parser.add_argument("--config-file")
    args, _ = parser.parse_known_args()

    if args.model_name is None:
        if args.config_file is None:
            parser.error("one of --model-name or --config-file is required")
        engine = "megatron"
    else:
        from loongforge.models.catalog import get_model_entry

        try:
            engine = get_model_entry(args.model_name)["engine"]
        except ValueError as exc:
            parser.error(str(exc))

    parsed = _parse_args(engine)
    run = import_module("loongforge.training.registry").resolve(engine, parsed)
    return run()


if __name__ == "__main__":
    main()
