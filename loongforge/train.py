# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch training to the runner of the engine that owns the selected model."""

import argparse
from importlib import import_module
import logging

logging.basicConfig(level=logging.WARNING)


def main():
    """Run ``loongforge.training.<engine>.runner.main`` for the engine selected by ``--model-name``."""
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

    return import_module(f"loongforge.training.{engine}.runner").main()


if __name__ == "__main__":
    main()
