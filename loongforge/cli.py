# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Console entry point for the Megatron-based training stack."""

import argparse
import logging
import sys
from importlib.metadata import version


def run_training():
    """Run Megatron training with the process arguments."""
    logging.basicConfig(level=logging.WARNING)
    from loongforge.train import build_model_trainer, parse_train_args

    args = parse_train_args()
    trainer = build_model_trainer(args)
    trainer.train()


def main():
    """Handle standalone CLI options before loading the training stack."""
    args = sys.argv[1:]
    if args == ["--version"]:
        print(f"LoongForge {version('LoongForge')}")
        return
    if args in (["-h"], ["--help"]):
        parser = argparse.ArgumentParser(
            prog="LoongForge",
            description="Launch Megatron-based LLM, VLM, and diffusion training.",
            epilog="Training requires a configured GPU or XPU environment. "
            "For all training options, run python loongforge/train.py --help in that environment.",
        )
        parser.add_argument("--version", action="store_true", help="show the installed version and exit")
        parser.add_argument(
            "training_args", nargs=argparse.REMAINDER, metavar="TRAINING_ARGS", help="arguments passed to training"
        )
        parser.print_help()
        return
    run_training()
