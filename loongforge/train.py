# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training entry for the Megatron-based core stack (LLM / VLM / Diffusion).

This is NOT a unified entry for the whole framework. Embodied / VLA models run on a
separate torch-native (DDP/FSDP) subsystem with its own entry at
``loongforge/embodied/train.py`` — see ``loongforge/embodied/README.md``.
"""

import logging
logging.basicConfig(level=logging.WARNING)
from loongforge.train import parse_train_args
from loongforge.train import build_model_trainer


def main():
    """train cmd"""

    # parse args and config
    args = parse_train_args()

    # get model trainer
    trainer = build_model_trainer(args)

    # start training
    trainer.train()


if __name__ == "__main__":
    main()
