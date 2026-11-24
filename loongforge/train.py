# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training entry for the Megatron-based core stack (LLM / VLM / Diffusion).

This is NOT a unified entry for the whole framework. Embodied / VLA models run on a
separate torch-native (DDP/FSDP) subsystem with its own entry at
``loongforge/embodied/train.py`` — see ``loongforge/embodied/README.md``.
"""

import logging
import os
import sys

# `loongforge` imports the repository-level `tools.mcore_bridge` package at
# runtime (HF checkpoint loading). Make the entrypoint self-contained even when
# a launcher only puts the package directory on PYTHONPATH.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "tools")):
    # An existing PYTHONPATH entry may otherwise make another repo's ``tools``
    # package win before this checkout is imported.
    while _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

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
