# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training entry for the Megatron-based core stack (LLM / VLM / Diffusion).

This is NOT a unified entry for the whole framework. Embodied / VLA models run on a
separate torch-native (DDP/FSDP) subsystem with its own entry at
``loongforge/embodied/train.py`` — see ``loongforge/embodied/README.md``.
"""

from loongforge.cli import run_training


if __name__ == "__main__":
    run_training()
