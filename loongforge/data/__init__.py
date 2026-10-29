# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""model dataset"""

IGNORE_INDEX = -100


class TrainingPhase(object):
    """Training phase"""

    PRETRAIN = "pretrain"
    SFT = "sft"
