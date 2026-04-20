# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Megatron Core checkpoint conversion package."""

import os
import sys
from os.path import dirname
SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(SCRIPT_DIR))

from convert_checkpoint.common.common_config import CommonConfig

from convert_checkpoint.huggingface.huggingface_checkpoint import HuggingFaceCheckpoint
from convert_checkpoint.huggingface.huggingface_config import HuggingFaceConfig

from convert_checkpoint.mcore.mcore_checkpoint import McoreCheckpoint
from convert_checkpoint.mcore.mcore_config import McoreConfig