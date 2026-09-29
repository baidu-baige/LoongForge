# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Megatron Core checkpoint conversion package."""

import os
import sys
from os.path import dirname
SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(SCRIPT_DIR))

from mcore_checkpoint_convert.common.common_config import CommonConfig

from mcore_checkpoint_convert.huggingface.huggingface_checkpoint import HuggingFaceCheckpoint
from mcore_checkpoint_convert.huggingface.huggingface_config import HuggingFaceConfig

from mcore_checkpoint_convert.mcore.mcore_checkpoint import McoreCheckpoint
from mcore_checkpoint_convert.mcore.mcore_config import McoreConfig