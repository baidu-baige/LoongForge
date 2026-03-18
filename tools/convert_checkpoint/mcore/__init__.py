# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0

"""Megatron Core checkpoint conversion package."""

import os
import sys
from os.path import dirname
SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(SCRIPT_DIR))

from common.common_config import CommonConfig

from huggingface.huggingface_checkpoint import HuggingFaceCheckpoint
from huggingface.huggingface_config import HuggingFaceConfig

from mcore.mcore_checkpoint import McoreCheckpoint
from mcore.mcore_config import McoreConfig