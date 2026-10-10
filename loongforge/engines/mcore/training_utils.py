# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

import os
import logging
import sys

from megatron.training.log_handler import CustomHandler

# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)


# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add tools directory to Python path
tools_path = os.path.join(project_root, "tools")
if tools_path not in sys.path:
    sys.path.insert(0, tools_path)


from loongforge.training.mcore_runner import pretrain
from loongforge.engines.mcore.train_step import dump_model_input_example_once

__all__ = ["dump_model_input_example_once", "pretrain"]
