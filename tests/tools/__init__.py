# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""""Supported tasks"""

from tasks.base_task import BaseTask
from tasks.check_correctness_task import CorrectnessCheckTask
from tasks.check_perfness_task import PerfnessCheckTask
from tasks.check_precess_data_task import PrecessDataCheckTask


SUPPORTED_TASKS = {
    "check_correctness_task":CorrectnessCheckTask,
    "check_perfness_task": PerfnessCheckTask,
    "check_precess_data_task":PrecessDataCheckTask,
}