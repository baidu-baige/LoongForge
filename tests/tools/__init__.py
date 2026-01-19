#!/usr/bin/env python3
"""
__init__.py
"""
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################

from tasks.base_task import BaseTask
from tasks.check_correctness_task import CorrectnessCheckTask
from tasks.check_perfness_task import PerfnessCheckTask
from tasks.check_precess_data_task import PrecessDataCheckTask


SUPPORTED_TASKS = {
    "check_correctness_task":CorrectnessCheckTask,
    "check_perfness_task": PerfnessCheckTask,
    "check_precess_data_task":PrecessDataCheckTask,
}