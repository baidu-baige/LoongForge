# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""check performance"""

from tasks.base_task import BaseTask, TaskResut
from tools.color_logger import create_color_logger
import os, time
from copy import deepcopy
import json, re, yaml
import subprocess
import os
import sys
import shutil
import random
from typing import Dict, List, Any

logger = create_color_logger(name=__name__)
import json
import yaml
import glob
from datetime import timedelta

class PerfnessCheckTask(BaseTask):
    def __init__(self,
                 model_description: Dict[str, Any],
                 task_description: Dict[str, Any],
                 model: List[Dict[Any, Any]],
                 model_configer: object,
                 args
                ) -> None:
        super().__init__(model_description,
                 task_description,
                 model,
                 model_configer,
                 args,
                 task_type = "perf",
                )
        self.class_name = self.__class__.__name__
        self.__init_ckpt_file__()


    def __call__(self) -> TaskResut:
        if not self.MODEL_RUNNABLE:
            logger.warn(f"{self.class_name} current model {self.model_name} does not support {self.class_name} task, skipping!!!")
            return TaskResut()

        for index, scenario_data_list in enumerate(self.model["scenarios"]):
            for scenario_name, scenario_data in scenario_data_list.items():
                for training_type_name, training_type_data in scenario_data.items():
                    if scenario_name != "perf":
                        continue
                    if training_type_name not in self.input_cmd_args.training_type:
                        continue
                    model_name = self.model_name
                    logger.info(f"{self.class_name} Model [{model_name}] - [{scenario_name}] Execution Start ...")

                    # Step2:
                    step2_name = "Step2"
                    self.start_loongforge(index, step2_name, scenario_name, training_type_name)
                    step2_scenario_lock_file = os.path.join(self.model["model_lock_file_path"], scenario_name, step2_name, self.master_addr, f"{self.rank_name}_lock.txt")
                    self.wait_async_pod_complete(step2_scenario_lock_file, model_name, f"{scenario_name}_{step2_name}")
                    logger.info(f"{self.class_name} Model [{model_name}] - [{scenario_name}] - [{step2_name}] Completed \n")

                    logger.info(f"{self.class_name} Model [{model_name}] - [{scenario_name}] Execution End \n")

        return TaskResut()
