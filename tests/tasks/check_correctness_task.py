# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""check correctness"""

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

class CorrectnessCheckTask(BaseTask):
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
                 args
                )
        # self.__init_ckpt_file__()


    def __call__(self) -> TaskResut:
        if not self.MODEL_RUNNABLE:
            logger.warn(f"CorrectnessCheckTask current model {self.model_name} does not support CorrectnessCheckTask, skipping!!!")
            return TaskResut()

        for index, scenario_data_list in enumerate(self.model["scenarios"]):
            for scenario_name, scenario_data in scenario_data_list.items():
                for training_type_name, training_type_data in scenario_data.items():
                    if scenario_name != "function":
                        continue
                    if training_type_name not in self.input_cmd_args.training_type:
                        continue
                    model_name = self.model_name
                    logger.info(f"CorrectnessCheckTask Model [{model_name}] - [{scenario_name}] - [{training_type_name}] Execution Start ...")
                    
                    # Step1:
                    step1_name = "Step1"
                    if step1_name in self.model["scenarios"][index][scenario_name][training_type_name]:
                        if "RUNNABLE_FLAG" not in self.model["scenarios"][index][scenario_name][training_type_name][step1_name] or \
                                ("RUNNABLE_FLAG" in self.model["scenarios"][index][scenario_name][training_type_name][step1_name] and \
                                    self.model["scenarios"][index][scenario_name][training_type_name][step1_name]["RUNNABLE_FLAG"] == "true"):
                            self.start_loongforge_convert_ckpt(index, step1_name, scenario_name, training_type_name)
                            step1_scenario_lock_file = os.path.join(self.model["model_lock_file_path"], scenario_name, step1_name, self.master_addr, f"{self.rank_name}_lock.txt")
                            self.wait_async_pod_complete(step1_scenario_lock_file, model_name, f"{scenario_name}_{step1_name}")
                            logger.info(f"CorrectnessCheckTask Model [{model_name}] - [{scenario_name}] - [{training_type_name}] - [{step1_name}] Completed \n")
                    
                    # Step1.5: mcore to hf reverse convert and check
                    step1_5_name = "Step1.5"
                    if step1_5_name in self.model["scenarios"][index][scenario_name][training_type_name]:
                        if "RUNNABLE_FLAG" not in self.model["scenarios"][index][scenario_name][training_type_name][step1_5_name] or \
                                ("RUNNABLE_FLAG" in self.model["scenarios"][index][scenario_name][training_type_name][step1_5_name] and \
                                    self.model["scenarios"][index][scenario_name][training_type_name][step1_5_name]["RUNNABLE_FLAG"] == "true"):
                            self.start_loongforge_reverse_convert_ckpt(index, step1_5_name, scenario_name, training_type_name)
                            step1_5_scenario_lock_file = os.path.join(self.model["model_lock_file_path"], scenario_name, step1_5_name, self.master_addr, f"{self.rank_name}_lock.txt")
                            self.wait_async_pod_complete(step1_5_scenario_lock_file, model_name, f"{scenario_name}_{step1_5_name}")
                            logger.info(f"CorrectnessCheckTask Model [{model_name}] - [{scenario_name}] - [{training_type_name}] - [{step1_5_name}] Completed \n")
                    
                    # Step2:
                    step2_name = "Step2"
                    if step2_name in self.model["scenarios"][index][scenario_name][training_type_name]:
                        if "RUNNABLE_FLAG" not in self.model["scenarios"][index][scenario_name][training_type_name][step2_name] or \
                                ("RUNNABLE_FLAG" in self.model["scenarios"][index][scenario_name][training_type_name][step2_name] and \
                                    self.model["scenarios"][index][scenario_name][training_type_name][step2_name]["RUNNABLE_FLAG"] == "true"):
                            self.start_loongforge(index, step2_name, scenario_name, training_type_name)
                            step2_scenario_lock_file = os.path.join(self.model["model_lock_file_path"], scenario_name, step2_name, self.master_addr, f"{self.rank_name}_lock.txt")
                            self.wait_async_pod_complete(step2_scenario_lock_file, model_name, f"{scenario_name}_{step2_name}")
                            logger.info(f"CorrectnessCheckTask Model [{model_name}] - [{scenario_name}] - [{training_type_name}] - [{step2_name}] Completed \n")

                    # Step3:
                    step3_name = "Step3"
                    if step3_name in self.model["scenarios"][index][scenario_name][training_type_name]:
                        if "RUNNABLE_FLAG" not in self.model["scenarios"][index][scenario_name][training_type_name][step3_name] or \
                                ("RUNNABLE_FLAG" in self.model["scenarios"][index][scenario_name][training_type_name][step3_name] and \
                                    self.model["scenarios"][index][scenario_name][training_type_name][step3_name]["RUNNABLE_FLAG"] == "true"):
                            self.start_loongforge(index, step3_name, scenario_name, training_type_name)
                            step3_scenario_lock_file = os.path.join(self.model["model_lock_file_path"], scenario_name, step3_name, self.master_addr, f"{self.rank_name}_lock.txt")
                            self.wait_async_pod_complete(step3_scenario_lock_file, model_name, f"{scenario_name}_{step3_name}")
                            logger.info(f"CorrectnessCheckTask Model [{model_name}] - [{scenario_name}] - [{step3_name}] Completed \n")

                    logger.info(f"CorrectnessCheckTask Model [{model_name}] - [{scenario_name}] - [{training_type_name}] Execution End \n")
        
        # Clean up ckpt files etc.
        # self.__clean_up__()

        return TaskResut()
