#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################

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
        self.__init_ckpt_file__()


    def __call__(self) -> TaskResut:
        if not self.MODEL_RUNNABLE:
            logger.warn(f"CorrectnessCheckTask 当前模型 {self.model_name} 不支持 CorrectnessCheckTask 任务，跳过！！！")
            return TaskResut()

        for index, scenario_data_list in enumerate(self.model["scenarios"]):
            for scenario_name, scenario_data in scenario_data_list.items():
                for training_type_name, training_type_data in scenario_data.items():
                    if scenario_name != "function":
                        continue
                    if training_type_name not in self.input_cmd_args.training_type:
                        continue
                    model_name = self.model_name
                    logger.info(f"CorrectnessCheckTask 模型【{model_name}】 - 【{scenario_name}】 - 【{training_type_name}】 执行开始 ...")
                    
                    # Step1:
                    step1_name = "Step1"
                    if "RUNNABLE_FLAG" not in self.model["scenarios"][index][scenario_name][training_type_name][step1_name] or \
                            ("RUNNABLE_FLAG" in self.model["scenarios"][index][scenario_name][training_type_name][step1_name] and \
                                self.model["scenarios"][index][scenario_name][training_type_name][step1_name]["RUNNABLE_FLAG"] == "true"):
                        self.start_aiak_convert_ckpt(index, step1_name, scenario_name, training_type_name)
                        step1_scenario_lock_file = os.path.join(self.model["model_lock_file_path"], scenario_name, step1_name, self.master_addr, f"{self.rank_name}_lock.txt")
                        self.wait_async_pod_complete(step1_scenario_lock_file, model_name, f"{scenario_name}_{step1_name}")
                        logger.info(f"CorrectnessCheckTask 模型【{model_name}】 - 【{scenario_name}】 - 【{training_type_name}】 - 【{step1_name}】完成 \n")
                    
                    # Step2:
                    step2_name = "Step2"
                    self.start_aiak_training_omni(index, step2_name, scenario_name, training_type_name)
                    step2_scenario_lock_file = os.path.join(self.model["model_lock_file_path"], scenario_name, step2_name, self.master_addr, f"{self.rank_name}_lock.txt")
                    self.wait_async_pod_complete(step2_scenario_lock_file, model_name, f"{scenario_name}_{step2_name}")
                    logger.info(f"CorrectnessCheckTask 模型【{model_name}】 - 【{scenario_name}】  - 【{training_type_name}】 - 【{step2_name}】完成 \n")

                    # Step3:
                    step3_name = "Step3"
                    if step3_name in self.model["scenarios"][index][scenario_name]:
                        self.start_aiak_training_omni(index, step3_name, scenario_name, training_type_name)
                        step3_scenario_lock_file = os.path.join(self.model["model_lock_file_path"], scenario_name, step3_name, self.master_addr, f"{self.rank_name}_lock.txt")
                        self.wait_async_pod_complete(step3_scenario_lock_file, model_name, f"{scenario_name}_{step3_name}")
                        logger.info(f"CorrectnessCheckTask 模型【{model_name}】 - 【{scenario_name}】 - 【{step3_name}】完成 \n")

                    logger.info(f"CorrectnessCheckTask 模型【{model_name}】 - 【{scenario_name}】 - 【{training_type_name}】执行结束 \n")
        
        # 清理 ckpt 等文件
        self.__clean_up__()

        return TaskResut()
