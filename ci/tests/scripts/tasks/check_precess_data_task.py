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

class PrecessDataCheckTask(BaseTask):
    """PrecessDataCheckTask"""
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
                 task_type = "preprocess_data",
                )
        self.class_name = self.__class__.__name__

    
    def deal_output(self, model_config, step_stage):
        """deal_output"""
        output_prefix = model_config["output_prefix"]

        if step_stage == "pretrain":
            if os.path.isdir(os.path.dirname(output_prefix)):
                shutil.rmtree(os.path.dirname(output_prefix))
            os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        elif step_stage == "sft":
            if os.path.isdir(output_prefix):
                shutil.rmtree(output_prefix)
            os.makedirs(output_prefix, exist_ok=True)
        else:
            # 检查预处理数据是否存在
            logger.error(f"deal_train_json 不支持其他 {step_stage} 模式 !!!")
            sys.exit(1)

    def assert_preprocess_data(self, model_config, step_stage):
        """assert_preprocess_data"""
        output_prefix = model_config["output_prefix"]
        output_dir = output_prefix

        if step_stage == "pretrain":
            output_dir = os.path.dirname(output_prefix)

            # 搜索路径下的.bin和.idx文件
            bin_files = glob.glob(os.path.join(output_dir, '*.bin'))
            idx_files = glob.glob(os.path.join(output_dir, '*.idx'))

            # 使用断言来确保找到了.bin和.idx文件
            assert len(bin_files) > 0, "No .bin files found in " + output_dir
            assert len(idx_files) > 0, "No .idx files found in " + output_dir

        elif step_stage == "sft":
            # 检查预处理数据是否存在
            dataset_info_files = glob.glob(os.path.join(output_dir, 'dataset_info.json'))
            state_files = glob.glob(os.path.join(output_dir, 'state.json'))

            # assert len(dataset_info_files) > 0, "No dataset_info files found in " + output_dir
            # assert len(state_files) > 0, "No state files found in " + output_dir
        else:
            # 检查预处理数据是否存在
            logger.error(f"assert_preprocess_data 不支持其他 {step_stage} 模式校验 !!!")
            sys.exit(1)


    def start_aiak_preprocess_data(self, index, step_stage, scenario_name):
        """start_aiak_preprocess_data"""
        step_name = "aiak_preprocess_data"
        logger.info(f"{step_stage} {step_name} Start Running ...")

        model_config = self.__init_model_scenarios_data__(index, scenario_name, step_stage)

        # ckpt 权重转化
        model_name = self.model_name
        node_nums = self.input_cmd_args.node_nums
        timeout = self.input_cmd_args.timeout
        scripts_root_path = model_config["scripts_root_path"]
        model_lock_file_path = model_config["model_lock_file_path"]
        training_log_path = model_config["training_log_path"]

        # 将配置文件转成env 环境变量传递给运行脚本
        env_vars_str = self.__convert_model_config_to_env__(model_config)
        self.deal_output(model_config, step_stage)

        step_stage_path = f'{model_lock_file_path}/{step_stage}/{self.master_addr}'
        model_lock_file = f'{step_stage_path}/{self.rank_name}_lock.txt'

        script_path = f"{scripts_root_path}/executor/{step_name}/run.sh"
        new_script_path = f"{training_log_path}/precess_data_{model_name}_{self.rank_name}_run.sh"
        start_command = f"{env_vars_str} bash {script_path}"
        self.create_shell_file(model_config, script_path, new_script_path)

        # 打开一个新的文件用来写入脚本的输出
        training_log_file = f"{training_log_path}/precess_data#{model_name}#nodes_{self.input_cmd_args.node_nums}#{self.rank_name}#run.log"

        start_command = f"{env_vars_str} bash -c \"set -o pipefail; bash {scripts_root_path}/executor/{step_name}/run.sh |tee {training_log_file}\""
        logger.info(f"{step_stage} {step_name} Start: {start_command} .")
        if os.system(start_command) != 0:
           raise RuntimeError(f"Start {step_stage} {step_name} error, cmd is {start_command}")
        
        # 等待所有pod 完成
        self.wait_async_pod_complete(
            model_lock_file,
            model_name,
            f"{scenario_name}_{step_name}",
            is_function=True,
            function=self.assert_preprocess_data,
            raise_on_error=True,
            model_config=model_config,
            step_stage=step_stage,
        )

        logger.info(f"{step_stage} End {step_name}")

    def __call__(self) -> TaskResut:
        if not self.MODEL_RUNNABLE:
            logger.warn(f"{self.class_name} 当前模型 {self.model_name} 不支持 {self.class_name} 任务，跳过！！！")
            return TaskResut()

        for index, scenario in enumerate(self.model["scenarios"]):
            for scenario_name, scenario_data in scenario.items():
                if scenario_name != "preprocess_data":
                    continue
                model_name = self.model_name
                logger.info(f"{self.class_name} 模型【{model_name}】 - 【{scenario_name}】执行开始 ...")
                
                for key, value in self.model["scenarios"][index][scenario_name].items():

                    # pretrain、sft:
                    step_name = key
                    logger.info(f"{self.class_name} 模型【{model_name}】 - 【{scenario_name}】 - 【{step_name}】 执行开始 ...")

                    self.start_aiak_preprocess_data(index, step_name, scenario_name)
                    step_scenario_lock_file = os.path.join(self.model["model_lock_file_path"], scenario_name, step_name, self.master_addr, f"{self.rank_name}_lock.txt")
                    self.wait_async_pod_complete(step_scenario_lock_file, model_name, f"{scenario_name}_{step_name}")

                    logger.info(f"{self.class_name} 模型【{model_name}】 - 【{scenario_name}】 - 【{step_name}】完成 \n")

                logger.info(f"{self.class_name} 模型【{model_name}】 - 【{scenario_name}】执行结束 \n")
    
        return TaskResut()