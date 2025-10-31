#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import os, json
import yaml, re
from string import Template
from copy import deepcopy
from typing import Dict, List, Any
from tasks import BaseTask, SUPPORTED_TASKS
from tools.color_logger import create_color_logger

logger = create_color_logger(name=__name__)

class ConfigManager(object):
    def __init__(self, args) -> None:
        self.args = args
        self.all_model_configs : List[Dict[Any, Any]] \
            = self.get_all_model_configs(self.args.configs_dir)


    def get_model_description(self, model_config) -> Dict[Any, Any]:
        return model_config["description"]


    @staticmethod
    def get_model_name(model_config) -> str:
        return model_config["model_name"]


    @staticmethod
    def get_tasks(model_config) -> List[str]:
        task_list = []
        for key, value in model_config["tasks"].items():
            if value:
                task_list.append(key)
        return task_list


    @staticmethod
    def get_task_runner(task_name) -> BaseTask:
        return SUPPORTED_TASKS[task_name]
    
    def get_all_model_configs(self, configs_dir: str) -> List[Dict[Any, Any]]:
        all_model_configs: List[Dict[Any, Any]] = []
        
        for model_name in self.args.models:
            try:
                file = f"{model_name}.yaml"
                config = self.load_config(configs_dir, file)

                if ConfigManager.get_model_name(config) in self.args.models:
                    tasks = ConfigManager.get_tasks(config)
                    for task in tasks:
                        if tasks not in self.args.tasks:
                            tasks.remove(task)
                        
                    all_model_configs.append(config)
            except Exception as e:
                logger.warning(f"Try to load {file}, but failed, {e}.")
                raise e


        return all_model_configs

    def load_config(self, configs_dir: str, model_file: str) -> Dict[Any, Any]:
        common_config_file = os.path.join(configs_dir, "common.yaml")
        model_config_file = os.path.join(configs_dir, model_file)

        def recursive_replace(match):
            key = match.group(1)
            return str(config.get(key, match.group(0)))

        # 加载 common.yaml
        with open(common_config_file, 'r') as f:
            common_config_str = f.read()

        # 加载 chatglm-6b.yaml
        with open(model_config_file, 'r') as f:
            model_config_str = f.read()

        # 将两个配置文件的内容解析为字典
        common_config = yaml.safe_load(common_config_str)
        model_config = yaml.safe_load(model_config_str)

        # 合并 common_config 和 model_config
        config = {**common_config, **model_config}

        # 递归替换所有占位符
        for _ in range(10):  # 设置最大迭代次数为10
            new_common_config_str = re.sub(r'\$([a-z_][a-z0-9_]*)', recursive_replace, common_config_str)
            new_model_config_str = re.sub(r'\$([a-z_][a-z0-9_]*)', recursive_replace, model_config_str)

            if new_common_config_str == common_config_str and new_model_config_str == model_config_str:
                # 如果两个字符串都没有发生变化，那么可以提前结束循环
                break

            common_config_str = new_common_config_str
            model_config_str = new_model_config_str

            # 更新 config
            common_config = yaml.safe_load(common_config_str)
            model_config = yaml.safe_load(model_config_str)
            config = {**common_config, **model_config}

        return config


    @staticmethod
    def get_all_model_tasks(tasks_dir: str) -> List[Dict[Any, Any]]:
        all_model_tasks: List[Dict[Any, Any]] = []
        all_tasks_file = os.listdir(tasks_dir)
        all_tasks = [os.path.splitext(file)[0] for file in all_tasks_file]
        for task in all_tasks:
            if task.startswith("check_"):
                all_model_tasks.append(task)
        return all_model_tasks

    @staticmethod
    def get_all_models(configs_dir: str) -> List[Dict[Any, Any]]:
        all_model: List[Dict[Any, Any]] = []
        all_model_file = os.listdir(configs_dir)
        all_tasks_name = [os.path.splitext(file)[0] for file in all_model_file]
        for task in all_tasks_name:
            if not task.startswith("common"):
                all_model.append(task)
        return all_model

    def get_scenarios_num(self) -> int:
        # 模型数量 + 场景数量。暂先不考虑任务数量
        # tasks = ConfigManager.get_tasks(config)
        num = 0
        for models in self.all_model_configs:
            # num += len(models["scenarios"].keys())
            num += 1
        return num