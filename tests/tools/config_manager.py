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
from typing import Dict, List, Any, TYPE_CHECKING, Optional, Tuple
from tools.color_logger import create_color_logger

if TYPE_CHECKING:
    from tasks import BaseTask

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
    def get_task_runner(task_name):
        """获取任务运行器（延迟导入以避免循环依赖）"""
        from tasks import SUPPORTED_TASKS
        return SUPPORTED_TASKS[task_name]
    
    def get_all_model_configs(self, configs_dir: str) -> List[Dict[Any, Any]]:
        all_model_configs: List[Dict[Any, Any]] = []
        
        for model_name in self.args.models:
            try:
                # 查找配置文件所在目录和文件名
                actual_configs_dir, yaml_filename = self.find_config_file(model_name)
                if actual_configs_dir is None:
                    logger.warning(f"Config file for model '{model_name}' not found in any config directory.")
                    continue
                
                config = self.load_config(actual_configs_dir, yaml_filename)

                if ConfigManager.get_model_name(config) in [m.split("/")[-1] if "/" in m else m for m in self.args.models]:
                    tasks = ConfigManager.get_tasks(config)
                    for task in tasks:
                        if tasks not in self.args.tasks:
                            tasks.remove(task)
                    
                    # 添加配置来源信息，便于调试和区分重名模型
                    config["_config_source"] = {
                        "dir": actual_configs_dir,
                        "file": yaml_filename,
                        "model_identifier": model_name  # 原始的模型标识符（可能带路径前缀）
                    }
                    all_model_configs.append(config)
            except Exception as e:
                logger.warning(f"Try to load config for '{model_name}', but failed: {e}")
                raise e

        return all_model_configs

    def load_baseline_data_from_json(self, json_path, training_type=None):
        """从JSON文件加载基准数据
        
        Args:
            json_path: JSON文件路径
            training_type: 训练类型（如 'pretrain', 'sft'），用于从 JSON 中选择对应数据
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"基准数据文件不存在: {json_path}")
        
        with open(json_path, 'r') as f:
            baseline_data = json.load(f)
        
        # 根据 training_type 获取对应的数据
        if isinstance(baseline_data, dict):
            if training_type and training_type in baseline_data:
                baseline_data = baseline_data[training_type]
            else:
                available_keys = list(baseline_data.keys())
                raise ValueError(f"training_type '{training_type}' 未指定或在 {json_path} 中不存在，可用的 training_type: {available_keys}")
        
        # 按 iteration 顺序提取所需数据
        lm_loss_list = [item["lm_loss"] for item in baseline_data]
        grad_norm_list = [item["grad_norm"] for item in baseline_data]
        elapsed_time_list = [item["elapsed_time_ms"] for item in baseline_data]
        throughput_list = [item["throughput"] for item in baseline_data]

        # 新增显存指标（可选）
        mem_allocated_avg_MB_list = [item["mem_allocated_avg_MB"] for item in baseline_data] 
        mem_max_allocated_avg_MB_list = [item["mem_max_allocated_avg_MB"] for item in baseline_data] 

        result = {
            "lm_loss": lm_loss_list,
            "grad_norm": grad_norm_list,
            "elapsed_time_per_iteration": elapsed_time_list,
            "throughput": throughput_list,
            "mem_allocated_avg_MB": mem_allocated_avg_MB_list,
            "mem_max_allocated_avg_MB": mem_max_allocated_avg_MB_list
        }
        return result

    @staticmethod
    def get_baseline_file_path(model_config, model_name):
        """
        获取 baseline 文件路径，根据模型来源目录选择 default 或 optional 目录下的 baseline。
        Args:
            model_config: 模型配置字典
            model_name: 模型名称
        Returns:
            baseline 文件的完整路径
        """
        # 优先从 model 配置中获取 BASELINE_PATH
        baseline_path = model_config.get("BASELINE_PATH")
        if baseline_path and os.path.isdir(baseline_path):
            baseline_file = os.path.join(baseline_path, f"{model_name}.json")
        else:
            # 判断模型来源目录
            config_source = model_config.get("_config_source", {})
            config_dir = config_source.get("dir", "")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            baseline_root = os.path.join(current_dir, "..", "baseline")
            if "optional_configs" in config_dir:
                baseline_file = os.path.join(baseline_root, "optional", f"{model_name}.json")
            else:
                baseline_file = os.path.join(baseline_root, "default", f"{model_name}.json")
        if not os.path.exists(baseline_file):
            raise FileNotFoundError(f"Baseline file not found: {baseline_file}")
        return baseline_file

    @staticmethod
    def get_baseline_data(self_unused, model_config, model_name, training_type=None):
        """获取 baseline 数据（供 base_task 调用）
        
        Args:
            self_unused: 占位参数，保持接口一致性（可传 None）
            model_config: 模型配置字典
            model_name: 模型名称
            training_type: 训练类型（如 'pretrain', 'sft'）
        
        Returns:
            baseline 数据列表，每个元素包含 lm_loss, grad_norm, elapsed_time_ms, throughput
        """
        baseline_file = ConfigManager.get_baseline_file_path(model_config, model_name)
        
        with open(baseline_file, 'r') as f:
            data = json.load(f)
        
        # 如果是列表，直接返回
        if isinstance(data, list):
            return data
        
        # 如果是字典，根据 training_type 获取对应数据
        if isinstance(data, dict):
            if training_type and training_type in data:
                return data[training_type]
            else:
                available_keys = list(data.keys())
                raise ValueError(f"training_type '{training_type}' 未指定或在 {baseline_file} 中不存在，可用的 training_type: {available_keys}")
        
        raise ValueError(f"Invalid JSON structure in {baseline_file}")

    def format_baseline_data_for_yaml(self, baseline_data):
        """将基准数据格式化为YAML字符串格式"""
        formatted_data = {}
        
        for key, value_list in baseline_data.items():
            # 将数值列表格式化为字符串，保留科学计数法格式
            formatted_str = "[\n"
            for item in value_list:
                formatted_str += f"  {item:.6E}\n"
            formatted_str += "]"
            formatted_data[key] = formatted_str
        
        return formatted_data

    def load_config(self, configs_dir: str, model_file: str) -> Dict[Any, Any]:
        # 优先使用当前目录的 common.yaml，如果不存在则使用主配置目录的
        common_config_file = os.path.join(configs_dir, "common.yaml")
        if not os.path.exists(common_config_file):
            # 回退到主配置目录的 common.yaml
            common_config_file = os.path.join(self.args.configs_dir, "common.yaml")
        
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

        # 检查是否有基准JSON文件路径配置，如果有则动态加载数据
        scenarios = config.get("scenarios", [])
        for scenario in scenarios:
            for scenario_type, scenario_data in scenario.items():
                if scenario_type == "function":
                    for training_type, training_data in scenario_data.items():
                        for step_name, step_data in training_data.items():
                            baseline_json_path = step_data.get("baseline_json_path")
                            if baseline_json_path and os.path.exists(baseline_json_path):
                                # 从JSON文件加载基准数据，传入 training_type
                                baseline_data = self.load_baseline_data_from_json(baseline_json_path, training_type)
                                # 将基准数据格式化为YAML格式
                                formatted_data = self.format_baseline_data_for_yaml(baseline_data)
                                # 更新配置数据
                                step_data.update(formatted_data)
                                logger.info(f"从JSON文件 {baseline_json_path} 动态加载 {training_type} 基准数据成功")

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
    def get_models_from_dir(configs_dir: str, recursive: bool = False, base_dir: str = None) -> List[str]:
        """从配置目录获取所有模型名称
        
        Args:
            configs_dir: 配置目录路径
            recursive: 是否递归扫描子目录
            base_dir: 用于计算相对路径的基础目录（用于生成带前缀的模型名）
        
        Returns:
            模型名称列表。如果是递归扫描子目录，返回格式为 "子目录/model_name"
        """
        models = []
        if not os.path.exists(configs_dir):
            return models
        
        if base_dir is None:
            base_dir = configs_dir
        
        try:
            items = os.listdir(configs_dir)
        except PermissionError:
            return models
        
        for item in items:
            item_path = os.path.join(configs_dir, item)
            
            if os.path.isfile(item_path):
                # 处理 yaml 文件
                name = os.path.splitext(item)[0]
                if not name.startswith("common") and item.endswith(".yaml"):
                    # 如果在子目录中，添加相对路径前缀
                    if configs_dir != base_dir:
                        rel_dir = os.path.relpath(configs_dir, base_dir)
                        models.append(f"{rel_dir}/{name}")
                    else:
                        models.append(name)
            elif os.path.isdir(item_path) and recursive:
                # 递归扫描子目录
                sub_models = ConfigManager.get_models_from_dir(
                    item_path, recursive=True, base_dir=base_dir
                )
                models.extend(sub_models)
        
        return models

    @staticmethod
    def get_models_from_subdir(base_dir: str, subdir: str) -> List[str]:
        """从指定子目录获取所有模型名称
        
        Args:
            base_dir: 基础目录（如 "optional_configs"）
            subdir: 子目录名称（如 "internvl3.5"）
        
        Returns:
            模型名称列表，格式为 "子目录/model_name"
        """
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(subdir_path) or not os.path.isdir(subdir_path):
            return []
        
        models = []
        for item in os.listdir(subdir_path):
            if item.endswith(".yaml") and not item.startswith("common"):
                name = os.path.splitext(item)[0]
                models.append(f"{subdir}/{name}")
        
        return models

    @staticmethod
    def get_all_models(configs_dir: str, 
                       extra_configs_dirs: List[str] = None,
                       recursive_extra: bool = True) -> List[str]:
        """从所有配置目录获取模型列表
        
        Args:
            configs_dir: 主配置目录（不递归）
            extra_configs_dirs: 额外配置目录列表（默认递归扫描）
            recursive_extra: 是否递归扫描额外配置目录
        
        Returns:
            模型名称列表
        """
        all_models = set()
        
        # 从主配置目录获取模型（不递归）
        all_models.update(ConfigManager.get_models_from_dir(configs_dir, recursive=False))
        
        # 从额外配置目录获取模型（默认递归）
        if extra_configs_dirs:
            for extra_dir in extra_configs_dirs:
                all_models.update(ConfigManager.get_models_from_dir(
                    extra_dir, recursive=recursive_extra, base_dir=extra_dir
                ))
        
        return sorted(list(all_models))

    @staticmethod
    def list_all_available_models(configs_dir: str, extra_configs_dirs: List[str] = None) -> Dict[str, List[str]]:
        """列出所有可用模型及其来源目录
        
        Args:
            configs_dir: 主配置目录
            extra_configs_dirs: 额外配置目录列表
        
        Returns:
            字典，key 为目录名，value 为该目录下的模型列表
        """
        result = {}
        
        # 主配置目录（不递归）
        models = ConfigManager.get_models_from_dir(configs_dir, recursive=False)
        if models:
            result[configs_dir] = sorted(models)
        
        # 额外配置目录（递归扫描）
        if extra_configs_dirs:
            for extra_dir in extra_configs_dirs:
                models = ConfigManager.get_models_from_dir(extra_dir, recursive=True, base_dir=extra_dir)
                if models:
                    result[extra_dir] = sorted(models)
        
        return result

    def find_config_file(self, model_name: str) -> Tuple[Optional[str], Optional[str]]:
        """查找模型配置文件所在的目录和实际文件名
        
        Args:
            model_name: 模型名称，支持格式：
                - "qwen2.5_vl_7b" - 简单模型名
                - "internvl3.5/internvl3.5_30b_a3b" - 带子目录前缀的模型名
        
        Returns:
            元组 (configs_dir, yaml_filename)，如果未找到返回 (None, None)
        """
        # 解析模型名称，判断是否包含子目录路径
        if "/" in model_name:
            # 带路径前缀的模型名，如 "internvl3.5/internvl3.5_30b_a3b"
            sub_path, actual_model_name = model_name.rsplit("/", 1)
            yaml_filename = f"{actual_model_name}.yaml"
            
            # 在额外配置目录的子目录中查找
            extra_dirs = getattr(self.args, 'extra_configs_dirs', []) or []
            for extra_dir in extra_dirs:
                config_file = os.path.join(extra_dir, sub_path, yaml_filename)
                if os.path.exists(config_file):
                    return os.path.join(extra_dir, sub_path), yaml_filename
        else:
            # 简单模型名
            yaml_filename = f"{model_name}.yaml"
            
            # 首先在主配置目录查找
            config_file = os.path.join(self.args.configs_dir, yaml_filename)
            if os.path.exists(config_file):
                return self.args.configs_dir, yaml_filename
            
            # 然后在额外配置目录查找（包括子目录）
            extra_dirs = getattr(self.args, 'extra_configs_dirs', []) or []
            for extra_dir in extra_dirs:
                # 先在根目录查找
                config_file = os.path.join(extra_dir, yaml_filename)
                if os.path.exists(config_file):
                    return extra_dir, yaml_filename
                
                # 递归在子目录中查找
                found = ConfigManager._find_yaml_in_subdir(extra_dir, yaml_filename)
                if found:
                    return found
        
        return None, None

    @staticmethod
    def _find_yaml_in_subdir(base_dir: str, yaml_filename: str) -> Optional[Tuple[str, str]]:
        """在子目录中递归查找 yaml 文件
        
        Args:
            base_dir: 基础目录
            yaml_filename: yaml 文件名
        
        Returns:
            元组 (目录路径, 文件名)，如果未找到返回 None
        """
        if not os.path.exists(base_dir):
            return None
        
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                config_file = os.path.join(item_path, yaml_filename)
                if os.path.exists(config_file):
                    return item_path, yaml_filename
                # 继续递归
                found = ConfigManager._find_yaml_in_subdir(item_path, yaml_filename)
                if found:
                    return found
        return None

    def get_scenarios_num(self) -> int:
        # 模型数量 + 场景数量。暂先不考虑任务数量
        # tasks = ConfigManager.get_tasks(config)
        num = 0
        for models in self.all_model_configs:
            # num += len(models["scenarios"].keys())
            num += 1
        return num