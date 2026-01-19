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
        """Get task runner (delayed import to avoid circular dependencies)"""
        from tasks import SUPPORTED_TASKS
        return SUPPORTED_TASKS[task_name]
    
    def get_all_model_configs(self, configs_dir: str) -> List[Dict[Any, Any]]:
        all_model_configs: List[Dict[Any, Any]] = []
        
        for model_name in self.args.models:
            try:
                # Find the directory and filename where the config file is located
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
                    
                    # Add config source information for debugging and distinguishing models with the same name
                    config["_config_source"] = {
                        "dir": actual_configs_dir,
                        "file": yaml_filename,
                        "model_identifier": model_name  # Original model identifier (may include path prefix)
                    }
                    all_model_configs.append(config)
            except Exception as e:
                logger.warning(f"Try to load config for '{model_name}', but failed: {e}")
                raise e

        return all_model_configs

    def load_baseline_data_from_json(self, json_path, training_type=None):
        """Load baseline data from JSON file

        Args:
            json_path: JSON file path
            training_type: Training type (e.g., 'pretrain', 'sft'), used to select corresponding data from JSON
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Baseline data file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            baseline_data = json.load(f)
        
        # Get corresponding data based on training_type
        if isinstance(baseline_data, dict):
            if training_type and training_type in baseline_data:
                baseline_data = baseline_data[training_type]
            else:
                available_keys = list(baseline_data.keys())
                raise ValueError(f"training_type '{training_type}' is not specified or does not exist in {json_path}, available training_type: {available_keys}")
        
        # Extract required data by iteration order
        lm_loss_list = [item["lm_loss"] for item in baseline_data]
        grad_norm_list = [item["grad_norm"] for item in baseline_data]
        elapsed_time_list = [item["elapsed_time_ms"] for item in baseline_data]
        throughput_list = [item["throughput"] for item in baseline_data]

        # Add new memory metrics (optional)
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
        Get baseline file path, select baseline from default or optional directory based on model source directory.
        Args:
            model_config: Model configuration dictionary
            model_name: Model name
        Returns:
            Full path to baseline file
        """
        # Prefer getting BASELINE_PATH from model config
        baseline_path = model_config.get("BASELINE_PATH")
        if baseline_path and os.path.isdir(baseline_path):
            baseline_file = os.path.join(baseline_path, f"{model_name}.json")
        else:
            # Determine model source directory
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
        """Get baseline data (for base_task to call)

        Args:
            self_unused: Placeholder parameter, keep interface consistent (can pass None)
            model_config: Model configuration dictionary
            model_name: Model name
            training_type: Training type (e.g., 'pretrain', 'sft')

        Returns:
            Baseline data list, each element contains lm_loss, grad_norm, elapsed_time_ms, throughput
        """
        baseline_file = ConfigManager.get_baseline_file_path(model_config, model_name)
        
        with open(baseline_file, 'r') as f:
            data = json.load(f)
        
        # If it's a list, return directly
        if isinstance(data, list):
            return data
        
        # If it's a dictionary, get corresponding data by training_type
        if isinstance(data, dict):
            if training_type and training_type in data:
                return data[training_type]
            else:
                available_keys = list(data.keys())
                raise ValueError(f"training_type '{training_type}' is not specified or does not exist in {baseline_file}, available training_type: {available_keys}")
        
        raise ValueError(f"Invalid JSON structure in {baseline_file}")

    def format_baseline_data_for_yaml(self, baseline_data):
        """Format baseline data into YAML string format"""
        formatted_data = {}
        
        for key, value_list in baseline_data.items():
            # Format numeric list as string, keep scientific notation format
            formatted_str = "[\n"
            for item in value_list:
                formatted_str += f"  {item:.6E}\n"
            formatted_str += "]"
            formatted_data[key] = formatted_str
        
        return formatted_data

    def load_config(self, configs_dir: str, model_file: str) -> Dict[Any, Any]:
        # Prefer using common.yaml from current directory, fall back to main config directory if not exists
        common_config_file = os.path.join(configs_dir, "common.yaml")
        if not os.path.exists(common_config_file):
            # Fall back to main config directory's common.yaml
            common_config_file = os.path.join(self.args.configs_dir, "common.yaml")
        
        model_config_file = os.path.join(configs_dir, model_file)

        def recursive_replace(match):
            key = match.group(1)
            return str(config.get(key, match.group(0)))

        # Load common.yaml
        with open(common_config_file, 'r') as f:
            common_config_str = f.read()

        # Load chatglm-6b.yaml
        with open(model_config_file, 'r') as f:
            model_config_str = f.read()

        # Parse content of both config files as dictionaries
        common_config = yaml.safe_load(common_config_str)
        model_config = yaml.safe_load(model_config_str)

        # Merge common_config and model_config
        config = {**common_config, **model_config}

        # Recursively replace all placeholders
        for _ in range(10):  # Set max iterations to 10
            new_common_config_str = re.sub(r'\$([a-z_][a-z0-9_]*)', recursive_replace, common_config_str)
            new_model_config_str = re.sub(r'\$([a-z_][a-z0-9_]*)', recursive_replace, model_config_str)

            if new_common_config_str == common_config_str and new_model_config_str == model_config_str:
                # If neither string has changed, end loop early
                break

            common_config_str = new_common_config_str
            model_config_str = new_model_config_str

            # Update config
            common_config = yaml.safe_load(common_config_str)
            model_config = yaml.safe_load(model_config_str)
            config = {**common_config, **model_config}

        # Check if there is baseline JSON file path config, if so load data dynamically
        scenarios = config.get("scenarios", [])
        for scenario in scenarios:
            for scenario_type, scenario_data in scenario.items():
                if scenario_type == "function":
                    for training_type, training_data in scenario_data.items():
                        for step_name, step_data in training_data.items():
                            baseline_json_path = step_data.get("baseline_json_path")
                            if baseline_json_path and os.path.exists(baseline_json_path):
                                # Load baseline data from JSON file, pass training_type
                                baseline_data = self.load_baseline_data_from_json(baseline_json_path, training_type)
                                # Format baseline data as YAML format
                                formatted_data = self.format_baseline_data_for_yaml(baseline_data)
                                # Update config data
                                step_data.update(formatted_data)
                                logger.info(f"Successfully loaded {training_type} baseline data from JSON file {baseline_json_path}")

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
        """Get all model names from config directory

        Args:
            configs_dir: Config directory path
            recursive: Whether to recursively scan subdirectories
            base_dir: Base directory for calculating relative path (used to generate model names with prefix)

        Returns:
            Model name list. If recursively scanning subdirectories, returns in "subdirectory/model_name" format
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
                # Handle yaml files
                name = os.path.splitext(item)[0]
                if not name.startswith("common") and item.endswith(".yaml"):
                    # If in subdirectory, add relative path prefix
                    if configs_dir != base_dir:
                        rel_dir = os.path.relpath(configs_dir, base_dir)
                        models.append(f"{rel_dir}/{name}")
                    else:
                        models.append(name)
            elif os.path.isdir(item_path) and recursive:
                # Recursively scan subdirectories
                sub_models = ConfigManager.get_models_from_dir(
                    item_path, recursive=True, base_dir=base_dir
                )
                models.extend(sub_models)
        
        return models

    @staticmethod
    def get_models_from_subdir(base_dir: str, subdir: str) -> List[str]:
        """Get all model names from specified subdirectory

        Args:
            base_dir: Base directory (e.g., "optional_configs")
            subdir: Subdirectory name (e.g., "internvl3.5")

        Returns:
            Model name list, in "subdirectory/model_name" format
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
        """Get model list from all config directories

        Args:
            configs_dir: Main config directory (not recursive)
            extra_configs_dirs: Extra config directories list (default recursive scan)
            recursive_extra: Whether to recursively scan extra config directories

        Returns:
            Model name list
        """
        all_models = set()
        
        # Get models from main config directory (not recursive)
        all_models.update(ConfigManager.get_models_from_dir(configs_dir, recursive=False))
        
        # Get models from extra config directories (default recursive)
        if extra_configs_dirs:
            for extra_dir in extra_configs_dirs:
                all_models.update(ConfigManager.get_models_from_dir(
                    extra_dir, recursive=recursive_extra, base_dir=extra_dir
                ))
        
        return sorted(list(all_models))

    @staticmethod
    def list_all_available_models(configs_dir: str, extra_configs_dirs: List[str] = None) -> Dict[str, List[str]]:
        """List all available models and their source directories

        Args:
            configs_dir: Main config directory
            extra_configs_dirs: Extra config directories list

        Returns:
            Dictionary with directory name as key and model list in that directory as value
        """
        result = {}
        
        # Main config directory (not recursive)
        models = ConfigManager.get_models_from_dir(configs_dir, recursive=False)
        if models:
            result[configs_dir] = sorted(models)
        
        # Extra config directories (recursive scan)
        if extra_configs_dirs:
            for extra_dir in extra_configs_dirs:
                models = ConfigManager.get_models_from_dir(extra_dir, recursive=True, base_dir=extra_dir)
                if models:
                    result[extra_dir] = sorted(models)
        
        return result

    def find_config_file(self, model_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Find the directory and actual filename of model config file

        Args:
            model_name: Model name, supports formats:
                - "qwen2.5_vl_7b" - Simple model name
                - "internvl3.5/internvl3.5_30b_a3b" - Model name with subdirectory prefix

        Returns:
            Tuple (configs_dir, yaml_filename), return (None, None) if not found
        """
        # Parse model name, check if it contains subdirectory path
        if "/" in model_name:
            # Model name with path prefix, like "internvl3.5/internvl3.5_30b_a3b"
            sub_path, actual_model_name = model_name.rsplit("/", 1)
            yaml_filename = f"{actual_model_name}.yaml"
            
            # Search in subdirectories of extra config directories
            extra_dirs = getattr(self.args, 'extra_configs_dirs', []) or []
            for extra_dir in extra_dirs:
                config_file = os.path.join(extra_dir, sub_path, yaml_filename)
                if os.path.exists(config_file):
                    return os.path.join(extra_dir, sub_path), yaml_filename
        else:
            # Simple model name
            yaml_filename = f"{model_name}.yaml"

            # First search in main config directory
            config_file = os.path.join(self.args.configs_dir, yaml_filename)
            if os.path.exists(config_file):
                return self.args.configs_dir, yaml_filename
            
            # Then search in extra config directories (including subdirectories)
            extra_dirs = getattr(self.args, 'extra_configs_dirs', []) or []
            for extra_dir in extra_dirs:
                # First search in root directory
                config_file = os.path.join(extra_dir, yaml_filename)
                if os.path.exists(config_file):
                    return extra_dir, yaml_filename
                
                # Recursively search in subdirectories
                found = ConfigManager._find_yaml_in_subdir(extra_dir, yaml_filename)
                if found:
                    return found
        
        return None, None

    @staticmethod
    def _find_yaml_in_subdir(base_dir: str, yaml_filename: str) -> Optional[Tuple[str, str]]:
        """Recursively find yaml file in subdirectories

        Args:
            base_dir: Base directory
            yaml_filename: yaml file name

        Returns:
            Tuple (directory path, filename), return None if not found
        """
        if not os.path.exists(base_dir):
            return None
        
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                config_file = os.path.join(item_path, yaml_filename)
                if os.path.exists(config_file):
                    return item_path, yaml_filename
                # Continue recursively
                found = ConfigManager._find_yaml_in_subdir(item_path, yaml_filename)
                if found:
                    return found
        return None

    def get_scenarios_num(self) -> int:
        # Model count + scenario count. Not considering task count for now
        # tasks = ConfigManager.get_tasks(config)
        num = 0
        for models in self.all_model_configs:
            # num += len(models["scenarios"].keys())
            num += 1
        return num