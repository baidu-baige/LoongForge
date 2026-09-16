# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""base_task.py"""

import os, time
from typing import Any, Dict, List
import shutil
import os, time
from copy import deepcopy
import json, re, yaml
import subprocess
import sys
import random
import torch
import numpy as np
from typing import Dict, List, Any
from tools.color_logger import create_color_logger
from tools.config_manager import ConfigManager
from utils.constants import task_created_flag, task_running_flag, task_finish_flag
from metric.metric import Metric

logger = create_color_logger(name=__name__)

# HF checkpoint check model layer mapping configuration
HF_CHECK_MODEL_MAPPING = {
    "Qwen2.5-VL-7B-Instruct": {
        "llm": ["model.layers.", 28],
        "vit": ["visual.blocks.", 32]
    },
    "InternVL2_5-8B": {
        "llm": ["language_model.model.layers.", 32],
        "vit": ["vision_model.encoder.layers.", 24]
    },
    "InternVL3_5-8B": {
        "llm": ["language_model.model.layers.", 36],
        "vit": ["vision_tower.encoder.layer.", 24]
    },
    "LLaVA-OneVision-1.5-4B": {
        "llm": ["model.layers.", 36],
        "vit": ["visual.blocks.", 24]
    },
}


class TaskResut(object):
    pass

    def __str__(self) -> str:
        pass

class BaseTask(object):
    _validation_results = []

    def __init__(self,
                 model_description: Dict[str, Any],
                 task_description: Dict[str, Any],
                 model: List[Dict[Any, Any]],
                 model_configer: object,
                 args,
                 task_type: str = "function",
                 ) -> None:
        self.model_description = model_description
        self.task_description = task_description
        self.model = model
        self.model_configer = model_configer
        self.input_cmd_args = args
        self.MODEL_RUNNABLE = model["MODEL_RUNNABLE"]
        self.model_name = model["model_name"]
        self.world_size = os.getenv('WORLD_SIZE')
        self.rank = os.getenv('RANK')
        self.rank_name = f'rank_{self.rank}'
        self.master_addr = os.getenv('MASTER_ADDR')
        self.is_final_pod = str(self.rank) == str(args.node_nums - 1)
        self.accuracy_relative_tolerance = args.accuracy_relative_tolerance
        self.performance_relative_tolerance = args.performance_relative_tolerance

        self.task_type = task_type

        self.metric = Metric()

    @classmethod
    def _get_diff_category(cls, model) -> str:
        config_source = model.get("_config_source", {}) if isinstance(model, dict) else {}
        config_dir = config_source.get("dir", "")
        if "optional_configs" in config_dir:
            return "optional"
        return "default"

    @classmethod
    def _resolve_diff_base_dir(cls, category: str) -> str:
        try:
            common_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "common.yaml"))
            if os.path.exists(common_yaml):
                training_log_path = cls._resolve_training_log_path(common_yaml)
                if training_log_path:
                    return os.path.join(training_log_path, "E2E", "diff", category)
        except Exception:
            pass
        return os.path.join(os.getcwd(), "E2E", "diff", category)

    @staticmethod
    def _resolve_training_log_path(common_yaml: str) -> str:
        with open(common_yaml, "r") as f:
            cfg = yaml.safe_load(f) or {}
        training_log_path = cfg.get("training_log_path")
        if not training_log_path:
            return ""
        pfs_path = cfg.get("pfs_path", "")
        if pfs_path:
            training_log_path = training_log_path.replace("${pfs_path}", pfs_path)
            training_log_path = training_log_path.replace("$pfs_path", pfs_path)
        return training_log_path

    @classmethod
    def _record_case_result(
        cls,
        model_name: str,
        training_type: str,
        category: str,
        passed: bool,
        failed_metrics: List[str],
        error_message: str = "",
        task_name: str = "",
    ):
        cls._validation_results.append({
            "model_name": model_name,
            "training_type": training_type or "unknown",
            "category": category or "default",
            "passed": bool(passed),
            "failed_metrics": failed_metrics or [],
            "error_message": error_message or "",
            "task_name": task_name or "",
        })

    @classmethod
    def write_validation_summary(cls):
        grouped = {}
        for item in cls._validation_results:
            grouped.setdefault(item["category"], []).append(item)

        if not grouped:
            grouped = {"default": []}

        for category, items in grouped.items():
            base_dir = cls._resolve_diff_base_dir(category)
            os.makedirs(base_dir, exist_ok=True)
            output_file = os.path.join(base_dir, "output.log")
            total = len(items)
            failed = sum(1 for x in items if not x["passed"])
            passed = total - failed
            lines = []
            lines.append(f"total_cases={total}")
            lines.append(f"passed_cases={passed}")
            lines.append(f"failed_cases={failed}")
            lines.append("")
            for item in items:
                status = "PASSED" if item["passed"] else "FAILED"
                case_name = f"{item['model_name']}#{item['training_type']}"
                error_message = item.get("error_message", "")
                if item["failed_metrics"]:
                    metrics = ",".join(item["failed_metrics"])
                    if error_message:
                        lines.append(f"{status}\t{case_name}\tfailed_metrics={metrics}\terror={error_message}")
                    else:
                        lines.append(f"{status}\t{case_name}\tfailed_metrics={metrics}")
                else:
                    if error_message:
                        lines.append(f"{status}\t{case_name}\terror={error_message}")
                    else:
                        lines.append(f"{status}\t{case_name}")

            with open(output_file, "w") as f:
                f.write("\n".join(lines))
            logger.info(f"Validation summary written: {output_file}")

    @classmethod
    def has_validation_failures(cls) -> bool:
        return any(not item.get("passed") for item in cls._validation_results)

    # Update lock file
    def update_lock_file(self, lock_file_path, task_flag):
        with open(lock_file_path, "a") as file:
            logger.info(f"update_lock_file {task_flag} start ...")
            file.seek(0) # Reset file pointer to beginning
            file.truncate() # Clear file content
            file.write(task_flag)
            logger.info(f"update_lock_file {task_flag} end")

    # Check lock file
    def check_lock_file(self, lock_file_path, task_flag):
        # Traverse the total number of file contents under lock_file_path folder to see if it equals expected, if equal True, else False, then check if content is all task_flag, if so True, else False
        count = 0
        flag = True

        parent_path = os.path.dirname(lock_file_path)
        expected_files = {f"rank_{i}_lock.txt" for i in range(self.input_cmd_args.node_nums)}
        for filename in os.listdir(parent_path):
            if filename not in expected_files:
                continue
            with open(os.path.join(parent_path, filename), "r") as f:
                content = f.read().strip()
                if content == task_flag:
                    count += 1
                else:
                    flag = False

        return count, (count == self.input_cmd_args.node_nums and flag)
    
    def wait_async_task_complete(self, lock_file_path, task_flag, model_name="", scenarios_name=""):
        # Check lock file, if all Pods are completed, enter next stage
        for _ in range(self.input_cmd_args.timeout // 10):
            _, state = self.check_lock_file(lock_file_path, task_flag)
            if state:
                logger.info(f"Model [{model_name}] {scenarios_name} all Pods completed")
                # if self.is_final_pod:
                #     parent_path = os.path.dirname(os.path.dirname(lock_file_path))
                #     shutil.rmtree(parent_path)
                break
            else:
                logger.info(f"Model [{model_name}] {scenarios_name} still has pending Pods, waiting...")
                time.sleep(10)
        else:
            raise TimeoutError(f"ERROR: Waited for other Pods more than {self.input_cmd_args.timeout} seconds")

    # lock_file exists and content == MASTER_ADDR, proving this task has completed writing to file, can continue execution
    # Otherwise, if not exists then create, and write content MASTER_ADDR
    def initialize_lock_file(self, lock_file, task_flag):
        # Check if directory exists, if not create it
        if not os.path.exists(os.path.dirname(lock_file)):
            os.makedirs(os.path.dirname(lock_file))

        # Open or create lock file
        with open(lock_file, 'a+') as file:
            logger.info(f"initialize_lock_file start ...")
            file.seek(0) # Reset file pointer to beginning
            file.truncate() # Clear file content
            file.write(task_flag)
            logger.info(f"initialize_lock_file end")
    
    def wait_async_pod_complete(self, lock_file, model_name="", scenarios_name="", is_function=False, function=None, raise_on_error=False, *args, **kwargs):
        logger.info(f"Model [{model_name}] {scenarios_name} waiting for other Pods to complete...")
        task_flag = task_finish_flag

        try:
            # Check if directory exists, if not create it
            if not os.path.exists(os.path.dirname(lock_file)):
                os.makedirs(os.path.dirname(lock_file))
        except Exception as e:
            logger.warning(f"warning: Failed to create directory, {e}")
            pass
        
        # Step 1: Write corresponding pod data to lock_file
        # Open or create lock file
        with open(lock_file, 'a+') as file:
            file.seek(0) # Reset file pointer to beginning
            file.truncate() # Clear file content
            file.write(task_flag)

        # Step 2: Check lock_file, if all Pods completed, enter next stage: 1. All file states are Finish. 2. File count is node_nums.
        for _ in range(self.input_cmd_args.timeout // 10):
            finish_count, state = self.check_lock_file(lock_file, task_flag)
            if state:
                logger.info(f"Model [{model_name}] {scenarios_name} all Pods completed, {finish_count} / {self.input_cmd_args.node_nums}")

                if is_function and self.is_final_pod:
                    if function and callable(function):
                        try:
                            logger.info(f"Executing {function.__name__} method:")
                            result = function(*args, **kwargs)  # Call function and pass arguments
                            logger.info(f"Function {function.__name__} execution completed, return value: {result}")
                        except Exception as e:
                            logger.error(f"Error executing function {function.__name__}: {e}")
                            # If raise_on_error parameter is True, raise exception
                            if raise_on_error:
                                raise Exception(f"Error executing function {function.__name__}: {e}")
                    else:
                        logger.error(f"Provided {function} is not a callable function.")
                break
            else:
                logger.info(f"Model [{model_name}] {scenarios_name} has pending Pods, {finish_count} / {self.input_cmd_args.node_nums} waiting...")
                time.sleep(10)
        else:
            raise TimeoutError(f"ERROR: Waited for other Pods more than {self.input_cmd_args.timeout} seconds")
    

    def __init_ckpt_file__(self) -> None:
        # Initialize data scenario, first entry needs to delete some files to avoid affecting subsequent tests
        scenario_name = ".init"
        init_lock_path = os.path.join(self.model["model_lock_file_path"], scenario_name, self.master_addr)
        init_lock_file = os.path.join(init_lock_path, f"{self.rank_name}_lock.txt")

        # Wait for all pods to be ready, then delete existing model ckpt files
        is_function = True
        if self.task_type == "perf":
            is_function = False

        self.wait_async_pod_complete(
            init_lock_file,
            self.model_name,
            f"{scenario_name}_{self.rank_name}",
            is_function=is_function,
            function=self.__del_ckpt_file__
        )

    def __del_ckpt_file__(self):
        step1_output_path = self.model["step1_output_path"]
        # Delete step1_output_path output files
        if os.path.exists(step1_output_path):
            shutil.rmtree(step1_output_path)

    def __clean_up__(self):
        # Initialize data scenario, first entry needs to delete some files to avoid affecting subsequent tests
        scenario_name = ".clean_up"
        init_lock_path = os.path.join(self.model["model_lock_file_path"], scenario_name, self.master_addr)
        init_lock_file = os.path.join(init_lock_path, f"{self.rank_name}_lock.txt")

        # Wait for all pods to be ready, then delete existing model ckpt files
        self.wait_async_pod_complete(
            init_lock_file,
            self.model_name,
            f"{scenario_name}_{self.rank_name}",
            is_function=True,
            function=self.__del_ckpt_file__
        )

    def __init_model_scenarios_data__(self, index, scenarios_name, step_name, training_type_name=None) -> None:
        model = deepcopy(self.model)
        function_step_data = {}
        
        if self.task_type == "perf":
            function_scenarios_name = "function"
            function_step_data = self.model["scenarios"][0][function_scenarios_name][training_type_name][step_name]
            # Merge function_step_data and model
            model = {**model, **function_step_data}

            # Normally get data under scenarios_name
            step_data = self.model["scenarios"][index][scenarios_name][training_type_name][step_name]
            # Merge function_step_data and model into one json
            model = {**model, **step_data}

        elif self.task_type == "function":
            # Iterate to get previous variables and replace with latest variables
            for i, scenario in enumerate(self.model["scenarios"]):
                local_index = i + 1
                local_step_name = f"Step{local_index}"

                function_scenarios_name = "function"
                # Check if step exists, if not skip
                if (local_step_name not in self.model["scenarios"][0][function_scenarios_name][training_type_name]):
                    logger.info(f"local_step_name {local_step_name} currently does not exist, skipping...")
                    continue

                function_step_data = self.model["scenarios"][0][function_scenarios_name][training_type_name][local_step_name]
                # Merge function_step_data and model
                model = {**model, **function_step_data}

                # Update variables to current step
                if local_step_name == step_name:
                    break
            
            # Normally get data under scenarios_name
            step_data = self.model["scenarios"][index][scenarios_name][training_type_name][step_name]
            # Merge function_step_data and model into one json
            model = {**model, **step_data}
        
        elif self.task_type == "preprocess_data":
            # Normally get data under scenarios_name
            step_data = self.model["scenarios"][index][scenarios_name][step_name]
            # Merge function_step_data and model into one json
            model = {**model, **step_data}

        # Set whether to use nccl
        use_nccl = "false"
        if self.input_cmd_args.use_nccl:
            use_nccl = "true"
        model["use_nccl"] = use_nccl

        dry_run = False
        if self.input_cmd_args.dry_run:
            dry_run = True
        model["dry_run"] = dry_run

        model["TRANING_MODEL"] = training_type_name
        
        return model


    def __convert_model_config_to_env__(self, model_config: Dict[Any, Any]) -> str:
        # Initialize an empty environment variable string
        env_vars_str = ""
        new_model_config = deepcopy(model_config)

        # Iterate dictionary, convert key-value pairs to environment variables
        for key, value in new_model_config.items():
            if key == "scenarios":
                continue

            if isinstance(value, str):
                # For _ARGS type parameters, special handling is required:
                # 1. Remove comment lines (lines starting with #)
                # 2. Convert newlines to spaces, merge multiple spaces into single space
                # 3. Strip leading and trailing whitespace
                # 4. Do not use json.dumps, pass raw string directly (bash will split by space)
                if "_ARGS" in key:
                    # Remove comment lines and empty lines
                    lines = []
                    for line in value.split('\n'):
                        stripped_line = line.strip()
                        # Skip empty lines and comment lines (including inline comments, e.g. "--arg # comment")
                        if stripped_line and not stripped_line.startswith('#'):
                            # Handle inline comments: remove # and content after it
                            if '#' in stripped_line:
                                # Check if # is inside quotes
                                in_quotes = False
                                quote_char = None
                                for i, char in enumerate(stripped_line):
                                    if char in ['"', "'"] and (i == 0 or stripped_line[i-1] != '\\'):
                                        if not in_quotes:
                                            in_quotes = True
                                            quote_char = char
                                        elif char == quote_char:
                                            in_quotes = False
                                            quote_char = None
                                    elif char == '#' and not in_quotes:
                                        stripped_line = stripped_line[:i].strip()
                                        break
                            lines.append(stripped_line)
                    # Join all non-comment lines with spaces, and normalize spaces
                    value = ' '.join(lines)
                    # Replace multiple consecutive spaces with single space
                    import re
                    value = re.sub(r'\s+', ' ', value).strip()
                    # For _ARGS, need to wrap the whole value in quotes to ensure values with spaces are passed correctly
                    # Escape quotes in value, then wrap with quotes
                    value = value.replace('"', '\\"')
                    value = f'"{value}"'
                elif key == "paramters" or "loss_in_value" in key:
                    value = json.dumps(value)
            else:
                value = json.dumps(value)
                value = f"\"{value}\""
            
            env_vars_str += f"{key}={value} "
        return env_vars_str
    

    
    def assert_loongforge(self, training_log_file, training_type=None, **kwargs):
        logger.info(f"Start assert_loongforge ...")

        if not self.is_final_pod:
            return
        # Collect training metric indicators (prefer rank_0 log if needed)
        loss_count = self.collect_metrics(training_log_file, strict=False)
        logger.info(f"Parsed loss_count={loss_count} from log: {training_log_file}")
        if loss_count == 0:
            for candidate_log in self._resolve_training_logs(training_log_file, training_type):
                if not os.path.exists(candidate_log):
                    continue
                if candidate_log == training_log_file:
                    continue
                logger.warning(f"No loss found yet, retry with log {candidate_log}")
                self.metric = Metric()
                loss_count = self.collect_metrics(candidate_log, strict=False)
                logger.info(f"Parsed loss_count={loss_count} from log: {candidate_log}")
                if loss_count != 0:
                    training_log_file = candidate_log
                    break
        assert loss_count != 0, "Loss list for this task is empty, please check if training task is normal!!!"
        # Optional: auto collect baseline from log
        if getattr(self.input_cmd_args, "auto_collect_baseline", False):
            self.save_baseline_from_log(training_log_file, training_type)
        # Unified validation of accuracy and performance metrics
        self.validate_metrics(self.model_name, training_type)
        logger.info(f"End assert_loongforge")

    def validate_metrics(self, model_name, training_type=None):
        """
        Unified validation of training accuracy and performance metrics.
        Accuracy metrics (lm_loss, grad_norm) use relative_tolerance=0.02, performance metrics (elapsed_time_ms, throughput) use relative_tolerance=0.05.
        Args:
            model_name: Model name, used to locate baseline JSON file
            training_type: Training type (e.g. 'pretrain', 'sft')
        """
        chip = getattr(self.input_cmd_args, "chip", "default")
        baseline_data = ConfigManager.get_baseline_data(None, self.model, model_name, training_type, chip=chip)
        case_failed_metrics = []
        case_passed = True
        category = self._get_diff_category(self.model)

        # Accuracy metrics
        accuracy_relative_tolerance = self.accuracy_relative_tolerance
        # lm_loss
        if hasattr(self.metric, 'lm_loss_list') and self.metric.lm_loss_list and 'lm_loss' in baseline_data[0]:
            expected_loss_list = [item['lm_loss'] for item in baseline_data]
            lm_loss_passed = self._compare_metric(
                actual_list=self.metric.lm_loss_list,
                expected_list=expected_loss_list,
                metric_name="lm_loss",
                tolerance=accuracy_relative_tolerance,
                is_relative=False,
                raise_on_fail=False
            )
            if not lm_loss_passed:
                case_passed = False
                case_failed_metrics.append("lm_loss")
            self._plot_loss_diffs(expected_loss_list, self.metric.lm_loss_list, model_name, training_type)
        # grad_norm
        if hasattr(self.metric, 'grad_norm_list') and self.metric.grad_norm_list and 'grad_norm' in baseline_data[0]:
            expected_grad_norm_list = [item['grad_norm'] for item in baseline_data]
            if not self.input_cmd_args.check_loss_only:
                grad_norm_passed = self._compare_metric(
                    actual_list=self.metric.grad_norm_list,
                    expected_list=expected_grad_norm_list,
                    metric_name="grad_norm",
                    tolerance=accuracy_relative_tolerance,
                    is_relative=False,
                    raise_on_fail=False
                )
                if not grad_norm_passed:
                    case_passed = False
                    case_failed_metrics.append("grad_norm")
            else:
                logger.info("Skip grad_norm check due to check_loss_only=True")
            self._plot_metric_compare(
                expected_list=expected_grad_norm_list,
                actual_list=self.metric.grad_norm_list,
                model_name=model_name,
                training_type=training_type,
                metric_name="grad_norm",
                y_label="grad_norm"
            )
        
        logger.info("Accuracy metrics validation passed!")

        # Performance metrics
        performance_relative_tolerance = self.performance_relative_tolerance
        num_iters = min(len(self.metric.elapsed_time_match), len(baseline_data))
        if num_iters == 0:
            logger.warning("Not enough data for performance validation")
            return
        
        def _check_avg_metric(actual, expected, name, tol):
            if len(actual) == 0 or len(expected) == 0:
                return
            avg_actual = np.mean(actual)
            avg_expected = np.mean(expected)
            if avg_expected == 0:
                rel_err = 0 if avg_actual == 0 else float('inf')
            else:
                rel_err = abs(avg_actual - avg_expected) / abs(avg_expected)
            
            if rel_err <= tol:
                logger.info(f"{name} avg comparison: Actual: {avg_actual:.4f} vs Expected: {avg_expected:.4f}, Relative Error: {rel_err*100:.2f}%, Passed!")
            else:
                logger.warning(f"{name} avg comparison: Actual: {avg_actual:.4f} vs Expected: {avg_expected:.4f}, Relative Error: {rel_err*100:.2f}% Exceeds {tol*100:.0f}%, Failed (Warning Only)!")

        # elapsed_time_ms
        if hasattr(self.metric, 'elapsed_time_match') and self.metric.elapsed_time_match and 'elapsed_time_ms' in baseline_data[0]:
            expected_elapsed_time = [item['elapsed_time_ms'] for item in baseline_data[:num_iters]]
            actual_elapsed_time = [float(x) for x in self.metric.elapsed_time_match[:num_iters]]
            _check_avg_metric(actual_elapsed_time, expected_elapsed_time, "elapsed_time_ms", performance_relative_tolerance)
            self._plot_metric_compare(
                expected_list=expected_elapsed_time,
                actual_list=actual_elapsed_time,
                model_name=model_name,
                training_type=training_type,
                metric_name="elapsed_time_ms",
                y_label="elapsed_time_ms"
            )
        
        # throughput
        if hasattr(self.metric, 'throughput') and self.metric.throughput and 'throughput' in baseline_data[0]:
            expected_throughput = [item['throughput'] for item in baseline_data[:num_iters]]
            actual_throughput = [float(x) for x in self.metric.throughput[:num_iters]]
            _check_avg_metric(actual_throughput, expected_throughput, "throughput", performance_relative_tolerance)
            self._plot_metric_compare(
                expected_list=expected_throughput,
                actual_list=actual_throughput,
                model_name=model_name,
                training_type=training_type,
                metric_name="throughput",
                y_label="throughput"
            )

        # Memory metrics
        # mem_allocated_avg_MB
        if hasattr(self.metric, 'mem_allocated_avg_MB') and self.metric.mem_allocated_avg_MB and 'mem_allocated_avg_MB' in baseline_data[0]:
            expected_mem_allocated = [item['mem_allocated_avg_MB'] for item in baseline_data[:num_iters]]
            actual_mem_allocated = [float(x) for x in self.metric.mem_allocated_avg_MB[:num_iters]]
            _check_avg_metric(actual_mem_allocated, expected_mem_allocated, "mem_allocated_avg_MB", performance_relative_tolerance)
            self._plot_metric_compare(
                expected_list=expected_mem_allocated,
                actual_list=actual_mem_allocated,
                model_name=model_name,
                training_type=training_type,
                metric_name="mem_allocated_avg_MB",
                y_label="mem_allocated_avg_MB"
            )
        
        # mem_max_allocated_avg_MB
        if hasattr(self.metric, 'mem_max_allocated_avg_MB') and self.metric.mem_max_allocated_avg_MB and 'mem_max_allocated_avg_MB' in baseline_data[0]:
            expected_mem_max_allocated = [item['mem_max_allocated_avg_MB'] for item in baseline_data[:num_iters]]
            actual_mem_max_allocated = [float(x) for x in self.metric.mem_max_allocated_avg_MB[:num_iters]]
            _check_avg_metric(actual_mem_max_allocated, expected_mem_max_allocated, "mem_max_allocated_avg_MB", performance_relative_tolerance)
            self._plot_metric_compare(
                expected_list=expected_mem_max_allocated,
                actual_list=actual_mem_max_allocated,
                model_name=model_name,
                training_type=training_type,
                metric_name="mem_max_allocated_avg_MB",
                y_label="mem_max_allocated_avg_MB"
            )
            
        logger.info("Performance metrics validation passed (Soft Check)!")

        self._record_case_result(
            model_name,
            training_type,
            category,
            case_passed,
            case_failed_metrics,
            task_name=getattr(self, "task_description", ""),
        )
    
    def collect_metrics(self, training_log_file, strict=True):
        self.metric.model_name = self.model_name
        # Collect training metric indicators
        with open(training_log_file, 'r') as file:
            lines = file.readlines()

        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        # Clear to prevent mixing when there are many scenarios
        self.metric.lm_loss_list = []
        self.metric.grad_norm_list = []
        self.metric.throughput = []
        self.metric.elapsed_time_match = []
        self.metric.mem_allocated_avg_MB = []
        self.metric.mem_max_allocated_avg_MB = []

        # Add loss value
        for i in range(len(lines)):
            line = ansi_escape.sub('', lines[i])
            loss_match = re.search(r'lm[ _-]loss:?\s*([\d\.E\+-]+)', line)
            if loss_match:
                training_lm_loss_str = str(loss_match.group(1)).strip()
                self.metric.lm_loss_list.append(training_lm_loss_str)
            # Get grad_norm value
            grad_norm_match = re.search(r'grad norm: ([\d\.E\+-]+)', line)
            if grad_norm_match:
                grad_norm_str = str(grad_norm_match.group(1)).strip()
                self.metric.grad_norm_list.append(grad_norm_str)
            # Get 'throughput (token/sec/GPU)' value
            throughput_match = re.search(r'throughput \(token/sec/GPU\): ([\d\.E\+-]+)', line)
            if throughput_match:
                throughput_str = str(throughput_match.group(1)).strip()
                self.metric.throughput.append(throughput_str)
            # Get elapsed time per iteration (ms) value
            elapsed_time_match = re.search(r'elapsed time per iteration \(ms\): ([\d\.E\+-]+)', line)
            if elapsed_time_match:
                elapsed_time_str = str(elapsed_time_match.group(1)).strip()
                self.metric.elapsed_time_match.append(elapsed_time_str)
            # Get mem_allocated_avg_MB (training log format e.g.: mem-allocated-bytes-avg(MB): 15896.39)
            mem_allocated_match = re.search(r'mem-allocated-bytes-avg\(MB\):\s*([\d\.E\+-]+)', line)
            if mem_allocated_match:
                mem_allocated_str = str(mem_allocated_match.group(1)).strip()
                self.metric.mem_allocated_avg_MB.append(mem_allocated_str)
            # Get mem_max_allocated_avg_MB (training log format e.g.: mem-max-allocated-bytes-avg(MB): 29039.47)
            mem_max_allocated_match = re.search(r'mem-max-allocated-bytes-avg\(MB\):\s*([\d\.E\+-]+)', line)
            if mem_max_allocated_match:
                mem_max_allocated_str = str(mem_max_allocated_match.group(1)).strip()
                self.metric.mem_max_allocated_avg_MB.append(mem_max_allocated_str)
            # Get global batch size
            batch_size_match = re.search(r'global batch size:\s*(\d+)', line)
            if batch_size_match:
                batch_size_str = str(batch_size_match.group(1)).strip()
                self.metric.global_batch_size.append(batch_size_str)

        if strict:
            assert len(self.metric.lm_loss_list) != 0, "Loss list for this task is empty, please check if training task is normal!!!"
        return len(self.metric.lm_loss_list)
        # logger.info(f"self.metric dict: {self.metric.obj_to_dict()}")

    def _resolve_rank0_training_log(self, training_log_file, training_type=None):
        try:
            log_dir = os.path.dirname(training_log_file)
            node_nums = self.input_cmd_args.node_nums
            if training_type:
                return os.path.join(
                    log_dir,
                    f"training#{self.model_name}#{training_type}#nodes_{node_nums}#rank_0#run.log",
                )
            return re.sub(r"#rank_\d+#run\.log$", "#rank_0#run.log", training_log_file)
        except Exception:
            return None

    def _resolve_training_logs(self, training_log_file, training_type=None):
        try:
            import glob
            log_dir = os.path.dirname(training_log_file)
            if training_type:
                pattern = os.path.join(
                    log_dir,
                    f"training#{self.model_name}#{training_type}#nodes_*#rank_*#run.log",
                )
            else:
                pattern = os.path.join(log_dir, "training#*#rank_*#run.log")

            logs = glob.glob(pattern)

            def _rank_from_path(p):
                m = re.search(r"#rank_(\d+)#run\.log$", p)
                return int(m.group(1)) if m else -1

            logs.sort(key=lambda p: (_rank_from_path(p)), reverse=True)

            if training_log_file in logs:
                logs.remove(training_log_file)
                logs.insert(0, training_log_file)

            return logs
        except Exception:
            return []

    def save_baseline_from_log(self, training_log_file, training_type=None):
        from tools import log2json
        chip = getattr(self.input_cmd_args, "chip", "default")
        baseline_file = ConfigManager.get_baseline_file_path_for_write(self.model, self.model_name, chip=chip)
        phase, records = log2json.parse_log_file(training_log_file)
        if not records:
            logger.warning(f"No baseline records parsed from log: {training_log_file}")
            return

        baseline_key = training_type or phase or "unknown"

        if os.path.exists(baseline_file):
            try:
                with open(baseline_file, "r") as f:
                    baseline_data = json.load(f)
            except Exception:
                baseline_data = {}
        else:
            baseline_data = {}

        if isinstance(baseline_data, list):
            baseline_data = {baseline_key: baseline_data}
        elif not isinstance(baseline_data, dict):
            baseline_data = {}

        baseline_data[baseline_key] = records

        with open(baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)
        logger.info(f"Baseline saved to {baseline_file} (training_type={baseline_key}, records={len(records)})")

    def _plot_loss_diffs(self, expected_loss_list, actual_loss_list, model_name, training_type=None):
        try:
            from tools.diff_vis import save_loss_diff_plots
            output_dir = None
            category = "default"
            config_source = self.model.get("_config_source", {})
            config_dir = config_source.get("dir", "")
            if "optional_configs" in config_dir:
                category = "optional"
            common_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "common.yaml"))
            if os.path.exists(common_yaml):
                training_log_path = self._resolve_training_log_path(common_yaml)
                if training_log_path:
                    output_dir = os.path.join(training_log_path, "E2E", "diff", category, model_name)
            save_loss_diff_plots(
                baseline_list=expected_loss_list,
                current_list=actual_loss_list,
                model_name=model_name,
                training_type=training_type or "unknown",
                output_dir=output_dir,
                category=category
            )
        except Exception as e:
            logger.warning(f"Loss plot generation failed: {e}")

    def _plot_metric_compare(self, expected_list, actual_list, model_name, training_type, metric_name, y_label=None):
        try:
            from tools.diff_vis import save_metric_compare_plot
            output_dir = None
            category = "default"
            config_source = self.model.get("_config_source", {})
            config_dir = config_source.get("dir", "")
            if "optional_configs" in config_dir:
                category = "optional"
            common_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "common.yaml"))
            if os.path.exists(common_yaml):
                training_log_path = self._resolve_training_log_path(common_yaml)
                if training_log_path:
                    output_dir = os.path.join(training_log_path, "E2E", "diff", category, model_name)
            save_metric_compare_plot(
                baseline_list=expected_list,
                current_list=actual_list,
                model_name=model_name,
                training_type=training_type or "unknown",
                metric_name=metric_name,
                y_label=y_label,
                output_dir=output_dir,
                category=category
            )
        except Exception as e:
            logger.warning(f"Metric plot generation failed ({metric_name}): {e}")
      
    def _compare_metric(self, actual_list, expected_list, metric_name, tolerance, is_relative=False, raise_on_fail=True):
        """
        Generic metric comparison function, supports absolute and relative error, allows some iterations to exceed tolerance range
        :param actual_list: Actual value list
        :param expected_list: Expected value list
        :param metric_name: Metric name
        :param tolerance: Tolerance (absolute or relative)
        :param is_relative: Whether to use relative error
        """
        if len(actual_list) != len(expected_list):
            logger.warning(f"Warning: {metric_name} length mismatch! Actual: {len(actual_list)}, Expected: {len(expected_list)}")

        is_close = []
        total_steps_evaluated = min(len(actual_list), len(expected_list))
        for index in range(total_steps_evaluated):
            actual_value = float(actual_list[index])
            expected_value = float(expected_list[index])
            if is_relative:
                if expected_value == 0:
                    relative_error = 0 if actual_value == 0 else float('inf')
                else:
                    relative_error = abs(actual_value - expected_value) / abs(expected_value)
                if relative_error <= tolerance:
                    logger.info(f"Group {index + 1} {metric_name} performance comparison: Actual: {actual_value} vs Expected: {expected_value} Relative Error: {relative_error*100:.2f}% (Tolerance: {tolerance*100:.0f}%), Test Passed!")
                    is_close.append(True)
                else:
                    logger.warning(f"Group {index + 1} {metric_name} performance comparison: Actual: {actual_value} vs Expected: {expected_value} Relative Error: {relative_error*100:.2f}% Exceeds tolerance {tolerance*100:.0f}%, Test Failed!!!")
                    is_close.append(False)
            else:
                difference = abs(actual_value - expected_value)
                if difference <= float(tolerance):
                    logger.info(f"Group {index + 1} {metric_name} comparison: Actual: {actual_value} vs Expected: {expected_value} Difference within expected {tolerance}, Test Passed!")
                    is_close.append(True)
                else:
                    logger.warning(f"Group {index + 1} {metric_name} comparison: Actual: {actual_value} vs Expected: {expected_value} Difference exceeds {tolerance}, Actual difference is: {difference}, Test Failed!!!")
                    is_close.append(False)

        num_failing_steps_allowed = min(max(total_steps_evaluated // 100, 1), 50)
        passing = np.sum(is_close) >= (total_steps_evaluated - num_failing_steps_allowed)
        if not passing:
            message = f"{metric_name} comparison failed: Allowed failing steps {num_failing_steps_allowed}, Actual passed steps {np.sum(is_close)}, Total steps {total_steps_evaluated}"
            if raise_on_fail:
                raise ValueError(message)
            logger.warning(message)
            return False
        logger.info(f"{metric_name} comparison passed: Allowed failing steps {num_failing_steps_allowed}, Actual passed steps {np.sum(is_close)}, Total steps {total_steps_evaluated}")
        return True
    
    def _get_hf_layer_state_dict(self, load_path: str, layer_prefix: str, layer_id: int):
        """
        Get state_dict for specified layer
        
        Args:
            load_path: hf checkpoint path
            layer_prefix: layer prefix, e.g. "model.layers."
            layer_id: layer id
            
        Returns:
            checked_keys: list of checked keys
            state_dict: corresponding layer's state_dict
        """
        from safetensors.torch import load_file
        
        meta_file_name = f"{load_path}/model.safetensors.index.json"
        if not os.path.exists(meta_file_name):
            # Single safetensors file
            filename = f"{load_path}/model.safetensors"
            state_dict = {}
            checked_keys = []
            load_state_dict = load_file(filename, device="cpu")
            for key, value in load_state_dict.items():
                if (f"{layer_prefix}{layer_id}." in key) or (layer_id == 0 and layer_prefix not in key):
                    checked_keys.append(key)
                    state_dict[key] = value
            return checked_keys, state_dict

        # Multiple safetensors files
        checked_keys = []
        need_files = []
        with open(meta_file_name, 'r') as f:
            file_content = json.load(f)
            for key, value in file_content["weight_map"].items():
                if (f"{layer_prefix}{layer_id}." in key) or (layer_id == 0 and layer_prefix not in key):
                    if key in checked_keys:
                        raise ValueError(f"duplicate key: {key}")
                    checked_keys.append(key)
                    if f"{load_path}/{value}" not in need_files:
                        need_files.append(f"{load_path}/{value}")
        
        state_dict = {}
        for filename in need_files:
            current_chunk = load_file(filename, device="cpu")
            state_dict.update(current_chunk)
        return checked_keys, state_dict

    def _check_hf_layer(self, src_load_path: str, dst_load_path: str, layer_prefix: str, layer_id: int, module_name: str):
        """
        Check consistency of single layer hf checkpoint
        
        Args:
            src_load_path: Source hf checkpoint path
            dst_load_path: Converted hf checkpoint path
            layer_prefix: Layer prefix
            layer_id: Layer id
            module_name: Module name (llm/vit)
        """
        logger.info(f"Checking {module_name} layer {layer_id} ...")
        
        src_checked_keys, src_state_dict = self._get_hf_layer_state_dict(src_load_path, layer_prefix, layer_id)
        dst_checked_keys, dst_state_dict = self._get_hf_layer_state_dict(dst_load_path, layer_prefix, layer_id)
        
        # Check if keys match
        if set(src_checked_keys) != set(dst_checked_keys):
            src_set = set(src_checked_keys)
            dst_set = set(dst_checked_keys)
            diff_src = list(src_set - dst_set)
            diff_dst = list(dst_set - src_set)
            raise ValueError(f"key mismatch.\nPresent in source but not in destination: {diff_src}\nPresent in destination but not in source: {diff_dst}")

        # Check if tensors corresponding to each key match
        for key in src_checked_keys:
            src_data = src_state_dict[key]
            dst_data = dst_state_dict[key]
            
            if src_data.shape != dst_data.shape:
                raise ValueError(f"{key} shape mismatch: src={src_data.shape}, dst={dst_data.shape}")
            
            if not torch.equal(src_data, dst_data):
                diff = (src_data.float() - dst_data.float()).abs().max()
                logger.warning(f"{key} value difference, max_diff={diff}")
                # If difference is too large, raise exception
                if diff > 1e-5:
                    raise ValueError(f"{key} value difference too large: max_diff={diff}")
        
        logger.info(f"Finished checking {module_name} layer {layer_id}")

    def check_hf_checkpoint(self, src_load_path: str, dst_load_path: str, model_name: str):
        """
        Check consistency between converted hf checkpoint and source hf checkpoint
        
        Args:
            src_load_path: Source hf checkpoint path
            dst_load_path: Converted hf checkpoint path
            model_name: Model name, used to get layer mapping configuration
        """
        logger.info(f"Start checking HF checkpoint consistency...")
        logger.info(f"Source path: {src_load_path}")
        logger.info(f"Converted path: {dst_load_path}")
        
        if model_name not in HF_CHECK_MODEL_MAPPING:
            raise ValueError(f"Unsupported model: {model_name}, please add configuration in HF_CHECK_MODEL_MAPPING")
        
        mapping = HF_CHECK_MODEL_MAPPING[model_name]
        
        # Check LLM layers
        llm_prefix, llm_num_layers = mapping["llm"]
        for layer_id in range(llm_num_layers):
            self._check_hf_layer(src_load_path, dst_load_path, llm_prefix, layer_id, "llm")
        
        # Check ViT layers
        vit_prefix, vit_num_layers = mapping["vit"]
        for layer_id in range(vit_num_layers):
            self._check_hf_layer(src_load_path, dst_load_path, vit_prefix, layer_id, "vit")
        
        logger.info(f"HF checkpoint consistency check passed!")

    def create_shell_file(self, model_config, script_path, new_script_path):
        import re
        env_vars = {}
        for var, value in model_config.items():
            env_vars[var] = str(value)

        with open(script_path, "r") as file:
            script = file.read()
        
        while True:
            new_script = script
            for var, value in env_vars.items():
                new_script = re.sub(f"\\$\\{{{var}\\}}", value, new_script)
            if new_script == script:  # If no replacement happened, break the loop
                break
            script = new_script

        # Ensure directory exists before creating file
        os.makedirs(os.path.dirname(new_script_path), exist_ok=True)
        with open(new_script_path, "w") as file:
            file.write(script)
    
    def start_loongforge_convert_ckpt(self, index, step_stage, scenario_name, training_type_name):
        """
        Start executing ckpt conversion task
        Args:
            index: Scenario index
            step_stage: Step stage name (e.g. Step1, Step2)
            scenario_name: Scenario name
            training_type_name: Training type name (e.g. pretrain, sft)
        """
        step_name = "loongforge_convert_ckpt"
        logger.info(f"{step_stage} {step_name} Start Running ...")

        model_config = self.__init_model_scenarios_data__(index, scenario_name, step_stage, training_type_name)

        # ckpt weight conversion
        model_name = self.model_name
        node_nums = self.input_cmd_args.node_nums
        timeout = self.input_cmd_args.timeout
        scripts_root_path = model_config["scripts_root_path"]
        model_lock_file_path = model_config["model_lock_file_path"]
        training_log_path = model_config["training_log_path"]

        # Convert config file to env variables passed to running script
        env_vars_str = self.__convert_model_config_to_env__(model_config)
        step_stage_path = f'{model_lock_file_path}/{step_stage}/{self.master_addr}'
        model_lock_file = f'{step_stage_path}/{self.rank_name}_lock.txt'
        script_path = f"{scripts_root_path}/executor/{step_name}/run.sh"
        new_script_path = f"{training_log_path}/convert_ckpt_{model_name}_{self.rank_name}_run.sh"
        start_command = f"{env_vars_str} bash {script_path}"
        self.create_shell_file(model_config, script_path, new_script_path)

        # Open a new file to write script output
        training_log_file = f"{training_log_path}/convert_ckpt_{model_name}_{self.rank_name}_run.log"
        start_command = f"{env_vars_str} bash -c \"set -o pipefail; bash {scripts_root_path}/executor/{step_name}/run.sh |tee {training_log_file}\""
        logger.info(f"{step_stage} {step_name} Start: {start_command} .")
        if os.system(start_command) != 0:
            raise RuntimeError(f"Start {step_stage} {step_name} error, cmd is {start_command}")

        # Wait for all pods to complete
        self.wait_async_pod_complete(model_lock_file, model_name, f"{scenario_name}_{step_name}")

        logger.info(f"{step_stage} End {step_name}")

    def start_loongforge_reverse_convert_ckpt(self, index, step_stage, scenario_name, training_type_name):
        """
        Execute mcore -> hf reverse conversion and verify consistency between converted hf checkpoint and source hf checkpoint
        """
        step_name = "loongforge_convert_ckpt"
        logger.info(f"{step_stage} reverse_{step_name} Start Running ...")
        model_config = self.__init_model_scenarios_data__(index, scenario_name, step_stage, training_type_name)

        # Get configuration
        model_name = self.model_name
        scripts_root_path = model_config["scripts_root_path"]
        model_lock_file_path = model_config["model_lock_file_path"]
        training_log_path = model_config["training_log_path"]

        # Convert config file to env variables passed to running script
        env_vars_str = self.__convert_model_config_to_env__(model_config)
        step_stage_path = f'{model_lock_file_path}/{step_stage}/{self.master_addr}'
        model_lock_file = f'{step_stage_path}/{self.rank_name}_lock.txt'
        script_path = f"{scripts_root_path}/executor/{step_name}/reverse_run.sh"
        new_script_path = f"{training_log_path}/reverse_convert_ckpt_{model_name}_{self.rank_name}_run.sh"
        start_command = f"{env_vars_str} bash {script_path}"
        self.create_shell_file(model_config, script_path, new_script_path)

        # Open a new file to write script output
        training_log_file = f"{training_log_path}/reverse_convert_ckpt_{model_name}_{self.rank_name}_run.log"
        start_command = f"{env_vars_str} bash -c \"set -o pipefail; bash {scripts_root_path}/executor/{step_name}/reverse_run.sh |tee {training_log_file}\""
        logger.info(f"{step_stage} reverse_{step_name} Start: {start_command} .")
        if os.system(start_command) != 0:
            raise RuntimeError(f"Start {step_stage} reverse_{step_name} error, cmd is {start_command}")

        # Get hf check related config and check
        hf_ckpt_path = model_config.get("HF_CKPT_PATH", "")
        reverse_hf_ckpt_path = model_config.get("REVERSE_HF_CKPT_PATH", "")
        hf_check_model_name = model_config.get("HF_CHECK_MODEL_NAME", "")
        if hf_ckpt_path and reverse_hf_ckpt_path and hf_check_model_name:
            if self.is_final_pod:
                logger.info(f"Start executing HF checkpoint consistency check...")
                self.check_hf_checkpoint(hf_ckpt_path, reverse_hf_ckpt_path, hf_check_model_name)
        else:
            logger.warning(f"Missing HF check config, skipping consistency check: HF_CKPT_PATH={hf_ckpt_path}, REVERSE_HF_CKPT_PATH={reverse_hf_ckpt_path}, HF_CHECK_MODEL_NAME={hf_check_model_name}")

        # Wait for all pods to complete
        self.wait_async_pod_complete(model_lock_file, model_name, f"{scenario_name}_reverse_{step_name}")

        logger.info(f"{step_stage} End reverse_{step_name}")

    def start_loongforge(self, index, step_stage, scenario_name, training_type_name):
        """
        Start executing training task
        Args:
            index: Scenario index
            step_stage: Step stage name (e.g. Step1, Step2)
            scenario_name: Scenario name
            training_type_name: Training type name (e.g. pretrain, sft)
        """
        step_name = "loongforge"
        logger.info(f"{step_stage} {step_name} Start Running ...")

        model_config = self.__init_model_scenarios_data__(index, scenario_name, step_stage, training_type_name)

        # ckpt weight conversion
        model_name = self.model_name
        node_nums = self.input_cmd_args.node_nums
        timeout = self.input_cmd_args.timeout
        scripts_root_path = model_config["scripts_root_path"]
        model_lock_file_path = model_config["model_lock_file_path"]
        training_log_path = model_config["training_log_path"]

        # Convert config file to env variables passed to running script
        env_vars_str = self.__convert_model_config_to_env__(model_config)

        step_stage_path = f'{model_lock_file_path}/{step_stage}/{self.master_addr}'
        model_lock_file = f'{step_stage_path}/{self.rank_name}_lock.txt'

        script_path = f"{scripts_root_path}/executor/{step_name}/run.sh"
        new_script_path = f"{training_log_path}/training_{model_name}_{training_type_name}_{self.rank_name}_run.sh"
        start_command = f"{env_vars_str} bash {script_path}"
        self.create_shell_file(model_config, script_path, new_script_path)

        # Open a new file to write script output
        training_log_file = f"{training_log_path}/training#{model_name}#{training_type_name}#nodes_{self.input_cmd_args.node_nums}#{self.rank_name}#run.log"

        start_command = f"{env_vars_str} bash -c \"set -o pipefail; bash {scripts_root_path}/executor/{step_name}/run.sh |tee {training_log_file}\""
        logger.info(f"{step_stage} {step_name} Start: {start_command} .")
        if os.system(start_command) != 0:
           raise RuntimeError(f"Start {step_stage} {step_name} error, cmd is {start_command}")
        
        # Wait for all pods to complete and assert this training result
        if self.task_type == "function":
            self.wait_async_pod_complete(
                model_lock_file,
                model_name,
                f"{scenario_name}_{step_name}",
                is_function=True,
                function=self.assert_loongforge,
                raise_on_error=True,
                training_log_file=training_log_file,
                training_type=training_type_name
            )
        else:
            self.wait_async_pod_complete(
            model_lock_file,
            model_name,
            f"{scenario_name}_{step_name}",
        )

        logger.info(f"{step_stage} End {step_name}")

    def __call__(self) -> TaskResut:
        raise NotImplementedError(f"must overide this method.")