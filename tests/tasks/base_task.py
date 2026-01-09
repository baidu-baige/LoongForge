#!/usr/bin/env python3
"""
base_task.py
"""
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################

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

# HF checkpoint 检查的模型层映射配置
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

    # 更新锁文件
    def update_lock_file(self, lock_file_path, task_flag):
        with open(lock_file_path, "a") as file:
            logger.info(f"update_lock_file {task_flag} start ...")
            file.seek(0) # 重置文件指针到开头
            file.truncate() # 清空文件内容
            file.write(task_flag)
            logger.info(f"update_lock_file {task_flag} end")

    # 检查锁文件
    def check_lock_file(self, lock_file_path, task_flag):
        # 遍历 lock_file_path 文件夹下所有文件的内容总数是否等于预期，相等True，否则返回 False，再判断内容都是 task_flag，是则返回 True，否则返回 False，
        count = 0
        flag = True

        parent_path = os.path.dirname(lock_file_path)
        for filename in os.listdir(parent_path):
            with open(os.path.join(parent_path, filename), "r") as f:
                content = f.read().strip()
                if content == task_flag:
                    count += 1
                else:
                    flag = False

        return count, (count == self.input_cmd_args.node_nums and flag)
    
    # 等待所有的 Pod/阶段 都完成
    def wait_async_task_complete(self, lock_file_path, task_flag, model_name="", scenarios_name=""):
        # 检查锁文件，如果所有的 Pod 都已经完成，就进入下一阶段
        for _ in range(self.input_cmd_args.timeout // 10):
            if self.check_lock_file(lock_file_path, task_flag):
                logger.info(f"模型 【{model_name}】{scenarios_name} 所有的 Pod 都已经完成")
                # if self.is_final_pod:
                #     parent_path = os.path.dirname(os.path.dirname(lock_file_path))
                #     shutil.rmtree(parent_path)
                break
            else:
                logger.info(f"模型 【{model_name}】{scenarios_name} 还有 Pod 没有完成，等待...")
                time.sleep(10)
        else:
            raise TimeoutError(f"ERROR: 等待其他 Pod 已超过{self.input_cmd_args.timeout} 秒")

    # lock_file 锁文件存在，并且内容 == MASTER_ADDR，证明此次任务已经写入文件完成，可以继续执行
    # 否则，不存在就创建，并写入内容 MASTER_ADDR
    def initialize_lock_file(self, lock_file, task_flag):
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(os.path.dirname(lock_file)):
            os.makedirs(os.path.dirname(lock_file))

        # 打开或创建锁文件
        with open(lock_file, 'a+') as file:
            logger.info(f"initialize_lock_file start ...")
            file.seek(0) # 重置文件指针到开头
            file.truncate() # 清空文件内容
            file.write(task_flag)
            logger.info(f"initialize_lock_file end")
    
    def wait_async_pod_complete(self, lock_file, model_name="", scenarios_name="", is_function=False, function=None, raise_on_error=False, *args, **kwargs):
        logger.info(f"模型 【{model_name}】{scenarios_name} 等待其他 Pod 完成 ...")
        task_flag = task_finish_flag

        try:
            # 检查目录是否存在，如果不存在则创建
            if not os.path.exists(os.path.dirname(lock_file)):
                os.makedirs(os.path.dirname(lock_file))
        except Exception as e:
            logger.warning(f"warning: 创建目录失败，{e}")
            pass
        
        # 第一步：将对应的pod的数据写入到 lock_file 文件
        # 打开或创建锁文件
        with open(lock_file, 'a+') as file:
            file.seek(0) # 重置文件指针到开头
            file.truncate() # 清空文件内容
            file.write(task_flag)

        # 第二步：检查 lock_file 文件，如果所有的 Pod 都已经完成，就进入下一阶段：1、所有文件状态都是Finish。2、文件总数是 node_nums。
        for _ in range(self.input_cmd_args.timeout // 10):
            finish_count, state = self.check_lock_file(lock_file, task_flag)
            if state:
                logger.info(f"模型 【{model_name}】{scenarios_name} 所有的 Pod 都已经完成, {finish_count} / {self.input_cmd_args.node_nums}")

                if is_function and self.is_final_pod:
                    if function and callable(function):
                        try:
                            logger.info(f"执行 {function.__name__} 方法:")
                            result = function(*args, **kwargs)  # 调用函数并传入参数
                            logger.info(f"函数 {function.__name__} 执行完成，返回值: {result}")
                        except Exception as e:
                            logger.error(f"执行函数 {function.__name__} 时出错: {e}")
                            # 如果 raise_on_error 参数为 True，则抛出异常
                            if raise_on_error:
                                raise Exception(f"执行函数 {function.__name__} 时出错: {e}")
                    else:
                        logger.error(f"提供的 {function} 不是可调用的函数。")
                break
            else:
                logger.info(f"模型 【{model_name}】{scenarios_name} 还有 Pod 没有完成，{finish_count} / {self.input_cmd_args.node_nums} 等待...")
                time.sleep(10)
        else:
            raise TimeoutError(f"ERROR: 等待其他 Pod 已超过{self.input_cmd_args.timeout} 秒")
    

    def __init_ckpt_file__(self) -> None:
        # 初始化数据场景，首次进入需要删除一些文件，避免影响后续测试
        scenario_name = ".init"
        init_lock_path = os.path.join(self.model["model_lock_file_path"], scenario_name, self.master_addr)
        init_lock_file = os.path.join(init_lock_path, f"{self.rank_name}_lock.txt")

        # 等待所有pod就绪后，删除模型已经存在的 ckpt 文件
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
        # 删除 step1_output_path 的输出文件
        if os.path.exists(step1_output_path):
            shutil.rmtree(step1_output_path)
    
    def __clean_up__(self):
        # 初始化数据场景，首次进入需要删除一些文件，避免影响后续测试
        scenario_name = ".clean_up"
        init_lock_path = os.path.join(self.model["model_lock_file_path"], scenario_name, self.master_addr)
        init_lock_file = os.path.join(init_lock_path, f"{self.rank_name}_lock.txt")

        # 等待所有pod就绪后，删除模型已经存在的 ckpt 文件
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
            # 合并 function_step_data 和 model
            model = {**model, **function_step_data}

            # 正常获取scenarios_name场景下的数据
            step_data = self.model["scenarios"][index][scenarios_name][training_type_name][step_name]
            # 合并 function_step_data 和 model 为一个json
            model = {**model, **step_data}
        
        elif self.task_type == "function":
            # 迭代获取之前的变量以及替换最新的变量
            for i, scenario in enumerate(self.model["scenarios"]):
                local_index = i + 1
                local_step_name = f"Step{local_index}"

                function_scenarios_name = "function"
                # 检查步骤是否存在，如果不存在则跳过
                if (local_step_name not in self.model["scenarios"][0][function_scenarios_name][training_type_name]):
                    logger.info(f"local_step_name {local_step_name} 当前不存在，跳过 ...")
                    continue
  
                function_step_data = self.model["scenarios"][0][function_scenarios_name][training_type_name][local_step_name]
                # 合并 function_step_data 和 model
                model = {**model, **function_step_data}

                # 更新变量到当前的步骤
                if local_step_name == step_name:
                    break
            
            # 正常获取scenarios_name场景下的数据
            step_data = self.model["scenarios"][index][scenarios_name][training_type_name][step_name]
            # 合并 function_step_data 和 model 为一个json
            model = {**model, **step_data}
        
        elif self.task_type == "preprocess_data":
            # 正常获取scenarios_name场景下的数据
            step_data = self.model["scenarios"][index][scenarios_name][step_name]
            # 合并 function_step_data 和 model 为一个json
            model = {**model, **step_data}

        # 设置是否使用nccl
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
        # 初始化一个空的环境变量字符串
        env_vars_str = ""
        new_model_config = deepcopy(model_config)

        # 遍历字典，将键值对转换为环境变量
        for key, value in new_model_config.items():
            if key == "scenarios":
                continue

            if isinstance(value, str):
                # 对于 _ARGS 类型的参数，需要特殊处理：
                # 1. 去除注释行（以 # 开头的行)
                # 2. 将换行符转换为空格，多个空格合并为单个空格
                # 3. 去除首尾空白
                # 4. 不使用 json.dumps，直接传递原始字符串（bash 会按空格分割)
                if "_ARGS" in key:
                    # 去除注释行和空行
                    lines = []
                    for line in value.split('\n'):
                        stripped_line = line.strip()
                        # 跳过空行和注释行（包括行内注释，如 "--arg # comment")
                        if stripped_line and not stripped_line.startswith('#'):
                            # 处理行内注释：移除 # 及其后面的内容
                            if '#' in stripped_line:
                                # 检查 # 是否在引号内
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
                    # 用空格连接所有非注释行，并规范化空格
                    value = ' '.join(lines)
                    # 将多个连续空格替换为单个空格
                    import re
                    value = re.sub(r'\s+', ' ', value).strip()
                    # 对于 _ARGS，需要用引号包裹整个值，以确保包含空格的值被正确传递
                    # 转义值中的引号，然后用引号包裹
                    value = value.replace('"', '\\"')
                    value = f'"{value}"'
                elif key == "paramters" or "loss_in_value" in key:
                    value = json.dumps(value)
            else:
                value = json.dumps(value)
                value = f"\"{value}\""
            
            env_vars_str += f"{key}={value} "
        return env_vars_str
    

    
    def assert_aiak_training_omni(self, training_log_file, training_type=None, **kwargs):
        logger.info(f"Start assert_aiak_training_omni ...")

        if not self.is_final_pod:
            return
        # 收集训练过程中metric 指标
        self.collect_metrics(training_log_file)
        # 统一校验精度和性能指标
        self.validate_metrics(self.model_name, training_type)
        logger.info(f"End assert_aiak_training_omni")

    def validate_metrics(self, model_name, training_type=None):
        """
        统一验证训练的精度和性能指标。
        精度指标(lm_loss, grad_norm)使用relative_tolerance=0.02，性能指标(elapsed_time_ms, throughput)使用relative_tolerance=0.05。
        Args:
            model_name: 模型名称，用于定位 baseline JSON 文件
            training_type: 训练类型（如 'pretrain', 'sft')
        """
        baseline_data = ConfigManager.get_baseline_data(None, self.model, model_name, training_type)
        # 精度指标
        accuracy_relative_tolerance = self.accuracy_relative_tolerance
        # lm_loss
        if hasattr(self.metric, 'lm_loss_list') and self.metric.lm_loss_list and 'lm_loss' in baseline_data[0]:
            expected_loss_list = [item['lm_loss'] for item in baseline_data]
            self._compare_metric(
                actual_list=self.metric.lm_loss_list,
                expected_list=expected_loss_list,
                metric_name="lm_loss",
                tolerance=accuracy_relative_tolerance,
                is_relative=False
            )
        # grad_norm
        if hasattr(self.metric, 'grad_norm_list') and self.metric.grad_norm_list and 'grad_norm' in baseline_data[0]:
            expected_grad_norm_list = [item['grad_norm'] for item in baseline_data]
            self._compare_metric(
                actual_list=self.metric.grad_norm_list,
                expected_list=expected_grad_norm_list,
                metric_name="grad_norm",
                tolerance=accuracy_relative_tolerance,
                is_relative=False
            )
        logger.info("精度指标验证通过!")

        # 性能指标
        performance_relative_tolerance = self.performance_relative_tolerance
        num_iters = min(len(self.metric.elapsed_time_match), len(baseline_data))
        if num_iters == 0:
            logger.warning("没有足够的数据进行性能验证")
            return
        # elapsed_time_ms
        if hasattr(self.metric, 'elapsed_time_match') and self.metric.elapsed_time_match and 'elapsed_time_ms' in baseline_data[0]:
            expected_elapsed_time = [item['elapsed_time_ms'] for item in baseline_data[:num_iters]]
            actual_elapsed_time = [float(x) for x in self.metric.elapsed_time_match[:num_iters]]
            self._compare_metric(
                actual_list=actual_elapsed_time,
                expected_list=expected_elapsed_time,
                metric_name="elapsed_time_ms",
                tolerance=performance_relative_tolerance,
                is_relative=True
            )
        # throughput
        if hasattr(self.metric, 'throughput') and self.metric.throughput and 'throughput' in baseline_data[0]:
            expected_throughput = [item['throughput'] for item in baseline_data[:num_iters]]
            actual_throughput = [float(x) for x in self.metric.throughput[:num_iters]]
            self._compare_metric(
                actual_list=actual_throughput,
                expected_list=expected_throughput,
                metric_name="throughput",
                tolerance=performance_relative_tolerance,
                is_relative=True
            )

        # 显存指标
        # mem_allocated_avg_MB
        if hasattr(self.metric, 'mem_allocated_avg_MB') and self.metric.mem_allocated_avg_MB and 'mem_allocated_avg_MB' in baseline_data[0]:
            expected_mem_allocated = [item['mem_allocated_avg_MB'] for item in baseline_data[:num_iters]]
            actual_mem_allocated = [float(x) for x in self.metric.mem_allocated_avg_MB[:num_iters]]
            self._compare_metric(
                actual_list=actual_mem_allocated,
                expected_list=expected_mem_allocated,
                metric_name="mem_allocated_avg_MB",
                tolerance=performance_relative_tolerance,
                is_relative=True
            )
        # mem_max_allocated_avg_MB
        if hasattr(self.metric, 'mem_max_allocated_avg_MB') and self.metric.mem_max_allocated_avg_MB and 'mem_max_allocated_avg_MB' in baseline_data[0]:
            expected_mem_max_allocated = [item['mem_max_allocated_avg_MB'] for item in baseline_data[:num_iters]]
            actual_mem_max_allocated = [float(x) for x in self.metric.mem_max_allocated_avg_MB[:num_iters]]
            self._compare_metric(
                actual_list=actual_mem_max_allocated,
                expected_list=expected_mem_max_allocated,
                metric_name="mem_max_allocated_avg_MB",
                tolerance=performance_relative_tolerance,
                is_relative=True
            )
        logger.info("性能指标验证通过!")
    
    def collect_metrics(self, training_log_file):
        self.metric.model_name = self.model_name
        # 收集训练过程中metric 指标
        with open(training_log_file, 'r') as file:
            lines = file.readlines()

        # 清空，防止场景较多后混入
        self.metric.lm_loss_list = []
        self.metric.grad_norm_list = []
        self.metric.throughput = []
        self.metric.elapsed_time_match = []
        self.metric.mem_allocated_avg_MB = []
        self.metric.mem_max_allocated_avg_MB = []

        # 添加 loss value
        for i in range(len(lines)):
            line = lines[i]
            loss_match = re.search(r'lm loss: ([\d\.E\+-]+)', line)
            if loss_match:
                training_lm_loss_str = str(loss_match.group(1)).strip()
                self.metric.lm_loss_list.append(training_lm_loss_str)
            # 获取 grad_norm 的值
            grad_norm_match = re.search(r'grad norm: ([\d\.E\+-]+)', line)
            if grad_norm_match:
                grad_norm_str = str(grad_norm_match.group(1)).strip()
                self.metric.grad_norm_list.append(grad_norm_str)
            # 获取'throughput (token/sec/GPU)'的值
            throughput_match = re.search(r'throughput \(token/sec/GPU\): ([\d\.E\+-]+)', line)
            if throughput_match:
                throughput_str = str(throughput_match.group(1)).strip()
                self.metric.throughput.append(throughput_str)
            # 获取 elapsed time per iteration (ms) 的值
            elapsed_time_match = re.search(r'elapsed time per iteration \(ms\): ([\d\.E\+-]+)', line)
            if elapsed_time_match:
                elapsed_time_str = str(elapsed_time_match.group(1)).strip()
                self.metric.elapsed_time_match.append(elapsed_time_str)
            # 获取 mem_allocated_avg_MB（训练日志格式如: mem-allocated-bytes-avg(MB): 15896.39）
            mem_allocated_match = re.search(r'mem-allocated-bytes-avg\(MB\):\s*([\d\.E\+-]+)', line)
            if mem_allocated_match:
                mem_allocated_str = str(mem_allocated_match.group(1)).strip()
                self.metric.mem_allocated_avg_MB.append(mem_allocated_str)
            # 获取 mem_max_allocated_avg_MB（训练日志格式如: mem-max-allocated-bytes-avg(MB): 29039.47）
            mem_max_allocated_match = re.search(r'mem-max-allocated-bytes-avg\(MB\):\s*([\d\.E\+-]+)', line)
            if mem_max_allocated_match:
                mem_max_allocated_str = str(mem_max_allocated_match.group(1)).strip()
                self.metric.mem_max_allocated_avg_MB.append(mem_max_allocated_str)
            # 获取 global batch size
            batch_size_match = re.search(r'global batch size:\s*(\d+)', line)
            if batch_size_match:
                batch_size_str = str(batch_size_match.group(1)).strip()
                self.metric.global_batch_size.append(batch_size_str)

        assert len(self.metric.lm_loss_list) != 0, "此次任务的loss 为空，需要查看训练任务是否正常！！！"
        # logger.info(f"self.metric dict: {self.metric.obj_to_dict()}")
      
    def _compare_metric(self, actual_list, expected_list, metric_name, tolerance, is_relative=False):
        """
        通用指标对比函数，支持绝对误差和相对误差，允许部分iteration超出容忍范围
        :param actual_list: 实际值列表
        :param expected_list: 期望值列表
        :param metric_name: 指标名
        :param tolerance: 容忍度（绝对或相对）
        :param is_relative: 是否用相对误差
        """
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
                    logger.info(f"第 {index + 1} 组 {metric_name} 性能对比: 实际值: {actual_value} vs 预期值: {expected_value} 相对误差: {relative_error*100:.2f}% (容忍: {tolerance*100:.0f}%), 测试通过!")
                    is_close.append(True)
                else:
                    logger.warning(f'第 {index + 1} 组 {metric_name} 性能对比: 实际值: {actual_value} vs 预期值: {expected_value} 相对误差: {relative_error*100:.2f}% 超过容忍范围 {tolerance*100:.0f}%, 测试不通过!!!')
                    is_close.append(False)
            else:
                difference = abs(actual_value - expected_value)
                if difference <= float(tolerance):
                    logger.info(f"第 {index + 1} 组 {metric_name} 对比: 实际值: {actual_value} vs 预期值: {expected_value} 差异值预期在 {tolerance} 以内, 测试通过!")
                    is_close.append(True)
                else:
                    logger.warning(f'第 {index + 1} 组 {metric_name} 对比: 实际值: {actual_value} vs 预期值: {expected_value} 差异值超出 {tolerance}, 实际差值是: {difference}, 测试不通过!!!')
                    is_close.append(False)

        num_failing_steps_allowed = min(max(total_steps_evaluated // 100, 1), 50)
        passing = np.sum(is_close) >= (total_steps_evaluated - num_failing_steps_allowed)
        if not passing:
            raise ValueError(f"{metric_name} 对比未通过: 允许失败步数 {num_failing_steps_allowed}, 实际通过步数 {np.sum(is_close)}, 总步数 {total_steps_evaluated}")
        else:
            logger.info(f"{metric_name} 对比通过: 允许失败步数 {num_failing_steps_allowed}, 实际通过步数 {np.sum(is_close)}, 总步数 {total_steps_evaluated}")
    
    def _get_hf_layer_state_dict(self, load_path: str, layer_prefix: str, layer_id: int):
        """
        获取指定层的 state_dict
        
        Args:
            load_path: hf checkpoint 路径
            layer_prefix: 层前缀，如 "model.layers."
            layer_id: 层 id
            
        Returns:
            checked_keys: 检查的 key 列表
            state_dict: 对应层的 state_dict
        """
        from safetensors.torch import load_file
        
        meta_file_name = f"{load_path}/model.safetensors.index.json"
        if not os.path.exists(meta_file_name):
            # 单个 safetensors 文件
            filename = f"{load_path}/model.safetensors"
            state_dict = {}
            checked_keys = []
            load_state_dict = load_file(filename, device="cpu")
            for key, value in load_state_dict.items():
                if (f"{layer_prefix}{layer_id}." in key) or (layer_id == 0 and layer_prefix not in key):
                    checked_keys.append(key)
                    state_dict[key] = value
            return checked_keys, state_dict

        # 多个 safetensors 文件
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
        检查单层 hf checkpoint 的一致性
        
        Args:
            src_load_path: 原始 hf checkpoint 路径
            dst_load_path: 转换后的 hf checkpoint 路径
            layer_prefix: 层前缀
            layer_id: 层 id
            module_name: 模块名称（llm/vit）
        """
        logger.info(f"检查 {module_name} layer {layer_id} ...")
        
        src_checked_keys, src_state_dict = self._get_hf_layer_state_dict(src_load_path, layer_prefix, layer_id)
        dst_checked_keys, dst_state_dict = self._get_hf_layer_state_dict(dst_load_path, layer_prefix, layer_id)
        
        # 检查 key 是否一致
        if set(src_checked_keys) != set(dst_checked_keys):
            src_set = set(src_checked_keys)
            dst_set = set(dst_checked_keys)
            diff_src = list(src_set - dst_set)
            diff_dst = list(dst_set - src_set)
            raise ValueError(f"key 不一致.\n原始有但转换后没有: {diff_src}\n转换后有但原始没有: {diff_dst}")

        # 检查每个 key 对应的 tensor 是否一致
        for key in src_checked_keys:
            src_data = src_state_dict[key]
            dst_data = dst_state_dict[key]
            
            if src_data.shape != dst_data.shape:
                raise ValueError(f"{key} shape 不一致: src={src_data.shape}, dst={dst_data.shape}")
            
            if not torch.equal(src_data, dst_data):
                diff = (src_data.float() - dst_data.float()).abs().max()
                logger.warning(f"{key} 数值有差异, max_diff={diff}")
                # 如果差异过大，抛出异常
                if diff > 1e-5:
                    raise ValueError(f"{key} 数值差异过大: max_diff={diff}")
        
        logger.info(f"完成检查 {module_name} layer {layer_id}")

    def check_hf_checkpoint(self, src_load_path: str, dst_load_path: str, model_name: str):
        """
        检查转换后的 hf checkpoint 与原始 hf checkpoint 的一致性
        
        Args:
            src_load_path: 原始 hf checkpoint 路径
            dst_load_path: 转换后的 hf checkpoint 路径  
            model_name: 模型名称，用于获取层映射配置
        """
        logger.info(f"开始检查 HF checkpoint 一致性...")
        logger.info(f"原始路径: {src_load_path}")
        logger.info(f"转换后路径: {dst_load_path}")
        
        if model_name not in HF_CHECK_MODEL_MAPPING:
            raise ValueError(f"不支持的模型: {model_name}，请在 HF_CHECK_MODEL_MAPPING 中添加配置")
        
        mapping = HF_CHECK_MODEL_MAPPING[model_name]
        
        # 检查 LLM 层
        llm_prefix, llm_num_layers = mapping["llm"]
        for layer_id in range(llm_num_layers):
            self._check_hf_layer(src_load_path, dst_load_path, llm_prefix, layer_id, "llm")
        
        # 检查 ViT 层
        vit_prefix, vit_num_layers = mapping["vit"]
        for layer_id in range(vit_num_layers):
            self._check_hf_layer(src_load_path, dst_load_path, vit_prefix, layer_id, "vit")
        
        logger.info(f"HF checkpoint 一致性检查通过!")

    def create_shell_file(self, model_config, script_path, new_script_path):
        import re
        # 环境变量字典
        env_vars = {}
        # print(model_config)

        for var, value in model_config.items():
            env_vars[var] = str(value)

        # 读取脚本内容
        with open(script_path, "r") as file:
            script = file.read()
        
        # 不断替换变量，直到所有的变量都被替换掉
        while True:
            new_script = script
            for var, value in env_vars.items():
                new_script = re.sub(f"\\$\\{{{var}\\}}", value, new_script)
            if new_script == script:  # 如果没有发生任何替换，那么就结束循环
                break
            script = new_script

        # 保存新的脚本
        with open(new_script_path, "w") as file:
            file.write(script)
    
    def start_aiak_convert_ckpt(self, index, step_stage, scenario_name, training_type_name):
        step_name = "aiak_convert_ckpt"
        logger.info(f"{step_stage} {step_name} Start Running ...")

        model_config = self.__init_model_scenarios_data__(index, scenario_name, step_stage, training_type_name)

        # ckpt 权重转化
        model_name = self.model_name
        node_nums = self.input_cmd_args.node_nums
        timeout = self.input_cmd_args.timeout
        scripts_root_path = model_config["scripts_root_path"]
        model_lock_file_path = model_config["model_lock_file_path"]
        training_log_path = model_config["training_log_path"]

        # 将配置文件转成env 环境变量传递给运行脚本
        env_vars_str = self.__convert_model_config_to_env__(model_config)

        step_stage_path = f'{model_lock_file_path}/{step_stage}/{self.master_addr}'
        model_lock_file = f'{step_stage_path}/{self.rank_name}_lock.txt'

        script_path = f"{scripts_root_path}/executor/{step_name}/run.sh"
        new_script_path = f"{training_log_path}/convert_ckpt_{model_name}_{self.rank_name}_run.sh"
        start_command = f"{env_vars_str} bash {script_path}"
        self.create_shell_file(model_config, script_path, new_script_path)

        # 打开一个新的文件用来写入脚本的输出
        training_log_file = f"{training_log_path}/convert_ckpt_{model_name}_{self.rank_name}_run.log"

        start_command = f"{env_vars_str} bash -c \"set -o pipefail; bash {scripts_root_path}/executor/{step_name}/run.sh |tee {training_log_file}\""
        logger.info(f"{step_stage} {step_name} Start: {start_command} .")
        if os.system(start_command) != 0:
            raise RuntimeError(f"Start {step_stage} {step_name} error, cmd is {start_command}")

        # 等待所有pod 完成
        self.wait_async_pod_complete(model_lock_file, model_name, f"{scenario_name}_{step_name}")

        logger.info(f"{step_stage} End {step_name}")

    def start_aiak_reverse_convert_ckpt(self, index, step_stage, scenario_name, training_type_name):
        """
        执行 mcore -> hf 逆向转换并验证转换后的 hf checkpoint 与原始 hf checkpoint 的一致性
        """
        step_name = "aiak_convert_ckpt"
        logger.info(f"{step_stage} reverse_{step_name} Start Running ...")

        model_config = self.__init_model_scenarios_data__(index, scenario_name, step_stage, training_type_name)

        # 获取配置
        model_name = self.model_name
        scripts_root_path = model_config["scripts_root_path"]
        model_lock_file_path = model_config["model_lock_file_path"]
        training_log_path = model_config["training_log_path"]

        # 将配置文件转成env 环境变量传递给运行脚本
        env_vars_str = self.__convert_model_config_to_env__(model_config)

        step_stage_path = f'{model_lock_file_path}/{step_stage}/{self.master_addr}'
        model_lock_file = f'{step_stage_path}/{self.rank_name}_lock.txt'

        script_path = f"{scripts_root_path}/executor/{step_name}/reverse_run.sh"
        new_script_path = f"{training_log_path}/reverse_convert_ckpt_{model_name}_{self.rank_name}_run.sh"
        start_command = f"{env_vars_str} bash {script_path}"
        self.create_shell_file(model_config, script_path, new_script_path)

        # 打开一个新的文件用来写入脚本的输出
        training_log_file = f"{training_log_path}/reverse_convert_ckpt_{model_name}_{self.rank_name}_run.log"

        start_command = f"{env_vars_str} bash -c \"set -o pipefail; bash {scripts_root_path}/executor/{step_name}/reverse_run.sh |tee {training_log_file}\""
        logger.info(f"{step_stage} reverse_{step_name} Start: {start_command} .")
        if os.system(start_command) != 0:
            raise RuntimeError(f"Start {step_stage} reverse_{step_name} error, cmd is {start_command}")

        # 获取 hf check 相关配置并执行检查
        hf_ckpt_path = model_config.get("HF_CKPT_PATH", "")
        reverse_hf_ckpt_path = model_config.get("REVERSE_HF_CKPT_PATH", "")
        hf_check_model_name = model_config.get("HF_CHECK_MODEL_NAME", "")
        
        if hf_ckpt_path and reverse_hf_ckpt_path and hf_check_model_name:
            # 只在最后一个 pod 执行 check
            if self.is_final_pod:
                logger.info(f"开始执行 HF checkpoint 一致性检查...")
                self.check_hf_checkpoint(hf_ckpt_path, reverse_hf_ckpt_path, hf_check_model_name)
        else:
            logger.warning(f"缺少 HF check 配置，跳过一致性检查: HF_CKPT_PATH={hf_ckpt_path}, REVERSE_HF_CKPT_PATH={reverse_hf_ckpt_path}, HF_CHECK_MODEL_NAME={hf_check_model_name}")

        # 等待所有pod 完成
        self.wait_async_pod_complete(model_lock_file, model_name, f"{scenario_name}_reverse_{step_name}")

        logger.info(f"{step_stage} End reverse_{step_name}")

    def start_aiak_training_omni(self, index, step_stage, scenario_name, training_type_name):
        step_name = "aiak_training_omni"
        logger.info(f"{step_stage} {step_name} Start Running ...")

        model_config = self.__init_model_scenarios_data__(index, scenario_name, step_stage, training_type_name)

        # ckpt 权重转化
        model_name = self.model_name
        node_nums = self.input_cmd_args.node_nums
        timeout = self.input_cmd_args.timeout
        scripts_root_path = model_config["scripts_root_path"]
        model_lock_file_path = model_config["model_lock_file_path"]
        training_log_path = model_config["training_log_path"]

        # 将配置文件转成env 环境变量传递给运行脚本
        env_vars_str = self.__convert_model_config_to_env__(model_config)

        step_stage_path = f'{model_lock_file_path}/{step_stage}/{self.master_addr}'
        model_lock_file = f'{step_stage_path}/{self.rank_name}_lock.txt'

        script_path = f"{scripts_root_path}/executor/{step_name}/run.sh"
        new_script_path = f"{training_log_path}/training_{model_name}_{training_type_name}_{self.rank_name}_run.sh"
        start_command = f"{env_vars_str} bash {script_path}"
        self.create_shell_file(model_config, script_path, new_script_path)

        # 打开一个新的文件用来写入脚本的输出
        training_log_file = f"{training_log_path}/training#{model_name}#{training_type_name}#nodes_{self.input_cmd_args.node_nums}#{self.rank_name}#run.log"

        start_command = f"{env_vars_str} bash -c \"set -o pipefail; bash {scripts_root_path}/executor/{step_name}/run.sh |tee {training_log_file}\""
        logger.info(f"{step_stage} {step_name} Start: {start_command} .")
        if os.system(start_command) != 0:
           raise RuntimeError(f"Start {step_stage} {step_name} error, cmd is {start_command}")
        
        # 等待所有pod 完成并针对此次训练结果断言
        if self.task_type == "function":
            self.wait_async_pod_complete(
                model_lock_file,
                model_name,
                f"{scenario_name}_{step_name}",
                is_function=True,
                function=self.assert_aiak_training_omni,
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