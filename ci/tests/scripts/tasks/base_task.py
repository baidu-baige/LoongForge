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
from typing import Dict, List, Any
from tools.color_logger import create_color_logger
from utils.constants import task_created_flag, task_running_flag, task_finish_flag
from metric.metric import Metric

logger = create_color_logger(name=__name__)


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
        self.ckpt_loss_diff = args.ckpt_loss_diff

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
                if key == "paramters" or "_ARGS" in key or "loss_in_value" in key:
                    value = json.dumps(value)
            else:
                value = json.dumps(value)
                value = f"\"{value}\""
            
            env_vars_str += f"{key}={value} "
        return env_vars_str
    

    
    def assert_aiak_training_omni(self, expected_loss, training_log_file):
        logger.info(f"Start assert_aiak_training_omni ...")

        if not self.is_final_pod:
            return
        
        # 收集训练过程中metric 指标
        self.collect_metrics(training_log_file)
        
        # 校验点1, loss value 对比预存值
        self.validate_loss_value(expected_loss)

        logger.info(f"End assert_aiak_training_omni")
    
    def collect_metrics(self, training_log_file):
        self.metric.model_name = self.model_name
        # 收集训练过程中metric 指标
        with open(training_log_file, 'r') as file:
            lines = file.readlines()

        # 清空，防止场景较多后混入
        self.metric.lm_loss_list = []

        # 添加 loss value
        for i in range(len(lines)):
            line = lines[i]
            loss_match = re.search(r'lm loss: ([\d\.E\+-]+)', line)
            if loss_match:
                training_lm_loss_str = str(loss_match.group(1)).strip()
                self.metric.lm_loss_list.append(training_lm_loss_str)
            
            # 获取'throughput (token/sec/GPU)'的值
            throughput_match = re.search(r'throughput \(token/sec/GPU\): ([\d\.E\+-]+)', line)
            if throughput_match:
                throughput_str = str(throughput_match.group(1)).strip()
                self.metric.throughput.append(throughput_str)
            
            elapsed_time_match = re.search(r'elapsed time per iteration \(ms\): ([\d\.E\+-]+)', line)
            if elapsed_time_match:
                elapsed_time_str = str(elapsed_time_match.group(1)).strip()
                self.metric.elapsed_time_match.append(elapsed_time_str)
            
            batch_size_match = re.search(r'global batch size:\s*(\d+)', line)
            if batch_size_match:
                batch_size_str = str(batch_size_match.group(1)).strip()
                self.metric.global_batch_size.append(batch_size_str)
        
        assert(len(self.metric.lm_loss_list) != 0, "此次任务的loss 为空，需要查看训练任务是否正常！！！")
        
        # logger.info(f"self.metric dict: {self.metric.obj_to_dict()}")
    
    def validate_loss_value(self, expected_loss):
        import re

        # 预存的loss值
        expected_loss_in_value_list = [x for x in re.findall(r'[\d\.E\+-]+', expected_loss)]

        # 判断loss值是否符合预期
        for index in range(len(self.metric.lm_loss_list)):
            training_lm_loss_str = self.metric.lm_loss_list[index]
            expected_loss_in_value_str = str(expected_loss_in_value_list[index]).strip()
            
            training_lm_loss = float(training_lm_loss_str)
            expected_loss_in_value = float(expected_loss_in_value_str)

            difference = abs(training_lm_loss - expected_loss_in_value)

            if difference <= float(self.ckpt_loss_diff):
                logger.info(f"第 {index + 1} 组 loss 对比: 实际值: {training_lm_loss}({training_lm_loss_str}) vs 预期值: {expected_loss_in_value}({expected_loss_in_value_str}) 差异值预期在 {str( self.ckpt_loss_diff)} 以内, 测试通过!")
            else:
                raise ValueError(f'第 {index + 1} 组 loss 对比: 实际值: {training_lm_loss}({training_lm_loss_str}) vs 预期值: {expected_loss_in_value}({expected_loss_in_value_str}) 差异值预期在 {str( self.ckpt_loss_diff)} 以内, 实际差值是: {difference},测试不通过!!!')


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
                new_script = re.sub(f"\\$\{{{var}\}}", value, new_script)
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
                expected_loss=model_config["loss_in_value"],
                training_log_file=training_log_file
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