#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################


from tools.arguments import parse_args
from tools.color_logger import create_color_logger
from tools.config_manager import ConfigManager
from tqdm import tqdm
import time
import sys


logger = create_color_logger(name=__name__)


def prepare_models_list(args):
    """准备最终要运行的模型列表"""
    # 收集所有配置目录
    extra_dirs = list(args.extra_configs_dirs) if args.extra_configs_dirs else []
    
    # 如果指定了 --include_optional，添加 optional_configs 目录
    if args.include_optional:
        optional_dir = "optional_configs"
        if optional_dir not in extra_dirs:
            extra_dirs.append(optional_dir)
    
    # 更新 args 中的 extra_configs_dirs
    args.extra_configs_dirs = extra_dirs
    
    # 如果指定了 --optional_subdir，加载指定子目录的所有模型
    if args.optional_subdir:
        # 确保 optional_configs 在 extra_dirs 中
        if "optional_configs" not in extra_dirs:
            extra_dirs.append("optional_configs")
            args.extra_configs_dirs = extra_dirs
        
        # 获取指定子目录下的所有模型
        subdir_models = ConfigManager.get_models_from_subdir(
            "optional_configs",
            args.optional_subdir
        )
        
        if subdir_models:
            # 如果 models 为空或是默认值，则使用子目录的模型
            if not args.models or args.models == ConfigManager.get_all_models(args.configs_dir):
                args.models = subdir_models
            else:
                # 否则将子目录模型追加到现有模型列表
                args.models = list(args.models) + subdir_models
        
        logger.info(f"Models from optional_subdir '{args.optional_subdir}': {subdir_models}")
    
    # 如果指定了 --extra_models，将额外的模型追加到列表中
    if args.extra_models:
        # 确保 optional_configs 在 extra_dirs 中（因为 extra_models 通常来自 optional_configs）
        if "optional_configs" not in args.extra_configs_dirs:
            args.extra_configs_dirs = list(args.extra_configs_dirs) + ["optional_configs"]
        
        # 将 extra_models 追加到 models 列表
        args.models = list(args.models) + list(args.extra_models)
        logger.info(f"Extra models added: {args.extra_models}")


def main() -> None:
    args = parse_args()
    
    # 如果只是列出可用模型，直接退出（模型列表已在 print_args 中显示）
    if args.list_available_models:
        sys.exit(0)
    
    # 准备模型列表（应用过滤规则）
    prepare_models_list(args)

    model_configer = ConfigManager(args=args)
    total_scenarios_num = model_configer.get_scenarios_num()

    logger.info(f"Begin to run test, all model test num is {total_scenarios_num}. \n")
    scenario_result = []
    error_scenario = []

    # breakpoint()
    # 循环所有模型
    for index, model in enumerate(model_configer.all_model_configs):
        model_name = model_configer.get_model_name(model)
        model_description = model_configer.get_model_description(model)
        logger.info(f"Run {model_name} test, {index + 1} / {total_scenarios_num}")

        # 循环所有支持的任务
        for task_name in model_configer.get_tasks(model):
            if task_name not in args.tasks:
                continue
            task_runner = model_configer.get_task_runner(task_name)
            result = task_runner(model_description=model_description,
                                task_description=task_name,
                                model=model,
                                model_configer=model_configer,
                                args=args)()
            scenario_result.append(result)

        logger.info(f"Finish {model_name} test, {index + 1} / {total_scenarios_num}.\n")

    logger.info(f"Finish all jobs run ipipe from main.py.")

if __name__ == "__main__":
    main()

    # 停留几秒，给后置收集性能数据动作留出buffer
    time.sleep(30)