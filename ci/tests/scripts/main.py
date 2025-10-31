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


logger = create_color_logger(name=__name__)


def main() -> None:
    args = parse_args()

    model_configer = ConfigManager(args=args)
    total_scenarios_num = model_configer.get_scenarios_num()

    logger.info(f"Begin to run test, all model test num is {total_scenarios_num}. \n")
    scenario_result = []
    error_scenario = []

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