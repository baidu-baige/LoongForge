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
    """Prepare final list of models to run"""
    # Collect all config directories
    extra_dirs = list(args.extra_configs_dirs) if args.extra_configs_dirs else []

    # If --include_optional is specified, add optional_configs directory
    if args.include_optional:
        optional_dir = "optional_configs"
        if optional_dir not in extra_dirs:
            extra_dirs.append(optional_dir)

    # Update extra_configs_dirs in args
    args.extra_configs_dirs = extra_dirs

    # If --optional_subdir is specified, load all models from specified subdirectory
    if args.optional_subdir:
        # Ensure optional_configs is in extra_dirs
        if "optional_configs" not in extra_dirs:
            extra_dirs.append("optional_configs")
            args.extra_configs_dirs = extra_dirs

        # Get all models under specified subdirectory
        subdir_models = ConfigManager.get_models_from_subdir(
            "optional_configs",
            args.optional_subdir
        )

        if subdir_models:
            # If models is empty or is default value, use subdirectory's models
            if not args.models or args.models == ConfigManager.get_all_models(args.configs_dir):
                args.models = subdir_models
            else:
                # Otherwise append subdirectory models to existing model list
                args.models = list(args.models) + subdir_models

        logger.info(f"Models from optional_subdir '{args.optional_subdir}': {subdir_models}")

    # If --extra_models is specified, append extra models to list
    if args.extra_models:
        # Ensure optional_configs is in extra_dirs (because extra_models usually comes from optional_configs)
        if "optional_configs" not in args.extra_configs_dirs:
            args.extra_configs_dirs = list(args.extra_configs_dirs) + ["optional_configs"]

        # Append extra_models to models list
        args.models = list(args.models) + list(args.extra_models)
        logger.info(f"Extra models added: {args.extra_models}")


def main() -> None:
    args = parse_args()
    
    # If only listing available models, exit directly (model list already displayed in print_args)
    if args.list_available_models:
        sys.exit(0)
    
    #Prepare model list (apply filtering rules)
    prepare_models_list(args)

    model_configer = ConfigManager(args=args)
    total_scenarios_num = model_configer.get_scenarios_num()

    logger.info(f"Begin to run test, all model test num is {total_scenarios_num}. \n")
    scenario_result = []
    error_scenario = []

    # breakpoint()
    # Loop through all models
    for index, model in enumerate(model_configer.all_model_configs):
        model_name = model_configer.get_model_name(model)
        model_description = model_configer.get_model_description(model)
        logger.info(f"Run {model_name} test, {index + 1} / {total_scenarios_num}")

        # Loop through all supported tasks
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

    # Pause for a few seconds, leave buffer for post-collection performance data action
    time.sleep(30)