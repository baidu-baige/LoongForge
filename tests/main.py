# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Main entry point for the training script."""

from tools.arguments import parse_args
from tools.color_logger import create_color_logger
from tools.config_manager import ConfigManager
from tasks.base_task import BaseTask
from utils.resume_state import load_state, save_state, mark_model, get_completed_models
from tqdm import tqdm
import time
import sys
import os


logger = create_color_logger(name=__name__)


def prepare_models_list(args):
    """
    Determine the list of models to run based on arguments (models, extra_models, optional_subdir, include_optional).
    Supports running models from default 'configs/' or 'optional_configs/'.
    For specific running modes, please refer to tests/README.md.
    """
    extra_dirs = list(args.extra_configs_dirs) if args.extra_configs_dirs else []
    need_optional = args.include_optional or args.optional_subdir or args.extra_models
    if need_optional:
        if "optional_configs" not in extra_dirs:
            extra_dirs.append("optional_configs")
    args.extra_configs_dirs = extra_dirs

    # If args.models is None (not specified by user), decide default behavior based on other arguments
    if args.models is None:
        if args.optional_subdir or args.extra_models:
            args.models = []
        elif args.include_optional:
            default_models = ConfigManager.get_all_models(args.configs_dir)
            optional_models = ConfigManager.get_models_from_dir("optional_configs", recursive=True)
            args.models = list(default_models) + list(optional_models)
            logger.info("Running all models from configs + optional_configs (Method 5)...")
        else:
            args.models = ConfigManager.get_all_models(args.configs_dir)
    
    # If --optional_subdir is specified, load all models in the specified subdirectory
    if args.optional_subdir:
        # Get all models under the specified subdirectory
        subdir_models = ConfigManager.get_models_from_subdir(
            "optional_configs",
            args.optional_subdir
        )

        if subdir_models:
            args.models = list(args.models) + subdir_models
            logger.info(f"Models from optional_subdir '{args.optional_subdir}': {subdir_models}")
    
    # If --extra_models is specified, append extra models to the list
    if args.extra_models:
        # Append extra_models to the models list
        args.models = list(args.models) + list(args.extra_models)
        logger.info(f"Extra models added: {args.extra_models}")

    # Deduplicate and sort
    if args.models:
        args.models = sorted(list(set(args.models)))


def main() -> None:
    args = parse_args()
    
    # If only listing available models, exit directly (list displayed in print_args)
    if args.list_available_models:
        sys.exit(0)
    
    # Prepare model list (apply filtering rules)
    prepare_models_list(args)

    resume_state = None
    completed_models = set()
    resume_enabled = bool(args.resume_state_file)
    if resume_enabled and os.getenv("RANK", "0") == "0":
        resume_state = load_state(args.resume_state_file)
        completed_models = get_completed_models(
            resume_state,
            args.resume_policy,
            args.tasks,
            args.training_type,
        )
        if completed_models:
            logger.info(f"Resume enabled: skip {len(completed_models)} completed models")

    model_configer = ConfigManager(args=args)
    total_scenarios_num = model_configer.get_scenarios_num()

    logger.info(f"Begin to run test, all model test num is {total_scenarios_num}. \n")
    scenario_result = []
    error_scenario = []

    # Loop through all models
    for index, model in enumerate(model_configer.all_model_configs):
        model_name = model_configer.get_model_name(model)
        model_description = model_configer.get_model_description(model)
        if model_name in completed_models:
            logger.info(f"Skip {model_name} (already completed)")
            continue
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

        if resume_enabled and os.getenv("RANK", "0") == "0":
            model_results = [item for item in BaseTask._validation_results if item.get("model_name") == model_name]
            model_passed = all(item.get("passed") for item in model_results) if model_results else True
            task_passed = {}
            for item in model_results:
                task_name = item.get("task_name") or ""
                if not task_name:
                    continue
                if task_name not in task_passed:
                    task_passed[task_name] = True
                task_passed[task_name] = task_passed[task_name] and bool(item.get("passed"))
            mark_model(
                resume_state,
                model_name,
                passed=model_passed,
                meta={
                    "tasks": list(args.tasks),
                    "training_type": list(args.training_type),
                    "category": BaseTask._get_diff_category(model),
                    "tasks_passed": task_passed,
                },
            )
            save_state(args.resume_state_file, resume_state)

        logger.info(f"Finish {model_name} test, {index + 1} / {total_scenarios_num}.\n")

    logger.info(f"Finish all jobs run ipipe from main.py.")

    rank = os.getenv("RANK")
    is_final_pod = False
    if rank is not None:
        try:
            is_final_pod = int(rank) == int(args.node_nums) - 1
        except Exception:
            is_final_pod = False

    if is_final_pod:
        BaseTask.write_validation_summary()
        if BaseTask.has_validation_failures():
            logger.error("Validation failed: at least one case did not pass.")
            sys.exit(1)

if __name__ == "__main__":
    main()

    # Pause for a few seconds to leave a buffer for the subsequent collection of performance data
    time.sleep(30)