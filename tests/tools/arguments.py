#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import argparse
from tasks import SUPPORTED_TASKS
from tools.config_manager import ConfigManager


def _print_available_models(args, print_func):
    """Print list of available models (follows immediately after arguments)"""
    config_dir = getattr(args, 'configs_dir', 'configs')
    include_optional = getattr(args, 'include_optional', False)
    extra_configs_dirs = getattr(args, 'extra_configs_dirs', []) or []
    
    # Collect all configuration directories
    all_dirs = [config_dir]
    if extra_configs_dirs:
        all_dirs.extend(extra_configs_dirs)
    if include_optional:
        optional_dir = "optional_configs"
        if optional_dir not in all_dirs:
            all_dirs.append(optional_dir)
    
    # Get all models under directories
    available = ConfigManager.list_all_available_models(
        config_dir,
        extra_configs_dirs=all_dirs[1:] if len(all_dirs) > 1 else None
    )
    
    print_func('')
    print_func('-------------------- available models ---------------------')
    
    total_count = 0
    for directory, models in available.items():
        # Group by subdirectories
        grouped = {}
        for model in models:
            if "/" in model:
                subdir, name = model.rsplit("/", 1)
                if subdir not in grouped:
                    grouped[subdir] = []
                grouped[subdir].append(model)
            else:
                if "_root_" not in grouped:
                    grouped["_root_"] = []
                grouped["_root_"].append(model)
        
        # Display directory name
        print_func(f'  [{directory}]')

        # Display models in root directory first
        if "_root_" in grouped:
            for model in sorted(grouped["_root_"]):
                print_func(f'    • {model}')
            del grouped["_root_"]
        
        # Display models in subdirectories (indented)
        for subdir in sorted(grouped.keys()):
            print_func(f'    [{subdir}/]')
            for model in sorted(grouped[subdir]):
                print_func(f'      • {model}')
        
        total_count += len(models)
    
    print_func(f'  Total: {total_count} models')
    print_func('------------------------------------------------------------')


def print_args(args, indents=48, std_out=print, need_endl=False):
    """Print arguments."""
    def __custom_print(msg, std_out=print, need_endl=False):
        std_out(msg)
        if need_endl:
            std_out("\n")

    from functools import partial
    custom_print = partial(__custom_print, std_out=std_out, need_endl=need_endl)

    custom_print('------------------------ arguments ------------------------')
    str_list = []
    for arg in vars(args):
        dots = '.' * (indents - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        custom_print(arg)

    custom_print('-------------------- end of arguments ---------------------')
    
    # Print available models list
    _print_available_models(args, custom_print)

def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='aiak tools',
                                     allow_abbrev=False)
    config_dir = "configs"
    optional_config_dir = "optional_configs"
    task_dir = "tasks"
    
    # Get all available models (including subdirectories under main config directory and optional config directory)
    all_available_models = ConfigManager.get_all_models(
        config_dir, 
        extra_configs_dirs=[optional_config_dir],
        recursive_extra=True
    )

    parser.add_argument("--models",
                        type=str,
                        nargs='+',
                        default=ConfigManager.get_all_models(config_dir),
                        help=f"The model we need to test. Support formats: 'model_name' or 'subdir/model_name'. Available models: {', '.join(all_available_models[:10])}...")
    parser.add_argument("--tasks",
                        type=str,
                        nargs='+',
                        default=["check_correctness_task"],
                        choices=ConfigManager.get_all_model_tasks(task_dir),
                        help="The task we need to run for models.")
    parser.add_argument("--configs_dir",
                        type=str,
                        default=config_dir,
                        help="The dir we save all configs.")
    parser.add_argument("--tasks_dir",
                        type=str,
                        default=task_dir,
                        help="The dir task.")
    parser.add_argument("--node_nums",
                        type=int,
                        default=1,
                        help="node_nums.")
    parser.add_argument("--gpu_nums",
                        type=int,
                        default=8,
                        help="gpu_nums.")
    parser.add_argument("--accuracy_relative_tolerance",
                        type=float,
                        default=0.02,
                        help="accuracy_relative_tolerance.")
    parser.add_argument("--performance_relative_tolerance",
                        type=float,
                        default=0.05,
                        help="performance_relative_tolerance.")
    parser.add_argument('--use_nccl',
                        action='store_true',
                        help='Use bccl or nccl for distributed training')
    parser.add_argument('--training_type',
                        type=str,
                        nargs='+',
                        default=['pretrain', 'sft'],
                        choices=['pretrain', 'sft'],
                        help='pretrain or sft')
    parser.add_argument('--dry_run',
                        action='store_true',
                        help='dry run')                 
    parser.add_argument("--timeout",
                        type=int,
                        default=600,
                        help="timeout.")
    
    # Optional configuration related parameters
    parser.add_argument("--extra_configs_dirs",
                        type=str,
                        nargs='*',
                        default=[],
                        help="Additional config directories to load models from.")
    parser.add_argument("--include_optional",
                        action='store_true',
                        help="Include models from optional_configs directory (including subdirectories).")
    parser.add_argument("--list_available_models",
                        action='store_true',
                        help="List all available models from all config directories and exit.")
    parser.add_argument("--optional_subdir",
                        type=str,
                        default=None,
                        help="Load all models from a specific subdirectory under optional_configs (e.g., 'internvl3.5', 'qwen2.5_vl').")
    parser.add_argument("--extra_models",
                        type=str,
                        nargs='*',
                        default=[],
                        help="Additional models to run from optional_configs (e.g., 'internvl3.5/internvl3.5_30b_a3b').")

    args = parser.parse_args()
    print_args(args)
    # breakpoint()
    return args