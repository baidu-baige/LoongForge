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

def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='aiak tools',
                                     allow_abbrev=False)
    config_dir = "configs"
    task_dir = "tasks"

    parser.add_argument("--models",
                        type=str,
                        nargs='+',
                        default=ConfigManager.get_all_models(config_dir),
                        choices=ConfigManager.get_all_models(config_dir),
                        help="The model we need to test.")
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
    parser.add_argument("--ckpt_loss_diff",
                        type=float,
                        default=0.1,
                        help="ckpt_loss_diff.")
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

    args = parser.parse_args()
    print_args(args)
    return args