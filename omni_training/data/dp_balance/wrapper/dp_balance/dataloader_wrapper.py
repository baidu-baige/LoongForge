# Copyright 2026 The OmniTraining Authors.
# SPDX-License-Identifier: Apache-2.0

"""dataloader wrapper"""

from functools import wraps
from megatron.core import mpu
from megatron.training import get_args, get_timers

from omni_training.data.dp_balance.dataloader.data_balance import (
    reorder_data_across_dp,
)


def pin_memory_loop_wrapper(pin_memory_loop):
    """
    Wrap the pin_memory_loop to inject a dataset reordering hook.

    If the original pin_memory_loop does not specify a resort function,
    this wrapper automatically inserts `resort_dataset_fn` as the default.
    """

    @wraps(pin_memory_loop)
    def wrapper(*args, **kwargs):
        nonlocal pin_memory_loop
        if len(args) > 4 and args[4] is None:
            args_list = list(args)
            args_list[4] = resort_dataset_fn
            new_args = tuple(args_list)
        else:
            new_args = args
        pin_memory_loop(*new_args, **kwargs)

    return wrapper


def resort_dataset_fn(idx, data):
    """
    Reorder dataset with support for data parallel balancing

    Determines whether to perform cross-data-parallel reordering of data based on
    training parameters and model parallel state. Reordering is only executed when
    specific conditions are met.

    Args:
        idx: Data index
        data: Original data

    Returns:
        tuple: Returns original index and potentially reordered data
    """
    args = get_args()

    if (
        args.use_dp_balance
        and args.curr_iteration > args.dp_balance_warmup_iters[-1] + 1
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == 0
    ):
        data = reorder_data_across_dp(data)

    return idx, data
