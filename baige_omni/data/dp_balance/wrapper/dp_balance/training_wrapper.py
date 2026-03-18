# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0

"""training wrapper."""

from functools import wraps
from more_itertools import peekable
import torch

from megatron.core import mpu
from megatron.training import get_args, get_timers

from baige_omni.data.dp_balance.dataloader.warmup import (
    set_warmup_c1,
    set_warmup_groups,
    solve_computation_coef,
    get_seq_coefs,
    set_seq_coefs,
)
from baige_omni.data.dp_balance.wrapper.dp_balance.rerun_state_wrapper import RerunDataIterator

def train_log_decorator(training_log):
    """Training log decorator for collecting warmup time in data parallel balancing

    The main function of this decorator is to measure and record the forward computation time
    of the master node during the warmup phase (specific iterations) of data parallel balancing,
    which will be used for subsequent load balancing model calculations.

    Working principle:
    - Time collection is only activated when the following conditions are met:
        1. Current process is the master node of data parallel group (rank == 0)
        2. Data parallel balancing is enabled (use_dp_balance = True)
        3. Current iteration is in warmup phase (dp_balance_warmup_iters)

    Args:
        training_log: Original training log function to be decorated

    Returns:
        function: Wrapped training log function with same signature and return value as original
    """
    @wraps(training_log)
    def wrapper(*args, **kwargs):
        nonlocal training_log
        args_train = get_args()
        timers = get_timers()
        iteration = args_train.curr_iteration
        dp_group = mpu.get_data_parallel_group_gloo(
            with_context_parallel=False,
            partial_data_parallel=False,
        )
        rank = torch.distributed.get_rank(dp_group)
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(dp_group)
        active_btime, active_etime = None, None
        if (rank == data_parallel_global_ranks[0] and
            args_train.use_dp_balance and
            iteration in args_train.dp_balance_warmup_iters):
            active_btime = timers('interval-time').active_time()
        ret = training_log(*args, **kwargs)
        if (rank == data_parallel_global_ranks[0] and
            args_train.use_dp_balance and
            iteration in args_train.dp_balance_warmup_iters):
            active_etime = timers('interval-time').active_time()
            set_warmup_c1((active_etime - active_btime) * 1000)
        return ret
    return wrapper


def safe_peek(iterator):
    """
    Safely peek at the first element of an iterator while returning the complete iterator

    Args:
        iterator: Iterator object to peek at, can be None

    Returns:
        tuple: Tuple containing two elements:
            - First element: first element of iterator, returns None if iterator is None
            - Second element: complete iterator containing all elements starting from the first

    Raises:
        StopIteration: Raised when iterator is empty
    """
    if iterator is None:
        return None, iterator
    try:
        new_iterator = peekable(iterator)
    except StopIteration:
        raise
    first = new_iterator.peek()
    # Return the first element and a new iterator that starts from the first element
    return first, RerunDataIterator(new_iterator)


def train_step_decorator(train_step):
    """Training step decorator implementing data parallel load balancing logic encapsulation.

    This decorator is one of the core components of the data parallel balancing system.
    Its main responsibilities include:
    1. Safely prefetching data before forward propagation and setting warmup groups
       to collect sample features for load modeling
    2. Executing original training step and collecting computation latency data
    3. Fitting computation complexity model (quadratic + linear + constant terms)
       on the master node of data parallel group
    4. Broadcasting calibrated load model coefficients to all data parallel processes

    Load model formula:
        load ≈ SEQ2_COEF × seq_len² + SEQ_COEF × seq_len + SEQ_NUM_COEF

    This model is used for subsequent data reordering and load balancing decisions.

    Args:
        train_step: Original training step function to be decorated

    Returns:
        function: Wrapped training step function with same signature and return value as original
    """
    @wraps(train_step)
    def wrapper(*args, **kwargs):
        nonlocal train_step
        args_list = list(args)
        data_iterator = args_list[1]
        timers = get_timers()
        args_train = get_args()

        dp_group = mpu.get_data_parallel_group_gloo(
            with_context_parallel=False,
            partial_data_parallel=False,
        )
        rank = torch.distributed.get_rank(dp_group)
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(dp_group)
        iteration = args_train.curr_iteration
        if args_train.use_dp_balance:
            data, new_itertor = safe_peek(data_iterator)
            if data is not None:
                set_warmup_groups(data)

            args = (*args[:1], new_itertor, *args[2:])
        ret = train_step(*args, **kwargs)
        end_flag = solve_computation_coef()
        seq2_coef, seq_coef, seq_num_coef = get_seq_coefs()
        if (args_train.use_dp_balance and
                iteration == args_train.dp_balance_warmup_iters[-1] + 1):
            if rank == data_parallel_global_ranks[0]:
                comp_coefs = torch.tensor([float(seq2_coef),
                                           float(seq_coef), float(seq_num_coef)],
                                          dtype=torch.float, device=torch.device("cpu"))
            else:
                comp_coefs = torch.tensor([float(1.0),
                                           float(1.0), float(1.0)],
                                          dtype=torch.float, device=torch.device("cpu"))
            torch.distributed.broadcast(comp_coefs, src=data_parallel_global_ranks[0], group=dp_group)
            seq2_coef, seq_coef, seq_num_coef = (comp_coefs[0].item(),
                                                 comp_coefs[1].item(),
                                                 comp_coefs[2].item())
            set_seq_coefs(seq2_coef, seq_coef, seq_num_coef)
        return ret

    return wrapper

