# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""warmup tools for dp balance"""

import torch
import torch.distributed as dist

import numpy as np
from scipy.optimize import minimize

from typing import Dict, List, Tuple, Union

from megatron.training import get_args
from megatron.core import mpu
from loongforge.utils import constants

# Warm-up profiling buffers
WARMUP_VAR_GROUPS = []  # variables used for fitting load model
WARM_FORWARD_TIME = []  # corresponding forward latency

# Load model coefficients (estimated after warm-up)
# load ≈ a * len^2 + b * len + c * seq_num
SEQ2_COEF = float(0.0)  # quadratic term
SEQ_COEF = float(1.0)  # linear term
SEQ_NUM_COEF = float(0.0)  # per-sample overhead


def get_seq_coefs():
    """Get the coefficients of the attention computation cost model

    Returns:
        Tuple[float, float, float]:
            - Quadratic coefficient (SEQ2_COEF): coefficient for sequence length squared term
            - Linear coefficient (SEQ_COEF): coefficient for sequence length linear term
            - Constant coefficient (SEQ_NUM_COEF): fixed overhead coefficient per sequence
    """
    return SEQ2_COEF, SEQ_COEF, SEQ_NUM_COEF


def set_seq_coefs(seq2_coef, seq_coef, seq_num_coef):
    """Set the coefficients of the attention computation cost model

    Args:
        seq2_coef (float): coefficient for sequence length squared term
        seq_coef (float): coefficient for sequence length linear term
        seq_num_coef (float): fixed overhead coefficient per sequence
    """
    global SEQ2_COEF, SEQ_COEF, SEQ_NUM_COEF
    SEQ2_COEF = seq2_coef
    SEQ_COEF = seq_coef
    SEQ_NUM_COEF = seq_num_coef


def solve_computation_coef(init=(0, 0, 0)):
    """
    Fit the attention computation cost model from warm-up profiling data.

    The per-pack compute load is modeled as:
        load ≈ a * seq_len^2 + b * seq_len + c * seq_num

    Coefficients (a, b, c) are estimated by minimizing the squared error
    between predicted and measured forward latency during warm-up.
    """

    # Global model parameters to be updated in-place
    global SEQ2_COEF, SEQ_COEF, SEQ_NUM_COEF
    global WARM_FORWARD_TIME, WARMUP_VAR_GROUPS

    # Scaling factors to normalize variable magnitudes and improve optimizer stability
    S_a = 1e8  # scale for seq_len^2 term
    S_b = 1e4  # scale for seq_len term
    S_c = 1e1  # scale for seq_num term

    args_train = get_args()
    iteration = args_train.curr_iteration

    dp_group = mpu.get_data_parallel_group_gloo(
        with_context_parallel=False,
        partial_data_parallel=False,
    )

    dp_rank = mpu.get_data_parallel_rank()
    is_dp_root = dp_rank == 0

    # Only run during DP-balance warm-up phase
    if (
        not is_dp_root
        or not args_train.use_vlm_dp_balance
        or iteration != args_train.vlm_dp_balance_warmup_iters[-1] + 1
    ):
        return False

    # Warm-up profiling data:
    #   var_groups: per-DP variable groups [(seq_len^2, seq_len, seq_num), ...]
    #   C: measured forward latency
    var_groups, C = WARMUP_VAR_GROUPS, WARM_FORWARD_TIME

    def softmax_max(vals, alpha=20):
        """
        Smooth approximation of max(), used to model the DP synchronization cost
        dominated by the slowest rank.
        """
        vals = np.array(vals, dtype=float)
        m = vals.max()
        return m + (1 / alpha) * np.log(np.sum(np.exp(alpha * (vals - m))))

    def loss(vars, var_groups, C):
        """
        Objective function:
        minimize the squared error between predicted max DP load and
        observed forward latency.
        """
        x_t, y_t, z_t = vars  # scaled optimization variables
        err = 0.0

        for terms, Ci in zip(var_groups, C):
            # Predicted per-DP loads for this iteration
            vals = [
                (a / S_a) * x_t + (b / S_b) * y_t + (c / S_c) * z_t
                for (a, b, c) in terms
            ]
            err += (softmax_max(vals) - Ci) ** 2

        return err

    # Coefficients are constrained to be non-negative
    bounds = [(0, None)] * 3

    # Solve the nonlinear least-squares problem
    res = minimize(loss, init, args=(var_groups, C), bounds=bounds)

    # Map optimized variables back to original scale
    SEQ2_COEF = res.x[0] / S_a
    SEQ_COEF = res.x[1] / S_b
    SEQ_NUM_COEF = res.x[2] / S_c

    return True


def set_warmup_c1(c1):
    """
    Record forward latency during the DP-balance warm-up phase.

    This function appends the measured forward computation time `c1`
    for the current iteration, which is later used to fit the DP
    computation cost model.
    """
    global WARM_FORWARD_TIME
    args_train = get_args()
    iteration = args_train.curr_iteration
    if (
        not args_train.use_vlm_dp_balance
        or not iteration in args_train.vlm_dp_balance_warmup_iters
        or iteration == args_train.vlm_dp_balance_warmup_iters[0]
    ):
        return
    WARM_FORWARD_TIME.append(c1)


def set_warmup_groups(data: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]):
    """
    Collect per-DP sequence statistics during the warm-up phase.

    For the current iteration, this function computes:
        - sum(seq_len^2)
        - sum(seq_len)
        - number of sequences

    across all micro-batches on each DP rank, gathers them across all DP
    ranks, and stores the resulting per-DP variable group. These statistics
    are later used to fit the DP computation cost model.

    Args:
        data: A single micro-batch dict, or a list of micro-batch dicts
              (one per micro-batch in the iteration).
    """
    global WARMUP_VAR_GROUPS
    args_train = get_args()
    iteration = args_train.curr_iteration

    if (
        not args_train.use_vlm_dp_balance
        or not iteration in args_train.vlm_dp_balance_warmup_iters
        or iteration == args_train.vlm_dp_balance_warmup_iters[0]
    ):
        return

    # Normalize to list for uniform handling
    if isinstance(data, dict):
        data_list = [data]
    else:
        data_list = data

    dp_group = mpu.get_data_parallel_group_gloo(
        with_context_parallel=False,
        partial_data_parallel=False,
    )
    dp_size = dp_group.size()

    # Accumulate statistics across all micro-batches
    total_seq_num = 0
    total_seq_lenth_sum = None
    total_seq_lenth_square_sum = None

    for micro_batch in data_list:
        cu_lengths = None
        if args_train.model_family == "intern_vl":
            cu_lengths = micro_batch["attn_mask"]
        elif args_train.model_family in constants.VisionLanguageModelFamilies.names():
            cu_lengths = micro_batch["cu_lengths"]
        cu_lengths = cu_lengths.squeeze(0)

        # Number of sequences in this micro-batch
        seq_num = cu_lengths.numel() - 1
        total_seq_num += seq_num

        # Per-sequence lengths and their squared values
        seq_lenth = cu_lengths[1:] - cu_lengths[:-1]
        seq_lenth_square = seq_lenth**2

        if total_seq_lenth_sum is None:
            total_seq_lenth_sum = seq_lenth.sum()
            total_seq_lenth_square_sum = seq_lenth_square.sum()
        else:
            total_seq_lenth_sum = total_seq_lenth_sum + seq_lenth.sum()
            total_seq_lenth_square_sum = total_seq_lenth_square_sum + seq_lenth_square.sum()

    seq_num_tensor = torch.tensor(
        [total_seq_num],
        device=total_seq_lenth_sum.device,
        dtype=torch.long,
    )

    # Prepare all-gather buffers
    seq_num_list = [torch.zeros_like(seq_num_tensor) for _ in range(dp_size)]
    seq_lenth_sum_list = [torch.zeros_like(total_seq_lenth_sum) for _ in range(dp_size)]
    seq_lenth_square_sum_list = [
        torch.zeros_like(total_seq_lenth_square_sum) for _ in range(dp_size)
    ]

    # Gather statistics across DP ranks
    dist.all_gather(seq_num_list, seq_num_tensor, group=dp_group)
    dist.all_gather(seq_lenth_sum_list, total_seq_lenth_sum, group=dp_group)
    dist.all_gather(seq_lenth_square_sum_list, total_seq_lenth_square_sum, group=dp_group)

    dp_rank = mpu.get_data_parallel_rank()
    is_dp_root = dp_rank == 0

    # Convert tensors to Python scalars
    seq_lenth_square_sum_list = [t.item() for t in seq_lenth_square_sum_list]
    seq_lenth_sum_list = [t.item() for t in seq_lenth_sum_list]
    seq_num_list = [t.item() for t in seq_num_list]

    # Store per-DP variable group: (sum(seq_len^2), sum(seq_len), seq_num)
    var_group = [
        (a, b, c)
        for (a, b, c) in zip(
            seq_lenth_square_sum_list,
            seq_lenth_sum_list,
            seq_num_list,
        )
    ]
    if is_dp_root:
        WARMUP_VAR_GROUPS.append(var_group)


def load_estimate_per_sample(seq_len) -> float:
    """
    Estimate the relative computation load of a single sample.

    Cost model:
        load ≈ a * seq_len^2 + b * seq_len + c

    This estimate is used for sample reordering and DP load balancing.
    Coefficients are calibrated during the warm-up phase.
    """
    return SEQ2_COEF * seq_len**2 + SEQ_COEF * seq_len + SEQ_NUM_COEF
