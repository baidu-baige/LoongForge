# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
data balance.

This module provides functionality for load balancing across data parallel ranks
in distributed training. It handles sample redistribution to ensure even workload
distribution among DP ranks.

Main functions:
- gather_sample_info_across_dp: Gather sample information across DP ranks
- solve_sample_dp_reorder_plan: Compute optimal redistribution plan
- redistribute_tensors: Execute tensor redistribution
- reorder_data_for_internvl: Reorder InternVL model data
- reorder_data_for_vlm: Reorder VLM data
- reorder_data_across_dp: Main entry point for data reordering
"""

import torch
from functools import partial
from megatron.core import mpu
import torch.distributed as dist
from typing import List

from megatron.training import get_args
from loongforge.utils import constants
from loongforge.data.dp_balance.rebalance.pack import (
    # intern_vl
    InternVLDataSample,
    depack_data_for_intern_vl,
    pack_data_for_intern_vl,
    # vlm
    VLMDataSample,
    depack_data_for_vlm,
    pack_data_for_vlm,
)

from loongforge.data.dp_balance.rebalance.reconstruct import (
    # intern_vl
    reconstruct_llm_for_internvl,
    reconstruct_image_flags_for_internvl,
    reconstruct_pixel_values_for_internvl,
    # vlm
    reconstruct_llm_for_vlm,
    reconstruct_visual_grid_thw_for_vlm,
    reconstruct_visual_for_vlm,
)

from loongforge.data.dp_balance.rebalance.warmup import (
    load_estimate_per_sample,
)


class _MicroBatchLoadTracker:
    """Tracks accumulated DP costs across micro-batches within an iteration.

    In distributed training with multiple micro-batches per iteration,
    this tracker accumulates the per-DP-rank load from previous micro-batches
    so that subsequent micro-batch balancing decisions consider the total
    iteration load.
    """

    def __init__(self):
        self._dp_costs = None
        self._iteration = -1
        # Cumulative skip/apply counters
        self._vit_skip = 0
        self._vit_apply = 0
        self._dp_skip = 0
        self._dp_apply = 0

    def get_historical_costs(self, dp_size):
        """Get accumulated costs, auto-resetting on new iteration."""
        args = get_args()
        iteration = args.curr_iteration
        if (
            self._iteration != iteration
            or self._dp_costs is None
            or len(self._dp_costs) != dp_size
        ):
            self._dp_costs = [0.0] * dp_size
            self._iteration = iteration
        return list(self._dp_costs)

    def update(self, micro_batch_costs):
        """Add this micro-batch's per-DP costs to the accumulator."""
        if self._dp_costs is not None:
            for i in range(len(self._dp_costs)):
                self._dp_costs[i] += micro_batch_costs[i]

    def record_skip(self, caller):
        """Record a skip event for the given caller category."""
        if caller == "ViT":
            self._vit_skip += 1
        else:
            self._dp_skip += 1

    def record_apply(self, caller):
        """Record an apply event for the given caller category."""
        if caller == "ViT":
            self._vit_apply += 1
        else:
            self._dp_apply += 1

    def get_stats_str(self):
        """Return a formatted summary string of skip/apply counters."""
        vit_total = self._vit_skip + self._vit_apply
        vlm_total = self._dp_skip + self._dp_apply
        return (
            f"ViT_rebalance: {self._vit_apply}/{vit_total} applied, "
            f"VLM_rebalance: {self._dp_apply}/{vlm_total} applied"
        )


_load_tracker = _MicroBatchLoadTracker()


def get_dp_group_by_device(tensor):
    """
    Get the appropriate data parallel group based on tensor's device type.

    For CUDA tensors, returns the data parallel group. When encoder hetero DP is enabled,
    temporarily switches to text_decoder state to get the tensor-and-data parallel group,
    then switches back to image_encoder state.

    For non-CUDA tensors (e.g., CPU), returns the gloo-based data parallel group.

    Args:
        tensor: A tensor to determine the device type and select the appropriate DP group.

    Returns:
        The data parallel group for the tensor's device.
    """
    from loongforge.train.initialize import change_parallel_state

    # if tensor is on CUDA, it's for vit balance
    if tensor.is_cuda:
        args = get_args()
        if args.enable_encoder_hetero_dp:
            change_parallel_state("text_decoder")
            dp_group = mpu.get_tensor_and_data_parallel_group()
            change_parallel_state("image_encoder")
            return dp_group
        else:
            return mpu.get_data_parallel_group(
                with_context_parallel=False,
                partial_data_parallel=False,
            )
    # if tensor is on CPU, it's for VLM balance
    else:
        return mpu.get_data_parallel_group_gloo(
            with_context_parallel=False,
            partial_data_parallel=False,
        )


def gather_sample_info_across_dp(local_seq_lengths: torch.Tensor):
    """
    Gather per-sample sequence length information across all DP ranks.
    1) Gathers the sample count from each DP rank
    2) Pads local tensors to a uniform size
    3) All-gathers sequence lengths and local indices
    4) Reconstructs global per-sample metadata

    Returns:
        global_seq_lengths:  concatenated sequence lengths of all samples
        all_local_sample_index: original local index of each sample
        all_sample_src_dp_rank: source DP rank of each sample
    """
    assert local_seq_lengths.dim() == 1, "local_seq_lengths must be a 1D tensor."

    # Number of local samples on this DP rank
    local_sample_num = int(local_seq_lengths.numel())

    # -----------------------------
    # Step 1: Gather sample counts
    # -----------------------------
    local_sample_num_tensor = torch.tensor(
        [local_sample_num],
        device=local_seq_lengths.device,
        dtype=torch.int,
    )

    dp_group = get_dp_group_by_device(local_seq_lengths)
    dp_size = dp_group.size()

    all_sample_count_list = [
        torch.zeros_like(local_sample_num_tensor) for _ in range(dp_size)
    ]

    dist.all_gather(
        all_sample_count_list,
        local_sample_num_tensor,
        group=dp_group,
    )

    sample_num_list = [int(c.item()) for c in all_sample_count_list]
    max_sample_num = max(sample_num_list)

    # ------------------------------------------------
    # Step 2: Pad local lengths and indices to max size
    # ------------------------------------------------
    padded_len = torch.zeros(
        max_sample_num,
        dtype=local_seq_lengths.dtype,
        device=local_seq_lengths.device,
    )
    padded_len[:local_sample_num] = local_seq_lengths

    local_index = torch.arange(
        local_sample_num,
        device=local_seq_lengths.device,
        dtype=torch.int,
    )
    padded_idx = torch.full(
        (max_sample_num,),
        -1,
        dtype=torch.int,
        device=local_seq_lengths.device,
    )
    padded_idx[:local_sample_num] = local_index

    # --------------------------------
    # Step 3: All-gather padded tensors (merged into single communication)
    # --------------------------------
    padded_combined = torch.stack(
        [padded_len.to(torch.int), padded_idx], dim=0
    )  # [2, max_sample_num]
    gathered_combined = [
        torch.zeros_like(padded_combined) for _ in range(dp_size)
    ]

    dist.all_gather(gathered_combined, padded_combined, group=dp_group)

    gathered_len = [g[0] for g in gathered_combined]
    gathered_idx = [g[1] for g in gathered_combined]

    # --------------------------------------------------
    # Step 4: Remove padding and build global metadata
    # --------------------------------------------------
    local_sample_lengths = []
    local_sample_index = []
    local_sample_dp_rank = []

    for dp_rank in range(dp_size):
        sample_num = sample_num_list[dp_rank]
        assert sample_num >= 0, f"sample_num must be >= 0, but got {sample_num}"
        if sample_num == 0:
            continue

        local_sample_lengths.append(gathered_len[dp_rank][:sample_num])
        local_sample_index.append(gathered_idx[dp_rank][:sample_num])
        local_sample_dp_rank.append(
            torch.full(
                (sample_num,),
                dp_rank,
                dtype=torch.int,
                device=local_seq_lengths.device,
            )
        )

    # Concatenate into global tensors
    global_seq_lengths = torch.cat(local_sample_lengths, dim=0)
    all_local_sample_index = torch.cat(local_sample_index, dim=0)
    all_sample_src_dp_rank = torch.cat(local_sample_dp_rank, dim=0)

    return (
        global_seq_lengths,
        all_local_sample_index,
        all_sample_src_dp_rank,
    )


def solve_sample_dp_reorder_plan(
    global_sample_lengths: torch.Tensor,
    local_sample_index: torch.Tensor,
    sample_src_dp_rank: torch.Tensor,
    cost_fn=None,
    pack_len_ratio: float = None,  # None = no constraint (ViT mode)
    max_iter: int = 20,
    swap_tolerance: float = 0.05,
    dp_historical_costs=None,
    cross_micro_batch_balance: bool = True,
    caller: str = "",
):
    """
    Unified DP load balancing solver.

    Modes:
        - pack_len_ratio is not None -> LLM packing mode
        - pack_len_ratio is None -> ViT mode (no constraint)

    Args:
        global_sample_lengths: Global sequence lengths of all samples.
        local_sample_index: Local sample indices corresponding to global samples.
        sample_src_dp_rank: Source DP rank of each sample.
        cost_fn: Cost function for load balancing. Defaults to squared length (ViT mode).
        pack_len_ratio: Maximum pack length ratio for LLM packing mode. None for ViT mode.
        max_iter: Maximum number of refinement iterations.
        swap_tolerance: Tolerance for swap refinement.
        dp_historical_costs: Accumulated per-DP costs from previous micro-batches
            in the current iteration. When provided, the solver considers these
            historical costs during greedy assignment and refinement so that
            the total iteration load is balanced across DP ranks.
            None means no historical costs (single micro-batch or first micro-batch).
        cross_micro_batch_balance: Whether to use historical costs from previous
            micro-batches for balancing. When False, dp_historical_costs is ignored.
            Set to False for ViT mode where each micro-batch can be independently
            well-balanced without packing constraints.
        caller: Identifier for the calling context (e.g., "InternVL", "VLM", "ViT"),
            used in log messages.

    Returns:
        Tuple of (plan, micro_batch_dp_costs):
            - plan: List[List[Tuple[int, int]]] or None. DP reorder plan, where each
              inner list corresponds to a DP rank and contains (local_idx, src_dp_rank)
              tuples. None if balancing is not needed or not beneficial.
            - micro_batch_dp_costs: List[float]. Per-DP costs contributed by this
              micro-batch only (after reordering, or original distribution if skipped).

    Raises:
        ValueError: If input tensors have invalid dimensions.
    """

    dp_group = get_dp_group_by_device(global_sample_lengths)
    dp_size = dp_group.size()

    # ----------------------------------------
    # 1) cost function
    # ----------------------------------------
    if cost_fn is None:
        cost_fn = lambda x: float(x * x)  # default: ViT

    # ----------------------------------------
    # 2) build items
    # ----------------------------------------
    items = []
    seq_lens = []

    for sample_len, local_idx, src_rank in zip(
        global_sample_lengths, local_sample_index, sample_src_dp_rank
    ):
        seq_len = int(sample_len)
        cost = float(cost_fn(float(sample_len)))
        items.append((cost, seq_len, int(local_idx), int(src_rank)))
        seq_lens.append(seq_len)

    # Current micro-batch original load per DP (before reorder)
    current_load_per_dp = [0.0] * dp_size
    for cost, _, _, src_rank in items:
        current_load_per_dp[src_rank] += cost

    # Total load including history from previous micro-batches
    has_history = (
        cross_micro_batch_balance
        and dp_historical_costs is not None
        and any(c > 0 for c in dp_historical_costs)
    )
    if has_history:
        total_load_per_dp = [
            dp_historical_costs[i] + current_load_per_dp[i]
            for i in range(dp_size)
        ]
    else:
        total_load_per_dp = current_load_per_dp

    dp_average_load = sum(total_load_per_dp) / dp_size
    max_dp_load = max(total_load_per_dp)
    max_seq_load = max([item[0] for item in items])

    imbalance_ratio = (max_dp_load / dp_average_load) - 1 if dp_average_load > 0 else 0
    

    tag = f"[DP Balance][{caller}] " if caller else "[DP Balance] "

    args = get_args()
    dp_rank = dist.get_rank(dp_group)
    is_first_dp_rank = dp_rank == 0
    tp_rank = mpu.get_tensor_model_parallel_rank()
    is_first_tp_rank = tp_rank == 0
    verbose = getattr(args, "dp_balance_verbose", False) and is_first_dp_rank and is_first_tp_rank

    # Select trigger threshold based on caller (VLM vs ViT mode)
    if caller == "ViT":
        trigger_threshold = getattr(args, "dp_balance_trigger_threshold_vit", 0.2)
    else:
        trigger_threshold = getattr(args, "dp_balance_trigger_threshold_vlm", 0.2)

    if (
        max_seq_load > dp_average_load
        or imbalance_ratio < trigger_threshold
    ):
        _load_tracker.record_skip(caller)
        if verbose:
            skip_reason = (
                "single sample dominates avg load"
                if max_seq_load > dp_average_load
                else f"imbalance {imbalance_ratio:.4f} < {trigger_threshold}"
            )
            load_str = ", ".join(f"{v:.1f}" for v in current_load_per_dp)
            print(
                f"{tag}SKIP | reason: {skip_reason}\n"
                f"  imbalance : {imbalance_ratio:.4f}\n"
                f"  load/dp   : [{load_str}]\n"
                f"  cumulative: {_load_tracker.get_stats_str()}"
            )
        return None, current_load_per_dp

    # LPT
    items.sort(key=lambda x: x[0], reverse=True)
    # ----------------------------------------
    # 3) init states (start from historical costs for micro-batch awareness)
    # ----------------------------------------
    if has_history:
        dp_costs = list(dp_historical_costs)
    else:
        dp_costs = [0.0] * dp_size
    dp_buckets = [[] for _ in range(dp_size)]

    use_pack_constraint = pack_len_ratio is not None

    if use_pack_constraint:
        avg_pack_len = sum(seq_lens) / dp_size
        pack_cap = avg_pack_len * pack_len_ratio
        dp_pack_lens = [0] * dp_size

        def can_place(r, seq_len):
            return dp_pack_lens[r] + seq_len <= pack_cap

    else:
        dp_pack_lens = None

        def can_place(r, seq_len):
            return True

    # ----------------------------------------
    # 4) greedy assignment
    # ----------------------------------------
    for cost, seq_len, local_idx, src_rank in items:
        placed = False

        for r in sorted(range(dp_size), key=lambda i: dp_costs[i]):
            if can_place(r, seq_len):
                dp_buckets[r].append((cost, seq_len, local_idx, src_rank))
                dp_costs[r] += cost
                if dp_pack_lens is not None:
                    dp_pack_lens[r] += seq_len
                placed = True
                break

        if not placed:
            r = min(range(dp_size), key=lambda i: dp_costs[i])
            dp_buckets[r].append((cost, seq_len, local_idx, src_rank))
            dp_costs[r] += cost
            if dp_pack_lens is not None:
                dp_pack_lens[r] += seq_len

    # ----------------------------------------
    # 5) refinement
    # ----------------------------------------
    for _ in range(max_iter):
        max_r = max(range(dp_size), key=lambda i: dp_costs[i])
        min_r = min(range(dp_size), key=lambda i: dp_costs[i])

        if (dp_costs[max_r] - dp_costs[min_r]) / max(
            dp_costs[min_r], 1e-6
        ) < swap_tolerance:
            break

        max_bucket = dp_buckets[max_r]
        min_bucket = dp_buckets[min_r]

        if not max_bucket or not min_bucket:
            break

        a = max(max_bucket, key=lambda x: x[0])
        b = min(min_bucket, key=lambda x: x[0])

        cost_a, len_a = a[0], a[1]
        cost_b, len_b = b[0], b[1]

        old_diff = max(dp_costs) - min(dp_costs)

        # ---- move
        move_ok = (not use_pack_constraint) or (dp_pack_lens[min_r] + len_a <= pack_cap)

        if move_ok:
            move_costs = list(dp_costs)
            move_costs[max_r] -= cost_a
            move_costs[min_r] += cost_a
            move_diff = max(move_costs) - min(move_costs)
        else:
            move_diff = float("inf")

        # ---- swap
        swap_ok = (not use_pack_constraint) or (
            dp_pack_lens[max_r] - len_a + len_b <= pack_cap
            and dp_pack_lens[min_r] - len_b + len_a <= pack_cap
        )

        if swap_ok:
            swap_costs = list(dp_costs)
            swap_costs[max_r] = dp_costs[max_r] - cost_a + cost_b
            swap_costs[min_r] = dp_costs[min_r] - cost_b + cost_a
            swap_diff = max(swap_costs) - min(swap_costs)
        else:
            swap_diff = float("inf")

        # ---- select
        if move_diff < old_diff or swap_diff < old_diff:
            if move_diff <= swap_diff:
                a_idx = max_bucket.index(a)
                max_bucket.pop(a_idx)
                min_bucket.append(a)
                dp_costs = move_costs
                if use_pack_constraint:
                    dp_pack_lens[max_r] -= len_a
                    dp_pack_lens[min_r] += len_a
            else:
                a_idx = max_bucket.index(a)
                b_idx = min_bucket.index(b)
                max_bucket.pop(a_idx)
                min_bucket.pop(b_idx)
                max_bucket.append(b)
                min_bucket.append(a)
                dp_costs = swap_costs
                if use_pack_constraint:
                    dp_pack_lens[max_r] = dp_pack_lens[max_r] - len_a + len_b
                    dp_pack_lens[min_r] = dp_pack_lens[min_r] - len_b + len_a
        else:
            break

    # ----------------------------------------
    # 6) export
    # ----------------------------------------
    # Compute per-DP costs contributed by this micro-batch only
    hist = dp_historical_costs if has_history else [0.0] * dp_size
    micro_batch_dp_costs = [dp_costs[i] - hist[i] for i in range(dp_size)]

    _load_tracker.record_apply(caller)
    if verbose:
        rebalanced_max = max(micro_batch_dp_costs)
        rebalanced_avg = sum(micro_batch_dp_costs) / dp_size if dp_size > 0 else 0
        rebalanced_imbalance = (rebalanced_max / rebalanced_avg) - 1 if rebalanced_avg > 0 else 0
        before_str = ", ".join(f"{v:.1f}" for v in current_load_per_dp)
        after_str = ", ".join(f"{v:.1f}" for v in micro_batch_dp_costs)
        print(
            f"{tag}APPLY\n"
            f"  before    : imbalance={imbalance_ratio:.4f}  load/dp=[{before_str}]\n"
            f"  after     : imbalance={rebalanced_imbalance:.4f}  load/dp=[{after_str}]\n"
            f"  cumulative: {_load_tracker.get_stats_str()}"
        )

    plan = [
        [(local_idx, src_rank) for (_, _, local_idx, src_rank) in bucket]
        for bucket in dp_buckets
    ]
    return plan, micro_batch_dp_costs


def get_reverse_reorder_plan(dp_reorder_plan, dp_size):
    """
    Construct reverse reorder plan from forward reorder plan.

    Args:
        dp_reorder_plan: Forward reorder plan.
            dp_reorder_plan[dst_dp_rank] = [(local_idx, src_dp_rank), ...]
            Each tuple means: dst_dp_rank needs the tensor at local_idx from src_dp_rank.
        dp_size: Number of DP ranks.

    Returns:
        reverse_reorder_plan: Reverse reorder plan.
            reverse_reorder_plan[src_dp_rank] = [(local_idx, dst_dp_rank), ...]
            Each tuple means: src_dp_rank should send the tensor at local_idx to dst_dp_rank.
    """

    if dp_reorder_plan is None:
        return None
    # the recv tensor are sorted by src_rank
    dp_reorder_plan = [sorted(sub, key=lambda x: x[1]) for sub in dp_reorder_plan]
    reverse_plan = [{} for _ in range(dp_size)]

    for tgt_dp_rank in range(len(dp_reorder_plan)):
        for new_idx, (local_idx, src_dp_rank) in enumerate(
            dp_reorder_plan[tgt_dp_rank]
        ):
            reverse_plan[src_dp_rank][local_idx] = (new_idx, tgt_dp_rank)

    result = [
        [v for k, v in sorted(d.items(), key=lambda x: x[0])] for d in reverse_plan
    ]

    return result


def redistribute_tensor_helper(
    local_tensor_list,
    dp_reorder_plan,
    global_seq_lengths,
    all_local_sample_index,
    all_sample_src_dp_rank,
    reconstruct_func,
):
    """
    Redistribute tensors across Data Parallel (DP) ranks according to a precomputed reorder plan.

    This function performs the actual DP communication via all_to_all_single.
    Each rank sends out local tensors and receives remote tensors, then reconstructs
    them into per-sample tensors using `reconstruct_func`.

    Notes:
        - This function EXECUTES communication (all_to_all_single).
        - `dp_reorder_plan` is only a plan; the real data movement happens here.
        - The returned list contains tensors already reordered for the current DP rank.

    Args:
        local_tensor_list:
            List of local tensors owned by the current DP rank.
        dp_reorder_plan:
            DP-level reorder plan. dp_reorder_plan[tgt_dp_rank] specifies which
            (local_idx, src_dp_rank) samples should be sent to tgt_dp_rank.
        global_seq_lengths:
            Global sequence length of each sample.
        all_local_sample_index:
            Local sample indices corresponding to global samples.
        all_sample_src_dp_rank:
            Source DP rank of each global sample.
        reconstruct_func:
            Callable used to reconstruct flattened tensors into original shapes.

    Returns:
        List[torch.Tensor]:
            Reconstructed tensors received by the current DP rank, in reordered order.
    """

    dp_group = get_dp_group_by_device(local_tensor_list[0])
    dp_size = dp_group.size()

    cur_dp_rank = torch.distributed.get_rank(dp_group)

    # tensors_to_send[r]: list of tensors to be sent to DP rank r
    tensors_to_send = [[] for _ in range(dp_size)]

    # send_splits[r]: total number of elements sent to DP rank r
    send_splits = [0 for _ in range(dp_size)]

    # recv_splits[r]: total number of elements expected from DP rank r
    recv_splits = [0 for _ in range(dp_size)]

    # recv_tensor_lengths[r]: per-sample lengths expected from DP rank r
    recv_tensor_lengths = [[] for _ in range(dp_size)]

    # Pre-build (src_dp_rank, local_idx) → seq_length lookup table
    # This replaces per-entry O(N) tensor mask search with O(1) dict lookup
    length_lookup = {}
    for i in range(global_seq_lengths.numel()):
        key = (int(all_sample_src_dp_rank[i]), int(all_local_sample_index[i]))
        length_lookup[key] = int(global_seq_lengths[i])

    # Build send/recv metadata strictly following dp_reorder_plan
    for tgt_dp_rank in range(len(dp_reorder_plan)):
        for local_tensor_idx, src_dp_rank in dp_reorder_plan[tgt_dp_rank]:
            tmp_length = length_lookup[(src_dp_rank, local_tensor_idx)]

            # Case 1: current rank is the source → send to target rank
            if src_dp_rank == cur_dp_rank:
                send_splits[tgt_dp_rank] += tmp_length
                tensors_to_send[tgt_dp_rank].append(local_tensor_list[local_tensor_idx])

            # Case 2: current rank is the target → receive from source rank
            if tgt_dp_rank == cur_dp_rank:
                recv_splits[src_dp_rank] += tmp_length
                recv_tensor_lengths[src_dp_rank].append(tmp_length)

    # Flatten all tensors to be sent
    send_tensor = torch.cat(
        [t.reshape(-1) for row in tensors_to_send for t in row], dim=0
    )

    # Allocate receive buffer
    recv_tensor = torch.zeros(
        sum(recv_splits),
        dtype=local_tensor_list[0].dtype,
        device=local_tensor_list[0].device,
    )

    # Safety checks before all_to_all
    if recv_tensor.dtype != send_tensor.dtype:
        recv_tensor = recv_tensor.to(send_tensor.dtype)

    assert recv_tensor.dtype == send_tensor.dtype
    assert sum(send_splits) == send_tensor.numel()
    assert sum(recv_splits) == recv_tensor.numel()

    # Check if gradients are required for this tensor
    # Use all_to_all that supports gradient backpropagation if required
    require_grad = send_tensor.requires_grad

    if require_grad:
        # Use torch.distributed.nn.functional.all_to_all which supports gradient backprop
        # This API supports backward but incurs slight additional overhead
        from torch.distributed.nn.functional import all_to_all

        send_list = list(send_tensor.split(send_splits))
        # Prepare output buffer as list
        recv_list = [
            torch.empty(
                recv_splits[i], dtype=send_tensor.dtype, device=send_tensor.device
            )
            for i in range(dp_size)
        ]
        # Perform all-to-all communication with gradient support
        # all_to_all expects: all_to_all(output_list, input_list, group=group)
        all_to_all(recv_list, send_list, group=dp_group)
        # Concatenate received tensors
        recv_tensor = torch.cat(recv_list)
        # Free send_tensor and send_list immediately to save memory
        del send_tensor, send_list, recv_list
    else:
        # Execute DP all-to-all communication (no gradient needed)
        dist.all_to_all_single(
            recv_tensor,
            send_tensor,
            recv_splits,
            send_splits,
            group=dp_group,
        )

    # Reconstruct per-sample tensors
    recv_tensor_lengths = [x for dp in recv_tensor_lengths for x in dp]
    # Use contiguous() to ensure proper memory layout and avoid memory fragmentation
    recv_tensor = recv_tensor.contiguous()
    redistributed_tensor_list = torch.split(recv_tensor, recv_tensor_lengths)
    redistributed_tensor_list = [reconstruct_func(t) for t in redistributed_tensor_list]

    return redistributed_tensor_list


def redistribute_tensors(local_tensor_list, dp_reorder_plan, reconstruct_func):
    """
    Redistribute tensors across DP ranks according to the reorder plan.

    This function gathers tensor length information across all DP ranks,
    then redistributes the tensors based on the precomputed reorder plan.

    Args:
        local_tensor_list: List of local tensors to be redistributed.
        dp_reorder_plan: DP reorder plan specifying source and destination ranks.
            If None, tensors are reconstructed in place without redistribution.
        reconstruct_func: Function to reconstruct tensor shape from flattened data.

    Returns:
        List[torch.Tensor]: Redistributed (or reconstructed) tensors.
    """
    if not local_tensor_list:
        return []
    if dp_reorder_plan is None:
        return [reconstruct_func(i) for i in local_tensor_list]
    local_tensor_lengths = torch.tensor(
        [t.numel() for t in local_tensor_list],
        dtype=torch.int,
        device=local_tensor_list[0].device,
    )

    (
        global_tensor_lengths,
        local_tenosr_index,
        tensor_src_dp_rank,
    ) = gather_sample_info_across_dp(local_tensor_lengths)

    return redistribute_tensor_helper(
        local_tensor_list,
        dp_reorder_plan,
        global_tensor_lengths,
        local_tenosr_index,
        tensor_src_dp_rank,
        reconstruct_func,
    )


def reorder_data_for_internvl(data):
    """
    Reorder data across DP ranks for InternVL models.

    This function performs load balancing for InternVL models by:
    1. Depacking the data into individual samples
    2. Computing optimal redistribution plan
    3. Redistributing samples across DP ranks
    4. Repacking the redistributed samples

    Args:
        data: Packed data batch from InternVL dataloader.

    Returns:
        Reordered data batch with samples redistributed across DP ranks.
    """
    from megatron.training import get_args
    args = get_args()
    local_sample_list = depack_data_for_intern_vl(data)
    local_pixel_values = []
    local_input_ids = []
    local_image_flags = []
    local_loss_weights = []
    local_labels = []

    for sample in local_sample_list:
        local_pixel_values.append(sample.pixel_values)
        local_input_ids.append(sample.input_ids)
        local_image_flags.append(sample.image_flags)
        local_loss_weights.append(sample.loss_weights)
        local_labels.append(sample.labels)

    local_input_ids_lengths = torch.tensor(
        [t.shape[-1] for t in local_input_ids],
        dtype=torch.int,
        device=local_input_ids[0].device,
    )

    (
        global_seq_lengths,
        local_sample_index,
        sample_src_dp_rank,
    ) = gather_sample_info_across_dp(local_input_ids_lengths)

    dp_group = get_dp_group_by_device(local_input_ids[0])
    dp_size = dp_group.size()

    # Get accumulated costs from previous micro-batches in this iteration
    historical_costs = _load_tracker.get_historical_costs(dp_size)

    dp_reorder_plan, micro_batch_costs = solve_sample_dp_reorder_plan(
        global_seq_lengths,
        local_sample_index,
        sample_src_dp_rank,
        cost_fn=load_estimate_per_sample,
        pack_len_ratio=args.dp_balance_max_len_ratio_vlm,
        dp_historical_costs=historical_costs,
        caller="InternVL",
    )

    # Accumulate this micro-batch's costs for future micro-batches
    _load_tracker.update(micro_batch_costs)

    if dp_reorder_plan is None:
        return data

    # LLM tensors (input_ids, labels, loss_weights) share the same sample
    # dimensions, gather once and reuse to avoid redundant all_gather calls.
    llm_lengths = torch.tensor(
        [t.numel() for t in local_input_ids],
        dtype=torch.int,
        device=local_input_ids[0].device,
    )
    llm_global_lengths, llm_local_idx, llm_src_rank = (
        gather_sample_info_across_dp(llm_lengths)
    )
    redistributed_input_ids = redistribute_tensor_helper(
        local_input_ids, dp_reorder_plan,
        llm_global_lengths, llm_local_idx, llm_src_rank,
        reconstruct_llm_for_internvl,
    )
    redistributed_labels = redistribute_tensor_helper(
        local_labels, dp_reorder_plan,
        llm_global_lengths, llm_local_idx, llm_src_rank,
        reconstruct_llm_for_internvl,
    )
    redistributed_loss_weights = redistribute_tensor_helper(
        local_loss_weights, dp_reorder_plan,
        llm_global_lengths, llm_local_idx, llm_src_rank,
        reconstruct_llm_for_internvl,
    )

    # pixel_values and image_flags have different element counts, gather separately
    redistributed_pixel_values = redistribute_tensors(
        local_pixel_values, dp_reorder_plan, reconstruct_pixel_values_for_internvl,
    )
    redistributed_image_flags = redistribute_tensors(
        local_image_flags, dp_reorder_plan, reconstruct_image_flags_for_internvl,
    )

    redistributed_samples = []
    for i in range(len(redistributed_input_ids)):
        sample = InternVLDataSample(
            redistributed_pixel_values[i],
            redistributed_input_ids[i],
            redistributed_image_flags[i],
            redistributed_loss_weights[i],
            redistributed_labels[i],
        )
        redistributed_samples.append(sample)

    new_data = pack_data_for_intern_vl(redistributed_samples)
    return new_data


def reorder_data_for_vlm(data):
    """
    Reorder data across DP ranks for VLM (Vision-Language Model) models.

    This function performs load balancing for VLM models by:
    1. Depacking the data into individual samples (LLM tokens, labels, attention masks,
       pixel values for images and videos)
    2. Computing optimal redistribution plan based on sequence lengths
    3. Redistributing samples across DP ranks
    4. Repacking the redistributed samples

    Args:
        data: Packed data batch from VLM dataloader.

    Returns:
        Reordered data batch with samples redistributed across DP ranks.
    """
    args = get_args()
    local_sample_list = depack_data_for_vlm(data)

    # llm_input
    local_tokens = []
    local_attn_mask = []
    local_labels = []

    # visual_input
    local_pixel_values_images = []
    local_pixel_values_videos = []
    local_image_thws = []
    local_vid_thws = []

    for sample in local_sample_list:
        local_tokens.append(sample.tokens)
        local_labels.append(sample.labels)
        local_attn_mask.append(sample.attn_mask)
        if sample.pixel_values_images is not None:
            local_pixel_values_images.append(sample.pixel_values_images)
            local_image_thws.append(sample.image_thw)
        if sample.pixel_values_videos is not None:
            local_pixel_values_videos.append(sample.pixel_values_videos)
            local_vid_thws.append(sample.vid_thw)

    local_token_lengths = torch.tensor(
        [t.shape[-1] for t in local_tokens],
        dtype=torch.int,
        device=local_tokens[0].device,
    )

    (
        global_seq_lengths,
        local_sample_index,
        sample_src_dp_rank,
    ) = gather_sample_info_across_dp(local_token_lengths)

    dp_group = get_dp_group_by_device(local_tokens[0])
    dp_size = dp_group.size()

    # Get accumulated costs from previous micro-batches in this iteration
    historical_costs = _load_tracker.get_historical_costs(dp_size)

    dp_reorder_plan, micro_batch_costs = solve_sample_dp_reorder_plan(
        global_seq_lengths,
        local_sample_index,
        sample_src_dp_rank,
        cost_fn=load_estimate_per_sample,
        pack_len_ratio=args.dp_balance_max_len_ratio_vlm,
        dp_historical_costs=historical_costs,
        caller="VLM",
    )

    # Accumulate this micro-batch's costs for future micro-batches
    _load_tracker.update(micro_batch_costs)

    if dp_reorder_plan is None:
        return data

    # Helper to redistribute a tensor list, reusing the shared gather results
    def _redistribute(tensor_list, reconstruct_fn):
        local_lengths = torch.tensor(
            [t.numel() for t in tensor_list],
            dtype=torch.int,
            device=tensor_list[0].device,
        )
        global_lengths, local_idx, src_rank = gather_sample_info_across_dp(
            local_lengths
        )
        return redistribute_tensor_helper(
            tensor_list,
            dp_reorder_plan,
            global_lengths,
            local_idx,
            src_rank,
            reconstruct_fn,
        )

    # LLM tensors share the same sample count, gather once and reuse
    llm_lengths = torch.tensor(
        [t.numel() for t in local_tokens],
        dtype=torch.int,
        device=local_tokens[0].device,
    )
    llm_global_lengths, llm_local_idx, llm_src_rank = (
        gather_sample_info_across_dp(llm_lengths)
    )
    redistributed_tokens = redistribute_tensor_helper(
        local_tokens, dp_reorder_plan,
        llm_global_lengths, llm_local_idx, llm_src_rank,
        reconstruct_llm_for_vlm,
    )
    redistributed_labels = redistribute_tensor_helper(
        local_labels, dp_reorder_plan,
        llm_global_lengths, llm_local_idx, llm_src_rank,
        reconstruct_llm_for_vlm,
    )
    redistributed_attn_mask = redistribute_tensor_helper(
        local_attn_mask, dp_reorder_plan,
        llm_global_lengths, llm_local_idx, llm_src_rank,
        reconstruct_llm_for_vlm,
    )

    redistributed_pixel_values_images = None
    redistributed_image_thws = None

    if local_pixel_values_images:
        visual_size = local_pixel_values_images[0].shape[-1]
        redistributed_pixel_values_images = _redistribute(
            local_pixel_values_images,
            partial(reconstruct_visual_for_vlm, visual_size=visual_size),
        )
        redistributed_image_thws = _redistribute(
            local_image_thws,
            reconstruct_visual_grid_thw_for_vlm,
        )

    redistributed_pixel_values_videos = None
    redistributed_video_thws = None
    if local_pixel_values_videos:
        visual_size = local_pixel_values_videos[0].shape[-1]
        redistributed_pixel_values_videos = _redistribute(
            local_pixel_values_videos,
            partial(reconstruct_visual_for_vlm, visual_size=visual_size),
        )
        redistributed_video_thws = _redistribute(
            local_vid_thws,
            reconstruct_visual_grid_thw_for_vlm,
        )

    redistributed_samples = []
    for i in range(len(redistributed_tokens)):
        redistributed_pixel_values_image = None
        redistributed_image_thw = None
        redistributed_pixel_values_video = None
        redistributed_video_thw = None
        if redistributed_pixel_values_images is not None:
            redistributed_pixel_values_image = redistributed_pixel_values_images[i]
            redistributed_image_thw = redistributed_image_thws[i]
        if redistributed_pixel_values_videos is not None:
            redistributed_pixel_values_video = redistributed_pixel_values_videos[i]
            redistributed_video_thw = redistributed_video_thws[i]
        sample = VLMDataSample(
            redistributed_tokens[i],
            redistributed_labels[i],
            redistributed_attn_mask[i],
            redistributed_pixel_values_image,
            redistributed_image_thw,
            redistributed_pixel_values_video,
            redistributed_video_thw,
        )
        redistributed_samples.append(sample)
    new_data = pack_data_for_vlm(redistributed_samples)

    return new_data


def reorder_data_across_dp(data):
    """
    Reorder data across DP ranks for load balancing.

    This function is the main entry point for redistributing samples across
    data parallel ranks to achieve better load balance. It automatically
    detects the model family and applies the appropriate reordering strategy.

    Args:
        data: Input data batch from the dataloader. For InternVL models, it should
            be a list where the last element contains the packed batch data.
            For other VLM models, the format is similar.

    Returns:
        list: Reordered data batch with samples redistributed across DP ranks.

    Raises:
        ValueError: If the model_family is not supported. Supported families are
            'intern_vl' and those defined in constants.VisionLanguageModelFamilies.
    """
    args = get_args()
    if args.model_family == "intern_vl":
        result = data[:-1] + [reorder_data_for_internvl(data[-1])]
        return result
    elif args.model_family in constants.VisionLanguageModelFamilies.names():
        result = data[:-1] + [reorder_data_for_vlm(data[-1])]
        return result
    else:
        raise ValueError(
            f"Unsupported model_family '{args.model_family}'. "
            f"Expected one of: "
            f"['intern_vl', "
            f"{', '.join(constants.VisionLanguageModelFamilies.names())}]"
        )
