# Copyright (c) 2026, AIAK team. All rights reserved.

"""data balance"""

import torch
from functools import partial
from megatron.core import mpu
import torch.distributed as dist
from typing import List

from megatron.training import get_args, get_timers
from omni_training.utils import get_tokenizer
from omni_training.utils import constants
from omni_training.data.dp_balance.dataloader.depack_and_pack import (
    # intern_vl
    InternVLDataSample,
    depack_data_for_intern_vl,
    pack_data_for_intern_vl,
    # vlm
    VLMDataSample,
    depack_data_for_vlm,
    pack_data_for_vlm,
)

from omni_training.data.dp_balance.dataloader.reconstruct import (
    # intern_vl
    reconstruct_llm_for_internvl,
    reconstruct_image_flags_for_internvl,
    reconstruct_pixel_values_for_internvl,
    # vlm
    reconstruct_llm_for_vlm,
    reconstruct_visual_grid_thw_for_vlm,
    reconstruct_visual_for_vlm,
    reconstruct_position_ids_for_vlm,
)

from omni_training.data.dp_balance.dataloader.warmup import (
    load_estimate_per_sample,
)


def gather_sample_info_across_dp(local_seq_lengths: torch.Tensor):
    """
    Gather per-sample sequence length information across all DP ranks.

    Each DP rank may hold a different number of samples. This function:
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

    dp_group = mpu.get_data_parallel_group_gloo(
        with_context_parallel=False,
        partial_data_parallel=False,
    )
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
    # Step 3: All-gather padded tensors
    # --------------------------------
    gathered_len = [torch.zeros_like(padded_len) for _ in range(dp_size)]
    gathered_idx = [torch.zeros_like(padded_idx) for _ in range(dp_size)]

    dist.all_gather(gathered_len, padded_len, group=dp_group)
    dist.all_gather(gathered_idx, padded_idx, group=dp_group)

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
    pack_len_ratio: float = 1.05,
    max_iter: int = 20,
    swap_tolerance: float = 0.05,
):
    """
    Compute a DP sample reordering *plan* for load balancing.

    This function only generates a reordering / assignment schema.
    It does NOT move tensors or perform communication.

    Intended usage:
        The returned plan is later consumed by an all_to_all-style
        data exchange to physically redistribute samples.

    Objective:
        Balance attention computation load across DP ranks by reassigning
        samples, while keeping per-DP pack length bounded.

    Constraint:
        pack_len(dp) <= avg_pack_len * pack_len_ratio

    Algorithm:
        1) Greedy assignment with LPT (Largest Processing Time first)
        2) Local refinement via move / swap (KL-style heuristic)

    Returns:
        dp_reorder_plan[tgt_dp_rank] = List[(local_sample_index, src_dp_rank)]
    """

    # Estimated accumulated computation cost per DP rank
    dp_group = mpu.get_data_parallel_group_gloo(
        with_context_parallel=False,
        partial_data_parallel=False,
    )
    dp_size = dp_group.size()
    dp_costs = torch.tensor([0.0 for _ in range(dp_size)])

    # --------------------------------------------------
    # 1) Build per-sample items
    # --------------------------------------------------
    # Each item: (estimated_cost, seq_len, local_index, source_dp_rank)
    items = []
    seq_lens = []

    for sample_len, local_idx, src_rank in zip(
        global_sample_lengths, local_sample_index, sample_src_dp_rank
    ):
        seq_len = int(sample_len)
        cost = load_estimate_per_sample(sample_len)

        items.append((float(cost), seq_len, int(local_idx), int(src_rank)))
        seq_lens.append(seq_len)

    # Average pack length as soft capacity constraint
    avg_pack_len = sum(seq_lens) / dp_size
    pack_cap = avg_pack_len * pack_len_ratio

    # Sort samples by descending cost (LPT heuristic)
    items.sort(key=lambda x: x[0], reverse=True)

    # --------------------------------------------------
    # 2) Greedy assignment under pack length constraint
    # --------------------------------------------------
    dp_reorder_plan_with_cost = [[] for _ in range(dp_size)]
    dp_pack_lens = torch.zeros(dp_size, dtype=torch.long)

    for cost, seq_len, local_idx, src_rank in items:
        placed = False

        # Try DP ranks with smallest current cost first
        for r in torch.argsort(dp_costs).tolist():
            if dp_pack_lens[r] + seq_len <= pack_cap:
                dp_reorder_plan_with_cost[r].append(
                    (cost, seq_len, local_idx, src_rank)
                )
                dp_costs[r] += cost
                dp_pack_lens[r] += seq_len
                placed = True
                break

        if not placed:
            # Fallback: assign to DP with smallest pack length
            r = int(dp_pack_lens.argmin())
            dp_reorder_plan_with_cost[r].append((cost, seq_len, local_idx, src_rank))
            dp_costs[r] += cost
            dp_pack_lens[r] += seq_len

    # --------------------------------------------------
    # 3) Local refinement (KL-style move / swap)
    # --------------------------------------------------
    for _ in range(max_iter):
        max_r = int(dp_costs.argmax())
        min_r = int(dp_costs.argmin())

        # Stop if relative imbalance is within tolerance
        if (dp_costs[max_r] - dp_costs[min_r]) / max(
            dp_costs[min_r], 1e-6
        ) < swap_tolerance:
            break

        max_bucket = dp_reorder_plan_with_cost[max_r]
        min_bucket = dp_reorder_plan_with_cost[min_r]
        if not max_bucket or not min_bucket:
            break

        # Candidate samples for refinement
        a = max(max_bucket, key=lambda x: x[0])  # largest in max DP
        b = min(min_bucket, key=lambda x: x[0])  # smallest in min DP

        cost_a, len_a, _, _ = a
        cost_b, len_b, _, _ = b

        old_diff = (dp_costs.max() - dp_costs.min()).item()

        # ---- Try move: a from max_r -> min_r
        if dp_pack_lens[min_r] + len_a <= pack_cap:
            move_costs = dp_costs.clone()
            move_packs = dp_pack_lens.clone()
            move_costs[max_r] -= cost_a
            move_costs[min_r] += cost_a
            move_packs[max_r] -= len_a
            move_packs[min_r] += len_a
            move_diff = (move_costs.max() - move_costs.min()).item()
        else:
            move_diff = float("inf")

        # ---- Try swap: a <-> b
        if (
            dp_pack_lens[max_r] - len_a + len_b <= pack_cap
            and dp_pack_lens[min_r] - len_b + len_a <= pack_cap
        ):
            swap_costs = dp_costs.clone()
            swap_packs = dp_pack_lens.clone()
            swap_costs[max_r] = dp_costs[max_r] - cost_a + cost_b
            swap_costs[min_r] = dp_costs[min_r] - cost_b + cost_a
            swap_packs[max_r] = dp_pack_lens[max_r] - len_a + len_b
            swap_packs[min_r] = dp_pack_lens[min_r] - len_b + len_a
            swap_diff = (swap_costs.max() - swap_costs.min()).item()
        else:
            swap_diff = float("inf")

        # Select best improving operation
        best_op = None
        best_diff = old_diff

        if move_diff < best_diff:
            best_op = ("move", a, move_costs, move_packs)
            best_diff = move_diff

        if swap_diff < best_diff:
            best_op = ("swap", a, b, swap_costs, swap_packs)
            best_diff = swap_diff

        if best_op is None:
            break

        # Apply selected operation
        if best_op[0] == "move":
            _, a_item, new_costs, new_packs = best_op
            max_bucket.remove(a_item)
            min_bucket.append(a_item)
            dp_costs = new_costs
            dp_pack_lens = new_packs
        else:
            _, a_item, b_item, new_costs, new_packs = best_op
            max_bucket.remove(a_item)
            min_bucket.remove(b_item)
            max_bucket.append(b_item)
            min_bucket.append(a_item)
            dp_costs = new_costs
            dp_pack_lens = new_packs

    # --------------------------------------------------
    # 4) Export reorder plan
    # --------------------------------------------------
    # dp_reorder_plan[dp_rank] = List[(local_sample_index, source_dp_rank)]
    dp_reorder_plan = [
        [(local_idx, src_rank) for (_, _, local_idx, src_rank) in bucket]
        for bucket in dp_reorder_plan_with_cost
    ]

    return dp_reorder_plan


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

    dp_group = mpu.get_data_parallel_group_gloo(
        with_context_parallel=False,
        partial_data_parallel=False,
    )
    dp_size = dp_group.size()

    cur_dp_rank = torch.distributed.get_rank(dp_group)

    # tensors_to_send[r]: list of tensors to be sent to DP rank r
    tensors_to_send = [[] for _ in range(dp_size)]

    # send_splits[r]: total number of elements sent to DP rank r
    send_splits = [0 for _ in range(dp_size)]

    # recv_splits[r]: total number of elements expected from DP rank r
    recv_splits = [0 for _ in range(dp_size)]

    # recv_tensor_lengths[r]: per-sample lengths expected from DP rank r
    recv_tensor_lenths = [[] for _ in range(dp_size)]

    # Build send/recv metadata strictly following dp_reorder_plan
    for tgt_dp_rank in range(len(dp_reorder_plan)):
        for local_tensor_idx, src_dp_rank in dp_reorder_plan[tgt_dp_rank]:
            mask = (all_sample_src_dp_rank == src_dp_rank) & (
                all_local_sample_index == local_tensor_idx
            )
            tmp_length = global_seq_lengths[mask].item()

            # Case 1: current rank is the source → send to target rank
            if src_dp_rank == cur_dp_rank:
                send_splits[tgt_dp_rank] += tmp_length
                tensors_to_send[tgt_dp_rank].append(local_tensor_list[local_tensor_idx])

            # Case 2: current rank is the target → receive from source rank
            if tgt_dp_rank == cur_dp_rank:
                recv_splits[src_dp_rank] += tmp_length
                recv_tensor_lenths[src_dp_rank].append(tmp_length)

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

    # Execute DP all-to-all communication
    dist.all_to_all_single(
        recv_tensor,
        send_tensor,
        recv_splits,
        send_splits,
        group=dp_group,
    )

    # Reconstruct per-sample tensors
    recv_tensor_lenths = [x for dp in recv_tensor_lenths for x in dp]
    redistributed_tensor_list = torch.split(recv_tensor, recv_tensor_lenths)
    redistributed_tensor_list = [
        reconstruct_func(t.clone()) for t in redistributed_tensor_list
    ]

    return redistributed_tensor_list


def redistribute_tensors(local_tensor_list, dp_reorder_plan, reconstruct_func):
    """
    redistribute tensors
    """
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
    reorder data for intern_vl
    """
    local_sample_list = depack_data_for_intern_vl(data)
    local_pixel_values = []
    local_input_ids = []
    local_image_flags = []
    local_loss_weights = []
    local_labels = []
    local_loss_mask = []

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

    dp_reorder_plan = solve_sample_dp_reorder_plan(
        global_seq_lengths,
        local_sample_index,
        sample_src_dp_rank,
    )

    redistribute_specs = {
        "input_ids": (local_input_ids, reconstruct_llm_for_internvl),
        "loss_weights": (local_loss_weights, reconstruct_llm_for_internvl),
        "labels": (local_labels, reconstruct_llm_for_internvl),
        "pixel_values": (local_pixel_values, reconstruct_pixel_values_for_internvl),
        "image_flags": (local_image_flags, reconstruct_image_flags_for_internvl),
    }

    redistributed = {
        name: redistribute_tensors(tensor, dp_reorder_plan, reconstruct_fn)
        for name, (tensor, reconstruct_fn) in redistribute_specs.items()
    }

    redistributed_input_ids = redistributed["input_ids"]
    redistributed_loss_weights = redistributed["loss_weights"]
    redistributed_labels = redistributed["labels"]
    redistributed_pixel_values = redistributed["pixel_values"]
    redistributed_image_flags = redistributed["image_flags"]

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
    reorder data for vlm
    """
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

    dp_reorder_plan = solve_sample_dp_reorder_plan(
        global_seq_lengths,
        local_sample_index,
        sample_src_dp_rank,
    )

    redistributed_tokens = redistribute_tensors(
        local_tokens, dp_reorder_plan, reconstruct_llm_for_vlm
    )
    redistributed_labels = redistribute_tensors(
        local_labels, dp_reorder_plan, reconstruct_llm_for_vlm
    )
    redistributed_attn_mask = redistribute_tensors(
        local_attn_mask, dp_reorder_plan, reconstruct_llm_for_vlm
    )

    redistributed_pixel_values_images = None
    redistributed_image_thws = None

    if local_pixel_values_images:
        visual_size = local_pixel_values_images[0].shape[-1]
        redistributed_pixel_values_images = redistribute_tensors(
            local_pixel_values_images,
            dp_reorder_plan,
            partial(reconstruct_visual_for_vlm, visual_size=visual_size),
        )

        redistributed_image_thws = redistribute_tensors(
            local_image_thws,
            dp_reorder_plan,
            reconstruct_visual_grid_thw_for_vlm,
        )

    redistributed_pixel_values_videos = None
    redistributed_video_thws = None
    if local_pixel_values_videos:
        visual_size = local_pixel_values_videos[0].shape[-1]
        redistributed_pixel_values_videos = redistribute_tensors(
            local_pixel_values_videos,
            dp_reorder_plan,
            partial(reconstruct_visual_for_vlm, visual_size=visual_size),
        )

        redistributed_video_thws = redistribute_tensors(
            local_vid_thws,
            dp_reorder_plan,
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
    reorder data
    """
    args = get_args()
    if args.model_family == "intern_vl":
        return data[:-1] + [reorder_data_for_internvl(data[-1])]
    elif args.model_family in constants.VisionLanguageModelFamilies.names():
        return data[:-1] + [reorder_data_for_vlm(data[-1])]
    else:
        raise ValueError(
            f"Unsupported model_family '{args.model_family}'. "
            f"Expected one of: "
            f"['intern_vl', "
            f"{', '.join(constants.VisionLanguageModelFamilies.names())}]"
        )

