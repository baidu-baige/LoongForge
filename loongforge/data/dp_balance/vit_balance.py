# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
ViT encoder data parallel balance.

This module provides functionality for load balancing the Vision Transformer (ViT)
encoder across data parallel ranks in distributed training. It handles sample
redistribution to ensure even workload distribution when processing images.

Main function:
- dp_balance_vit_encoder: Balance ViT encoder workload across DP ranks
"""

import torch
from loongforge.data.dp_balance.rebalance.balance import (
    gather_sample_info_across_dp,
    solve_sample_dp_reorder_plan,
    redistribute_tensors,
    get_reverse_reorder_plan,
    get_dp_group_by_device,
)
from loongforge.train.initialize import change_parallel_state
from loongforge.utils import get_args


def dp_balance_vit_encoder(vit_module, pixel_values, image_grid_thw):
    """
    Balance ViT encoder workload across DP ranks.

    This function performs load balancing for the Vision Transformer encoder by:
    1. Computing the input length for each image based on grid dimensions (THW)
    2. Gathering length information across all DP ranks
    3. Computing optimal redistribution plan
    4. Redistributing pixel_values and image_grid_thw to balance ViT compute load
    5. Running ViT forward pass on redistributed inputs
    6. Redistributing ViT outputs back to original DP ranks

    The redistribution ensures that images with similar compute costs are grouped
    together, reducing load imbalance across DP ranks.

    Args:
        vit_module: The ViT encoder module that takes (pixel_values, image_grid_thw)
            and returns (pixel_embeds, window_index, deepstack_pixel_embeds).
        pixel_values: Input pixel values of shape [total_images, hidden_size].
        image_grid_thw: Grid dimensions [T, H, W] for each image.

    Returns:
        tuple: (pixel_embeds, window_index, deepstack_pixel_embeds)
            - pixel_embeds: ViT output embeddings
            - window_index: Window indices for attention (may be None)
            - deepstack_pixel_embeds: Deep stack feature embeddings (may be empty list)
    """
    dp_group = get_dp_group_by_device(pixel_values)
    dp_size = dp_group.size()

    vit_hidden_size = pixel_values.shape[1]
    vit_input_lengths = (
        image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
    ).tolist()
    split_vit_input = pixel_values.split(vit_input_lengths, dim=0)
    global_vit_lengths, local_vit_index, tensor_src_dp_rank = (
        gather_sample_info_across_dp(
            torch.tensor(vit_input_lengths, dtype=torch.int, device=pixel_values.device)
        )
    )

    vit_dp_reorder_plan, _ = solve_sample_dp_reorder_plan(
        global_vit_lengths,
        local_vit_index,
        tensor_src_dp_rank,
        cost_fn=lambda l: l ** 2,
        pack_len_ratio=get_args().dp_balance_max_len_ratio_vit,
        cross_micro_batch_balance=False,
        caller="ViT",
    )

    # No reorder needed, directly forward
    if vit_dp_reorder_plan is None:
        return vit_module(pixel_values, image_grid_thw)

    # reorder vit input
    pixel_values = redistribute_tensors(
        split_vit_input,
        vit_dp_reorder_plan,
        reconstruct_func=lambda t: t.reshape(-1, vit_hidden_size),
    )
    pixel_values = torch.cat(pixel_values, dim=0)

    # reorder image_grid_thw
    local_thw_list = [thw_row.reshape(-1) for thw_row in image_grid_thw]
    reordered_image_grid_thw = redistribute_tensors(
        local_thw_list,
        vit_dp_reorder_plan,
        reconstruct_func=lambda t: t.reshape(-1, 3),
    )
    reordered_image_grid_thw = torch.cat(reordered_image_grid_thw, dim=0)

    # vit forward
    pixel_embeds, window_index, deepstack_pixel_embeds = vit_module(
        pixel_values, reordered_image_grid_thw
    )
    args = get_args()
    if args.enable_encoder_hetero_dp or args.enable_full_hetero_dp:
        change_parallel_state("image_encoder")

    pixel_embeds_hidden_size = pixel_embeds.shape[-1]

    # reorder vit output
    if pixel_embeds.numel() == 0:
        return pixel_embeds, window_index, deepstack_pixel_embeds
    pixel_embeds_lengths = (
        reordered_image_grid_thw[:, 0]
        * reordered_image_grid_thw[:, 1]
        * reordered_image_grid_thw[:, 2]
    ).tolist()
    reverse_reorder_plan = get_reverse_reorder_plan(vit_dp_reorder_plan, dp_size)
    split_pixel_embeds = pixel_embeds.split(pixel_embeds_lengths, dim=0)
    split_pixel_embeds = redistribute_tensors(
        split_pixel_embeds,
        reverse_reorder_plan,
        reconstruct_func=lambda t: t.reshape(-1, pixel_embeds_hidden_size),
    )
    pixel_embeds = torch.cat(split_pixel_embeds, dim=0)

    # reorder deepstack_pixel_embeds
    if len(deepstack_pixel_embeds) != 0:
        reordered_thw_merged = reordered_image_grid_thw.clone()
        reordered_thw_merged[:, 1:] = (
            reordered_thw_merged[:, 1:] // vit_module.spatial_merge_size
        )
        deepstack_feature_lengths = (
            reordered_thw_merged[:, 0]
            * reordered_thw_merged[:, 1]
            * reordered_thw_merged[:, 2]
        ).tolist()

        for i, deepstack_feature in enumerate(deepstack_pixel_embeds):
            deepstack_feature_split = deepstack_feature.split(
                deepstack_feature_lengths, dim=0
            )

            deepstack_feature_split = redistribute_tensors(
                deepstack_feature_split,
                reverse_reorder_plan,
                reconstruct_func=lambda t: t.reshape(-1, deepstack_feature.shape[-1]),
            )
            deepstack_pixel_embeds[i] = torch.cat(deepstack_feature_split, dim=0)

    # regenerate window_index
    if window_index is not None:
        window_index = vit_module.get_window_index(image_grid_thw)
    if args.enable_encoder_hetero_dp or args.enable_full_hetero_dp:
        change_parallel_state("text_decoder")
    return pixel_embeds, window_index, deepstack_pixel_embeds
