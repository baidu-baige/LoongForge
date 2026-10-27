# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""InternVL packed-batch layout for DP load balancing."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from loongforge.engines.megatron import get_args, get_tokenizer
from loongforge.engines.megatron.distributed.dp_balance.rebalance.layout import (
    BatchLayout,
)

# Stable protocol token used by the InternVL tokenizer. Kept local so the
# engine package does not import a model implementation.
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class InternVLDataSample(object):
    """
    Lightweight container representing a single (possibly multimodal) sample.
    """

    def __init__(
        self,
        pixel_values=None,
        input_ids=None,
        image_flags=None,
        loss_weights=None,
        labels=None,
    ):
        """
        Initialize a InternVLDataSample.

        All fields are optional to support text-only, image-only,
        or multimodal samples.
        """
        self.pixel_values = pixel_values
        self.input_ids = input_ids
        self.loss_weights = loss_weights
        self.labels = labels
        self.image_flags = image_flags


def depack_data_for_intern_vl(
    data: Dict[str, torch.Tensor]
) -> List[InternVLDataSample]:
    """
    De-pack a packed batch into a list of InternVLDataSample objects.

    Assumptions:
        - attention_mask encodes cumulative token offsets
        - IMG_CONTEXT_TOKEN marks image-token regions
        - image tokens map to pixel_values in fixed chunks (heuristic-based)

    Note:
        No communication or reordering across DP ranks happens here.
    """
    data_sample_list: List[InternVLDataSample] = []

    # Optional multimodal fields
    image_flags = data["image_flags"]
    pixel_values = data["imgs"]

    # Required fields
    input_ids = data["tokens"]
    labels = data["labels"]
    loss_weights = data["loss_weight"]
    attention_mask = data["attn_mask"]

    # Normalize loss_weights type
    if isinstance(loss_weights, list):
        loss_weights = torch.tensor(loss_weights, dtype=torch.float32)

    # attention_mask stores cumulative offsets, shape [N+1]
    attention_mask = attention_mask.squeeze(0)

    # ------------------------------------------------
    # 1) Filter out empty image entries
    # ------------------------------------------------
    filtered_image_flags = []
    filtered_pixel_values = []

    if image_flags is not None:
        for i in range(image_flags.shape[0]):
            if image_flags[i].numel() != 0 and int(image_flags[i].sum().item()) != 0:
                filtered_image_flags.append(image_flags[i])
                if pixel_values is not None:
                    filtered_pixel_values.append(pixel_values[i])

    if filtered_image_flags:
        filtered_image_flags = torch.stack(filtered_image_flags, dim=0)
        filtered_pixel_values = (
            torch.stack(filtered_pixel_values, dim=0) if filtered_pixel_values else None
        )
    else:
        filtered_image_flags = None
        filtered_pixel_values = None

    # ------------------------------------------------
    # 2) Slice packed tensors into per-sample segments
    # ------------------------------------------------
    img_size = get_args().force_image_size
    img_context_token_id = get_tokenizer().tokenizer.convert_tokens_to_ids(
        IMG_CONTEXT_TOKEN
    )

    for j in range(len(attention_mask) - 1):
        start_idx = int(attention_mask[j].item())
        end_idx = int(attention_mask[j + 1].item())

        # Token-level slicing
        cur_input_ids = input_ids[:, start_idx:end_idx].clone()
        cur_labels = labels[:, start_idx:end_idx].clone()
        cur_loss_weights = loss_weights[:, start_idx:end_idx].clone()

        # Default: placeholder image tensors
        # Last segment is padding-only and should not carry images
        cur_image_flags = torch.zeros(1, dtype=torch.int64, device=cur_input_ids.device)
        cur_pixel_values = torch.zeros(
            [1, 3, img_size, img_size],
            dtype=torch.float,
            device=cur_input_ids.device,
        )

        # ------------------------------------------------
        # 3) Assign pixel_values based on image tokens
        # ------------------------------------------------
        if filtered_pixel_values is not None:
            image_token_num = int((cur_input_ids == img_context_token_id).sum().item())

            if image_token_num > 0:
                # Heuristic: one image corresponds to ~256 image tokens
                num_imgs = max(1, image_token_num // 256)

                if filtered_pixel_values.shape[0] >= num_imgs:
                    cur_pixel_values = filtered_pixel_values[:num_imgs]
                    cur_image_flags = filtered_image_flags[:num_imgs]

                    # Consume assigned image entries
                    filtered_pixel_values = filtered_pixel_values[num_imgs:]
                    filtered_image_flags = filtered_image_flags[num_imgs:]

        # Construct per-sample container
        data_sample_list.append(
            InternVLDataSample(
                pixel_values=cur_pixel_values,
                input_ids=cur_input_ids,
                image_flags=cur_image_flags,
                labels=cur_labels,
                loss_weights=cur_loss_weights,
            )
        )
    return data_sample_list


def pack_data_for_intern_vl(
    data_sample_list: List[InternVLDataSample],
) -> Dict[str, Any]:
    """
    Pack InternVL samples into a model-ready batch dictionary.

    Notes:
        - This function ONLY performs local tensor packing.
        - It does NOT execute forward/backward, nor any DP/SP communication.
    Returns:
        Dict[str, Any]: packed batch data used directly by the model.
    """
    pixel_values_list = []
    image_flags_list = []
    input_ids_list = []
    labels_list = []
    loss_weights_list = []
    attention_offsets = [0]
    position_ids = []

    for s in data_sample_list:
        if s.pixel_values is not None:
            pixel_values_list.append(s.pixel_values)
        if s.image_flags is not None:
            image_flags_list.append(s.image_flags)
        input_ids_list.append(s.input_ids)
        labels_list.append(s.labels)
        loss_weights_list.append(s.loss_weights)
        attention_offsets.append(attention_offsets[-1] + s.input_ids.shape[-1])
        position_ids.extend(list(range(s.input_ids.shape[-1])))

    pixel_values = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
    image_flags = torch.cat(image_flags_list, dim=0) if image_flags_list else None

    input_ids = (
        torch.cat(input_ids_list, dim=1) if input_ids_list else torch.empty((0, 0))
    )

    labels = torch.cat(labels_list, dim=1) if labels_list else torch.empty((0, 0))

    loss_weights = (
        torch.cat(loss_weights_list, dim=1)
        if loss_weights_list
        else torch.empty((0, 0))
    )

    position_ids = torch.tensor(
        position_ids,
        dtype=torch.int64,
        device=input_ids.device if input_ids.numel() else torch.device("cpu"),
    )

    attention_mask = torch.tensor(
        attention_offsets,
        dtype=torch.int32,
        device=input_ids.device if input_ids.numel() else torch.device("cpu"),
    )

    attention_mask.unsqueeze_(0)

    batch_data = {
        "imgs": pixel_values,
        "position_ids": position_ids,
        "tokens": input_ids,
        "image_flags": image_flags,
        "attn_mask": attention_mask,
        "labels": labels,
        "loss_weight": loss_weights,
    }
    return batch_data


def reorder_data_for_internvl(data, ctx):
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

    dp_reorder_plan = ctx.plan(local_input_ids_lengths)

    if dp_reorder_plan is None:
        return data

    # LLM tensors (input_ids, labels, loss_weights) share the same sample
    # dimensions, gather once and reuse to avoid redundant all_gather calls.
    llm_gathered = ctx.gather(local_input_ids)
    redistributed_input_ids = ctx.redistribute(
        local_input_ids, dp_reorder_plan, llm_gathered, _reconstruct_llm,
    )
    redistributed_labels = ctx.redistribute(
        local_labels, dp_reorder_plan, llm_gathered, _reconstruct_llm,
    )
    redistributed_loss_weights = ctx.redistribute(
        local_loss_weights, dp_reorder_plan, llm_gathered, _reconstruct_llm,
    )

    # pixel_values and image_flags have different element counts, gather separately
    redistributed_pixel_values = ctx.redistribute_grouped(
        local_pixel_values, dp_reorder_plan, _reconstruct_pixel_values,
    )
    redistributed_image_flags = ctx.redistribute_grouped(
        local_image_flags, dp_reorder_plan, _reconstruct_image_flags,
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


def _reconstruct_llm(tensor):
    """Reshape packed LLM tensors to (1, seq_len) for InternVL."""
    return tensor.reshape(1, -1)


def _reconstruct_image_flags(tensor):
    """Image flag metadata; no reconstruction needed."""
    return tensor


def _reconstruct_pixel_values(tensor):
    """Reshape flattened pixels to (N, 3, H, W) using force_image_size."""
    img_size = get_args().force_image_size
    return tensor.reshape(-1, 3, img_size, img_size)


def _internvl_cumulative_lengths(micro_batch):
    return micro_batch["attn_mask"]


INTERNVL_BATCH_LAYOUT = BatchLayout(
    reorder=reorder_data_for_internvl,
    cumulative_lengths=_internvl_cumulative_lengths,
    caller="InternVL",
)
