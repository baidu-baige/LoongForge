# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLM packed-batch layout for DP load balancing.

Depacks a packed VLM batch into samples, redistributes them with the context the
engine passes in, and re-packs them. ``training/vlm`` installs it with
``engines.megatron.dp_balance.rebalance.layout.install_batch_layout``.
"""

from functools import partial

import torch


class VLMDataSample:
    """
    Lightweight container representing a single (possibly multimodal) sample.
    """

    def __init__(
        self,
        tokens,
        labels,
        attn_mask,
        pixel_values_images=None,
        image_thw=None,
        pixel_values_videos=None,
        vid_thw=None,
    ):
        self.tokens = tokens
        self.labels = labels
        self.attn_mask = attn_mask
        self.pixel_values_images = pixel_values_images
        self.image_thw = image_thw
        self.pixel_values_videos = pixel_values_videos
        self.vid_thw = vid_thw


def depack_data_for_vlm(data):
    """
    Depack a packed VLM batch into per-sample VLMDataSample list.

    Assumptions:
      - tokens / labels / masks are packed along seq dim
      - images / videos are concatenated in sample order
      - cu_lengths defines text boundaries
    """
    # -------- text packing info --------
    cu_lengths = data["cu_lengths"][0].tolist()
    sample_num = len(cu_lengths) - 1

    # -------- visual cursors (IMPORTANT: no in-place modification) --------
    images = data["imgs"]
    videos = data["pixel_values_videos"]

    has_image = images is not None and images.numel() > 1
    has_video = videos is not None and videos.numel() > 1

    image_cursor = 0
    video_cursor = 0

    vlm_samples = []

    for i in range(sample_num):
        start = cu_lengths[i]
        end = cu_lengths[i + 1]

        # -------- text --------
        tokens = data["tokens"][:, start:end]
        labels = data["labels"][:, start:end]
        attn_mask = data["attn_mask"][:, start:end]

        # -------- image --------
        pixel_values_images = None
        image_thw = None
        if has_image:
            image_thw = data["image_grid_thw"][i]
            img_t, img_h, img_w = image_thw
            img_len = int(img_t * img_h * img_w)

            assert images.shape[0] >= image_cursor + img_len, (
                f"Image tokens not enough for sample {i}: "
                f"need {img_len}, "
                f"got {images.shape[0] - image_cursor}"
            )

            pixel_values_images = images[image_cursor : image_cursor + img_len]
            image_cursor += img_len

        # -------- video --------
        pixel_values_videos = None
        vid_thw = None
        if has_video:
            vid_thw = data["video_grid_thw"][i]
            vid_t, vid_h, vid_w = vid_thw
            vid_len = int(vid_t * vid_h * vid_w)

            pixel_values_videos = videos[video_cursor : video_cursor + vid_len]
            video_cursor += vid_len

        # -------- build sample --------
        vlm_sample = VLMDataSample(
            tokens=tokens,
            labels=labels,
            attn_mask=attn_mask,
            pixel_values_images=pixel_values_images,
            image_thw=image_thw,
            pixel_values_videos=pixel_values_videos,
            vid_thw=vid_thw,
        )

        vlm_samples.append(vlm_sample)
    return vlm_samples


def pack_data_for_vlm(vlm_samples):
    """
    Re-pack a list of VLMDataSample into a packed batch dict.

    Returns:
        data (dict): compatible with original packed VLM input
    """

    assert len(vlm_samples) > 0, "Empty vlm_samples"

    device = vlm_samples[0].tokens.device

    # -------- text --------
    tokens_list = []
    labels_list = []
    attn_mask_list = []

    cu_lengths = [0]
    cur_len = 0

    for sample in vlm_samples:
        seq_len = sample.tokens.shape[1]

        tokens_list.append(sample.tokens)
        labels_list.append(sample.labels)
        attn_mask_list.append(sample.attn_mask)

        cur_len += seq_len
        cu_lengths.append(cur_len)

    tokens = torch.cat(tokens_list, dim=1)
    labels = torch.cat(labels_list, dim=1)
    attn_mask = torch.cat(attn_mask_list, dim=1)

    cu_lengths = torch.tensor(cu_lengths, device=device, dtype=torch.int32)

    # -------- images --------
    images_list = []
    image_grid_thw = []

    for sample in vlm_samples:
        if sample.pixel_values_images is not None:
            images_list.append(sample.pixel_values_images)
            image_grid_thw.append(sample.image_thw)

    images = None

    if len(images_list) > 0:
        images = torch.cat(images_list, dim=0)
        image_grid_thw = torch.stack(image_grid_thw, dim=0).reshape(-1, 3)
    else:
        image_grid_thw = None

    # -------- videos --------
    videos_list = []
    video_grid_thw = []

    for sample in vlm_samples:
        if sample.pixel_values_videos is not None:
            videos_list.append(sample.pixel_values_videos)
            video_grid_thw.append(sample.vid_thw)

    pixel_values_videos = None
    if len(videos_list) > 0:
        pixel_values_videos = torch.cat(videos_list, dim=0)
        video_grid_thw = torch.stack(video_grid_thw, dim=0).reshape(-1, 3)
    else:
        video_grid_thw = None

    max_lengths = (cu_lengths[1:] - cu_lengths[:-1]).max().view(1)
    cu_lengths = cu_lengths.unsqueeze(0)

    # -------- final batch --------
    new_data = {
        "tokens": tokens,
        "labels": labels,
        "max_lengths": max_lengths,
        "cu_lengths": cu_lengths,
        "attn_mask": attn_mask,
        "imgs": images,
        "pixel_values_videos": pixel_values_videos,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
    }

    return new_data


def reconstruct_llm_for_vlm(tensor):
    """Reshape packed LLM tensors to (1, seq_len) for VLM."""
    return tensor.reshape(1, -1)


def reconstruct_visual_grid_thw_for_vlm(tensor):
    """Reshape visual grid metadata to (num_visuals, 3)."""
    return tensor.reshape(-1, 3)


def reconstruct_visual_for_vlm(tensor, visual_size):
    """Reshape flattened visual embeddings to (num_tokens, visual_size)."""
    return tensor.reshape(-1, visual_size)


def reorder_data_for_vlm(data, ctx):
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

    dp_reorder_plan = ctx.plan(local_token_lengths)

    if dp_reorder_plan is None:
        return data

    # LLM tensors share the same sample count, gather once and reuse
    llm_gathered = ctx.gather(local_tokens)
    redistributed_tokens = ctx.redistribute(
        local_tokens, dp_reorder_plan, llm_gathered, reconstruct_llm_for_vlm,
    )
    redistributed_labels = ctx.redistribute(
        local_labels, dp_reorder_plan, llm_gathered, reconstruct_llm_for_vlm,
    )
    redistributed_attn_mask = ctx.redistribute(
        local_attn_mask, dp_reorder_plan, llm_gathered, reconstruct_llm_for_vlm,
    )

    redistributed_pixel_values_images = None
    redistributed_image_thws = None

    if local_pixel_values_images:
        visual_size = local_pixel_values_images[0].shape[-1]
        redistributed_pixel_values_images = ctx.redistribute_grouped(
            local_pixel_values_images,
            dp_reorder_plan,
            partial(reconstruct_visual_for_vlm, visual_size=visual_size),
        )
        redistributed_image_thws = ctx.redistribute_grouped(
            local_image_thws,
            dp_reorder_plan,
            reconstruct_visual_grid_thw_for_vlm,
        )

    redistributed_pixel_values_videos = None
    redistributed_video_thws = None
    if local_pixel_values_videos:
        visual_size = local_pixel_values_videos[0].shape[-1]
        redistributed_pixel_values_videos = ctx.redistribute_grouped(
            local_pixel_values_videos,
            dp_reorder_plan,
            partial(reconstruct_visual_for_vlm, visual_size=visual_size),
        )
        redistributed_video_thws = ctx.redistribute_grouped(
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


def vlm_cumulative_lengths(micro_batch):
    return micro_batch["cu_lengths"]
