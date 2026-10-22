# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""depack and pack data"""

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
