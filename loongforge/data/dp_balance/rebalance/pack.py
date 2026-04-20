# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""depack and pack data"""

from typing import Dict, Tuple, List, Any

import torch

from loongforge.utils import get_args, get_tokenizer

# Special token for image context and its token id
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
    attention_mask = attention_mask.squeeze_(0)

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
) -> Tuple[Any, ...]:
    """
    Pack a list of InternVLDataSample objects into model-ready batch tensors.

    Notes:
        - This function ONLY performs local tensor packing.
        - It does NOT execute forward/backward, nor any DP/SP communication.
        - The returned tuple is the final packed batch representation.

    Returns:
        Tuple[Any, ...]: packed batch data used directly by the model.
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

    use_sequence_parallel = get_args().sequence_parallel
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


class VLMDataSample:
    """
    Lightweight container representing a single (possibly multimodal) sample.
    """

    def __init__(
        self,
        tokens,
        labels,
        # position_ids,
        attn_mask,
        # loss_mask,
        pixel_values_images=None,
        image_thw=None,
        pixel_values_videos=None,
        vid_thw=None,
    ):
        self.tokens = tokens
        self.labels = labels
        # self.position_ids = position_ids
        self.attn_mask = attn_mask
        # self.loss_mask = loss_mask
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
    assert (
        data.get("images") is not None or data.get("pixel_values_videos") is not None
    ), (
        "No visual inputs found in batch data. "
        "Expected key 'images' or 'pixel_values_videos', "
        f"but got keys: {list(data.keys())}."
    )

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
        # loss_mask = data["loss_mask"][:, start:end]
        # position_ids = data["position_ids"][:, 0, start:end]

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
            # position_ids=position_ids,
            # loss_mask=loss_mask,
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
    dtype = vlm_samples[0].tokens.dtype

    # -------- text --------
    tokens_list = []
    labels_list = []
    attn_mask_list = []
    # loss_mask_list = []
    # position_ids_list = []

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
        image_grid_thw = torch.stack(image_grid_thw, dim=0).squeeze()
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
        video_grid_thw = torch.stack(video_grid_thw, dim=0).squeeze()
    else:
        video_grid_thw = None

    use_sequence_parallel = get_args().sequence_parallel
    max_lengths = (cu_lengths[1:] - cu_lengths[:-1]).max().view(1)
    cu_lengths = cu_lengths.unsqueeze(0)

    # -------- final batch --------
    new_data = {
        "tokens": tokens,
        "labels": labels,
        "max_lengths": max_lengths,
        "cu_lengths": cu_lengths,
        "attn_mask": attn_mask,
        # "loss_mask": loss_mask,
        # "position_ids": position_ids,
        "imgs": images,
        "pixel_values_videos": pixel_values_videos,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
    }

    return new_data
