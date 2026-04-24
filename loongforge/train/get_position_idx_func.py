"""Get position index function"""
import torch
from loongforge.utils import get_model_config
from typing import Tuple


def get_rope_index_qwen3vl(batch_data):
    """Different from the original implementation, Qwen3VLMoe use timestamps rather than absolute time position ids."""
    
    model_config = get_model_config()
    spatial_merge_size = 2
    mrope_position_deltas = []
    input_ids = batch_data.get("tokens", None)
    image_grid_thw = batch_data.get("image_grid_thw", None)
    video_grid_thw = batch_data.get("video_grid_thw", None)
    attention_mask = batch_data.get("attn_mask", None)
    VISION_START_TOKEN_ID = getattr(
        getattr(model_config, "image_encoder", None), 
        "vision_start_token_id", 
        151652
    )
    IMAGE_TOKEN_ID = getattr(
        getattr(model_config, "image_encoder", None), 
        "image_token_id", 
        151655
    )
    VIDEO_TOKEN_ID = getattr(
        getattr(model_config, "image_encoder", None),
        "video_token_id",
        151656
    )

    # Since we use timestamps to seperate videos, 
    # like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>,
    # the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1        

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 0] # 1
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == VISION_START_TOKEN_ID
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == IMAGE_TOKEN_ID).sum()
            video_nums = (vision_tokens == VIDEO_TOKEN_ID).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if IMAGE_TOKEN_ID in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(IMAGE_TOKEN_ID, st)
                else:
                    ed_image = len(input_tokens) + 1
                if VIDEO_TOKEN_ID in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(VIDEO_TOKEN_ID, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # t_index is always 0 because llm_grid_t is always 1 
                # (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 0] = llm_positions.to(position_ids.device) # 1
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 1, 1) # 0
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas
        

def get_rope_index_internvl(batch_data):
    """Build position ids based on the input tokens."""
    attention_mask = batch_data.get("attn_mask", None)
    x = batch_data.get("tokens", None)
    model_config = get_model_config()
    VISION_TOKEN_TYPE = model_config.get("vision_token_type", 1)
    LANGUAGE_TOKEN_TYPE = model_config.get("language_token_type", 0)
    if attention_mask is not None:
        tmp = x.clone()
        tmp[~(attention_mask.bool())] = -1
    else:
        tmp = x.clone()
    # image boi eoi token as LANGUAGE_TOKEN_TYPE
    is_boi_eoi = torch.zeros_like(x, dtype=torch.bool)
    is_boi_eoi[1:] |= (tmp[1:] == VISION_TOKEN_TYPE) & (tmp[:-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[0] |= tmp[0] == VISION_TOKEN_TYPE
    is_boi_eoi[:-1] |= (tmp[:-1] == VISION_TOKEN_TYPE) & (
        tmp[1:] == LANGUAGE_TOKEN_TYPE
    )
    is_boi_eoi[-1] |= tmp[-1] == VISION_TOKEN_TYPE
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
    # final position ids
    y = torch.zeros_like(x, dtype=torch.long)
    y[1:] = (tmp[1:] == LANGUAGE_TOKEN_TYPE) | (
        (tmp[1:] == VISION_TOKEN_TYPE) & (tmp[:-1] == LANGUAGE_TOKEN_TYPE)
    )
    y = y.cumsum(dim=-1)
    return y, None


def get_mrope_index(batch_data) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slightly modified from Qwen2_5VLForConditionalGeneration.get_rope_index"""
    model_config = get_model_config()
    spatial_merge_size = 2
    mrope_position_deltas = []
    input_ids = batch_data.get("tokens", None)
    image_grid_thw = batch_data.get("image_grid_thw", None)
    video_grid_thw = batch_data.get("video_grid_thw", None)
    attention_mask = batch_data.get("attn_mask", None)
    second_per_grid_ts = batch_data.get("second_per_grid_ts", None)
    VISION_START_TOKEN_ID = getattr(model_config, "vision_start_token_id", 151652)
    IMAGE_TOKEN_ID = getattr(model_config, "vision_token_id", 151655)
    VIDEO_TOKEN_ID = getattr(model_config, "video_token_id", 151656)
    assert input_ids is not None

    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 0]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == VISION_START_TOKEN_ID
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == IMAGE_TOKEN_ID).sum()
            video_nums = (vision_tokens == VIDEO_TOKEN_ID).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if IMAGE_TOKEN_ID in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(IMAGE_TOKEN_ID, st)
                else:
                    ed_image = len(input_tokens) + 1
                if VIDEO_TOKEN_ID in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(VIDEO_TOKEN_ID, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * 2

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 0] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.logical_not().long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 1, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def get_position_ids(batch_data):
    """Build position ids based on the input tokens."""
    attention_mask = batch_data.get("attn_mask", None).logical_not()
    assert (
        attention_mask is not None
    ), "attention_mask is required for rope position ids"
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids, None
