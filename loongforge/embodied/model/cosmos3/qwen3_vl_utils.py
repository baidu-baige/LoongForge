# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Qwen3-VL utility functions."""
# Core masking functions extracted from transformers.masking_utils for BAGEL compatibility
# Original Copyright 2025 HuggingFace Inc. team. Licensed under the Apache License, Version 2.0

from typing import Any, Callable, ClassVar, Optional, cast

import torch
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging

# from transformers.utils.generic import GeneralInterface
from transformers.utils.import_utils import is_torch_greater_or_equal

logger = logging.get_logger(__name__)

_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)

_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."
_SYSTEM_PROMPT_TRANSFER = (
    "You are a helpful assistant that generates images or videos following the user's instructions"
    " and control signals (edge maps, blur, depth, or segmentation)."
)
_SYSTEM_PROMPT_IMAGE_EDITING = "You are a helpful assistant who will edit images based on the user's instructions."


def tokenize_caption(
    caption: str,
    tokenizer: PreTrainedTokenizerBase,
    is_video: bool = False,
    use_system_prompt: bool = False,
    system_prompt: Optional[str] = None,
) -> list[int]:
    """Tokenize a text caption into token IDs using the Qwen2 chat template.

    Wraps the caption in a chat-style conversation (with a "user" role) and applies
    the tokenizer's chat template to produce the final token ID sequence, including
    any special tokens (e.g., BOS, role markers, generation prompt).

    Args:
        caption: The text caption to tokenize.
        tokenizer: A HuggingFace ``PreTrainedTokenizerBase`` (e.g. Qwen2Tokenizer or Fast tokenizer).
        is_video: If True (and use_system_prompt=True), uses the video system prompt;
            otherwise uses the image system prompt. Ignored when ``system_prompt`` is
            provided.
        use_system_prompt: If True, prepends a system prompt message to the conversation
            before the user caption. Ignored when ``system_prompt`` is provided.
        system_prompt: When supplied, this exact string is used as the system prompt,
            overriding both ``is_video`` and ``use_system_prompt``.

    Returns:
        List of token IDs representing the full chat-formatted caption.
    """
    conversations = []
    if system_prompt is not None:
        conversations.append({"role": "system", "content": system_prompt})
    elif use_system_prompt:
        _system_prompt = _SYSTEM_PROMPT_VIDEO if is_video else _SYSTEM_PROMPT_IMAGE
        conversations.append({"role": "system", "content": _system_prompt})
    conversations.append({"role": "user", "content": caption})

    tokenizer_output = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
        return_dict=False,
    )
    return cast(list[int], tokenizer_output)


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def sliding_window_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a sliding window pattern. Add it on top of a causal mask for a proper sliding
    window mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """Check if kv position is within the sliding window."""
        return kv_idx > q_idx - sliding_window

    return inner_mask


def and_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        """Compute the intersection of all mask functions."""
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask


def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a sliding window mask.
    """
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)


def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """Apply padding mask for the given batch and kv position."""
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def _vmap_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
    """
    Used to vmap our mask_functions over the q_idx and kv_idx dimensions of the inputs.
    """
    # We vmap the function 2 times, broadcasting the [q_idx, kv_idx] dimensions
    dimensions = [(None, None, None, 0), (None, None, 0, None)]
    if bh_indices:
        # We extend broadcasting over the [batch_idx, head_idx] dimensions
        dimensions.extend([(None, 0, None, None), (0, None, None, None)])

    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function


def prepare_padding_mask(
    attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int, _slice: bool = True
) -> Optional[torch.Tensor]:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it, and slicing
    according to the `kv_offset` if `_slice` is `True`.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        # Pad it if necessary
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
        # For flex, we should not slice them, only use an offset
        if _slice:
            # Equivalent to: `local_padding_mask = attention_mask[:, kv_offset : kv_offset + kv_length]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indices = torch.arange(kv_length, device=local_padding_mask.device)
            mask_indices += kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]
    return local_padding_mask


def eager_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    """
    Create a 4D float mask of shape `(batch_size, 1, query_length, kv_length)` where a value of 0 indicates that
    the element should take part in the attention computation, and -inf (minimum value for the given `dtype`) that
    it should not.
    """
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset

    # Create the 4D mask easily
    causal_mask = _vmap_for_bhqkv(mask_function, bh_indices=False)(
        None, None, cache_position, kv_arange
    )  # [q_len,kv_length]
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)  # [B,1,q_len,kv_length]
    if padding_mask is not None:
        causal_mask = causal_mask * padding_mask[:, None, None, :]  # [B,1,q_len,kv_length]

    min_dtype = torch.finfo(dtype).min
    # we need 0s where the tokens should be taken into account, and -inf otherwise
    mask = torch.where(
        causal_mask, torch.tensor(0.0, device=causal_mask.device, dtype=dtype), min_dtype
    )  # [B,1,q_len,kv_length]
    return mask


# class AttentionMaskInterface(GeneralInterface):
class AttentionMaskInterface:
    """Attention mask interface registry."""

    # Class instance object for mask interfaces
    _global_mapping: ClassVar = {
        "eager": eager_mask,
    }


# Global AttentionMaskInterface shared by all models
ALL_MASK_ATTENTION_FUNCTIONS: AttentionMaskInterface = AttentionMaskInterface()


def _preprocess_mask_arguments(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    layer_idx: Optional[int],
) -> tuple[bool, Optional[torch.Tensor], None, int, int]:
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code.
    """
    # If the mask is already 4D, simply return as-is
    if isinstance(attention_mask, torch.Tensor) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None

    # For TGI/vLLM backends or other custom attention: we don't need a mask
    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)

    # If using a cache, it can give all information about mask sizes based on seen tokens
    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    # Otherwise, the sizes are simply the input sizes
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    return False, attention_mask, None, kv_length, kv_offset


def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config).
    """
    # For hybrid cache structure, use the full_attention layers
    layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Potentially add the padding 2D mask
    if attention_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, padding_mask_function(attention_mask))

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        dtype=dtype,
        config=config,
    )
    return causal_mask


def get_rope_index(
    model: Any,
    input_ids: Optional[torch.Tensor] = None,  # [B,N]
    image_grid_thw: Optional[torch.Tensor] = None,  # [num_images,3]
    video_grid_thw: Optional[torch.Tensor] = None,  # [num_videos,3]
    attention_mask: Optional[torch.Tensor] = None,  # [B,N]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Qwen3-VL multimodal RoPE positions and deltas."""
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)  # [sum_T,3]
        video_grid_thw[:, 0] = 1

    spatial_merge_size = model.config.vision_config.spatial_merge_size
    image_token_id = model.config.image_token_id
    video_token_id = model.config.video_token_id
    vision_start_token_id = model.config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)  # [B,N]
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )  # [3,B,N]
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)  # [B,N]
        for i, sample_input_ids in enumerate(total_input_ids):
            sample_input_ids = sample_input_ids[attention_mask[i] == 1]  # [N_unmasked]
            vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)  # [N_media]
            vision_tokens = sample_input_ids[vision_start_indices + 1]  # [N_media]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = sample_input_ids.tolist()
            llm_pos_ids_list: list[torch.Tensor] = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)  # [3,text_len]

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()  # [T*H*W]
                h_index = (
                    torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                )  # [T*H*W]
                w_index = (
                    torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                )  # [T*H*W]
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)  # [3,T*H*W]
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)  # [3,text_len]

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)  # [3,N_unmasked]
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)  # [B,1]
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1  # [B,N]
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)  # [3,B,N]
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]  # [B,1]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]  # [B,1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        )  # [3,B,N]
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )  # [B,1]

    return position_ids, mrope_position_deltas


def get_image_features(
    model: Any,
    pixel_values: torch.Tensor,  # [N_patches,C,H,W]
    image_grid_thw: Optional[torch.Tensor] = None,  # [num_images,3]
) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]:
    """Encode images with the Qwen3-VL visual module."""
    pixel_values = pixel_values.type(model.visual.dtype)
    image_embeds, deepstack_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
    split_sizes = (image_grid_thw.prod(-1) // model.visual.spatial_merge_size**2).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    return image_embeds, deepstack_image_embeds


def get_placeholder_mask(
    model: Any,
    input_ids: Optional[torch.Tensor],  # [B,N]
    inputs_embeds: torch.Tensor,  # [B,N,hidden_size]
    image_features: Optional[torch.Tensor] = None,  # [N_image_tokens,hidden_size]
    video_features: Optional[torch.Tensor] = None,  # [N_video_tokens,hidden_size]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return expanded image/video placeholder masks and validate feature lengths."""
    if input_ids is None:
        special_image_mask = inputs_embeds == model.get_input_embeddings()(
            torch.tensor(model.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_image_mask = special_image_mask.all(-1)  # [B,N]
        special_video_mask = inputs_embeds == model.get_input_embeddings()(
            torch.tensor(model.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_video_mask = special_video_mask.all(-1)  # [B,N]
    else:
        special_image_mask = input_ids == model.config.image_token_id  # [B,N]
        special_video_mask = input_ids == model.config.video_token_id  # [B,N]

    n_image_tokens = special_image_mask.sum()
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
        raise ValueError(
            f"Image features and image tokens do not match: "
            f"tokens: {n_image_tokens}, features {image_features.shape[0]}"
        )

    n_video_tokens = special_video_mask.sum()
    special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
        raise ValueError(
            f"Videos features and video tokens do not match: "
            f"tokens: {n_video_tokens}, features {video_features.shape[0]}"
        )

    return special_image_mask, special_video_mask
