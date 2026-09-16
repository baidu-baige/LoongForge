# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X under the Apache-2.0 License.

"""Small Wall-X LeRobot preprocessing subset used by Wall-OSS-0.5."""

from __future__ import annotations

import json
import logging
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import BatchFeature

logger = logging.getLogger(__name__)


@dataclass
class NormStats:
    """NormStats."""
    min: torch.Tensor
    max: torch.Tensor
    delta: torch.Tensor


def load_norm_stats(norm_stats_path: str, key_mappings: Dict[str, Any]):
    """Load norm stats."""
    with open(norm_stats_path, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    action_key = key_mappings["action"]
    state_key = key_mappings["state"]

    action_q01 = torch.tensor(norm_stats["norm_stats"][action_key]["q01"])
    action_q99 = torch.tensor(norm_stats["norm_stats"][action_key]["q99"])
    state_q01 = torch.tensor(norm_stats["norm_stats"][state_key]["q01"])
    state_q99 = torch.tensor(norm_stats["norm_stats"][state_key]["q99"])
    return {
        "action": NormStats(
            min=action_q01,
            max=action_q99,
            delta=action_q99 - action_q01,
        ),
        "state": NormStats(
            min=state_q01,
            max=state_q99,
            delta=state_q99 - state_q01,
        ),
    }


def get_frame_instruction(
    instruction_info: Dict[str, Any],
    frame_idx: Optional[int] = None,
    truncate_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Optional[int]]:
    """Get frame instruction."""
    if truncate_keys is None:
        truncate_keys = [
            "subtask_generation",
            "distribute",
            "subtask_generation_zh",
            "distribute_zh",
        ]
    instruction_for_frame = {}
    split_end = None
    for key, value in instruction_info.items():
        if isinstance(value, dict):
            for frame_range, frame_instruction in value.items():
                start_frame, end_frame = map(int, frame_range.split(" "))
                if start_frame <= frame_idx < end_frame or start_frame == frame_idx:
                    instruction_for_frame[key] = frame_instruction
                    if truncate_keys is not None and split_end is None and key in truncate_keys:
                        split_end = end_frame + 1
                    break
        else:
            instruction_for_frame[key] = value
    return instruction_for_frame, split_end


def get_task_instruction(
    frame_instruction_info: Dict[str, Any],
    priority_order: Optional[OrderedDict] = None,
) -> str:
    """Get task instruction."""
    default_priority_order = OrderedDict(
        {
            "subtask_generation": 0.25,
            "subtask_generation_zh": 0.25,
            "distribute": 0.25,
            "distribute_zh": 0.25,
        }
    )
    priority_order = OrderedDict(priority_order) if priority_order is not None else default_priority_order
    got_instruction = False
    task_instruction = ""
    for key, prob in priority_order.items():
        if key in frame_instruction_info and frame_instruction_info[key] != "":
            if got_instruction and random.random() >= prob:
                continue
            task_instruction += f"\n{frame_instruction_info[key]}"
            got_instruction = True
            break
    if not got_instruction:
        task_instruction = frame_instruction_info.get("instruction", "")
    return task_instruction


def process_grounding_points(
    text: str,
    orig_height: int,
    orig_width: int,
    resized_height: int,
    resized_width: int,
    model_type: str,
) -> str:
    """Process grounding points."""
    point_pattern = re.compile(r"<point>(.*?)</point>")

    def process_match(match):
        """Process match."""
        coords_str = match.group(1)
        try:
            coords = list(map(int, re.findall(r"\d+", coords_str)))
            scale_w = resized_width / orig_width
            scale_h = resized_height / orig_height
            if len(coords) == 2:
                x, y = coords
                if model_type == "qwen2_5":
                    coords = [
                        max(0, min(round(x * scale_w), resized_width - 1)),
                        max(0, min(round(y * scale_h), resized_height - 1)),
                    ]
                else:
                    coords = [
                        max(0, min(999.999, (x / orig_width) * 1000)),
                        max(0, min(999.999, (y / orig_height) * 1000)),
                    ]
            elif len(coords) == 4:
                x1, y1, x2, y2 = coords
                if model_type == "qwen2_5":
                    coords = [
                        max(0, min(round(x1 * scale_w), resized_width - 1)),
                        max(0, min(round(y1 * scale_h), resized_height - 1)),
                        max(0, min(round(x2 * scale_w), resized_width - 1)),
                        max(0, min(round(y2 * scale_h), resized_height - 1)),
                    ]
                else:
                    coords = [
                        max(0, min(999.999, (x1 / orig_width) * 1000)),
                        max(0, min(999.999, (y1 / orig_height) * 1000)),
                        max(0, min(999.999, (x2 / orig_width) * 1000)),
                        max(0, min(999.999, (y2 / orig_height) * 1000)),
                    ]
            return f'<point>[{", ".join(map(str, coords))}]</point>'
        except (ValueError, TypeError):
            return match.group(0)

    return point_pattern.sub(process_match, text)


def get_wallx_normal_text(
    instruction_info: Dict[str, Any],
    action_chunk_size: int,
    frame_idx: int,
    priority_order: Optional[OrderedDict] = None,
    cam_mapping: Optional[Dict[str, str]] = None,
    generate_subtask_ratio: float = 0.0,
    camera_name_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[str, bool]:
    """Get wallx normal text."""
    role_start_symbol = "<|im_start|>"
    role_end_symbol = "<|im_end|>"
    vision_start_symbol = "<|vision_start|>"
    vision_end_symbol = "<|vision_end|>"
    image_pad_symbol = "<|image_pad|>"
    propri_symbol = "<|propri|>"
    action_symbol = "<|action|>"
    action_fast_symbol = "<|action_fast|>"

    prologue = f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"
    user_request = f"{role_start_symbol}user\nObservation:"
    if cam_mapping:
        camera_name_mapping = camera_name_mapping or {}
        for _, cam_name in cam_mapping.items():
            view_name = camera_name_mapping.get(cam_name, cam_name)
            user_request += f" {view_name}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
    user_request += "\nInstruction:"

    frame_instruction_info, _ = get_frame_instruction(instruction_info, frame_idx=frame_idx)
    generate_subtask = False
    priority_keys = ["subtask_generation", "distribute"]
    if bool(set(frame_instruction_info.keys()) & set(priority_keys)) and random.random() < generate_subtask_ratio:
        instruction = frame_instruction_info.get("instruction", "")
        text_prompt = "\nPredict the next action in language.\n"
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
        output_instruction = ""
        for key in priority_keys:
            if key in frame_instruction_info:
                output_instruction = frame_instruction_info[key]
                break
        assistant_output = f"{role_start_symbol}assistant\n{output_instruction}\n{role_end_symbol}"
        generate_subtask = True
    else:
        instruction = get_task_instruction(frame_instruction_info, priority_order=priority_order)
        text_prompt = f"\nPredict the next action in robot action.\nProprioception: {propri_symbol}\n"
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
        assistant_output = (
            f"{role_start_symbol}assistant\n{action_fast_symbol}{role_end_symbol}\n{action_symbol * action_chunk_size}"
        )

    return prologue + user_message + assistant_output, generate_subtask


def replace_action_token(
    text: List[str],
    norm_action: Optional[torch.Tensor],
    action_tokenizer,
    dof_masks: Optional[torch.Tensor] = None,
) -> List[str]:
    """Replace action token."""
    del norm_action, action_tokenizer, dof_masks
    return [t.replace("<|action_fast|><|im_end|>\n", "") for t in text]


def preprocesser_call(
    processor,
    images: Optional[Union[List, Any]] = None,
    text: Optional[Union[str, List[str]]] = None,
    videos: Optional[Union[List, Any]] = None,
    padding: Union[bool, str] = False,
    truncation: Optional[bool] = None,
    max_length: Optional[int] = None,
    return_tensors: str = "pt",
    norm_state=None,
    agent_pos_mask=None,
    state_bins: int = 256,
    state_drop_prob: float = 0.0,
) -> BatchFeature:
    """Preprocesser call."""
    if images is not None and len(images) > 0:
        image_inputs = processor.image_processor(images=images, return_tensors=return_tensors)
        image_grid_thw = image_inputs["image_grid_thw"]
    else:
        image_inputs = {}
        image_grid_thw = None

    if videos is not None:
        if not hasattr(processor, "video_processor"):
            raise RuntimeError("processor has no video_processor attribute")
        videos_inputs = processor.video_processor(videos=videos, return_tensors=return_tensors)
        video_grid_thw = videos_inputs["video_grid_thw"]
    else:
        videos_inputs = {}
        video_grid_thw = None

    if not isinstance(text, list):
        text = [text]

    if norm_state is not None:
        norm_state = norm_state.cpu().numpy() if isinstance(norm_state, torch.Tensor) else norm_state
        discretized = np.digitize(norm_state, bins=np.linspace(-1, 1, state_bins + 1)[:-1]) - 1
        discretized = discretized[:, 0, :]
        if agent_pos_mask is not None:
            mask = agent_pos_mask[:, 0, :].cpu().numpy().astype(bool)
        else:
            mask = np.ones(discretized.shape, dtype=bool)
        for i in range(len(text)):
            if "<|propri|>" not in text[i]:
                continue
            if state_drop_prob > 0 and random.random() < state_drop_prob:
                text[i] = text[i].replace("<|propri|>", "")
            else:
                state_str = " ".join(map(str, discretized[i, mask[i]]))
                text[i] = text[i].replace("<|propri|>", state_str)

    if image_grid_thw is not None:
        merge_length = processor.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while "<|image_pad|>" in text[i]:
                if index >= len(image_grid_thw):
                    logger.warning("More image placeholders than images; leaving extra placeholders unchanged")
                    break
                token_count = image_grid_thw[index].prod() // merge_length
                text[i] = text[i].replace("<|image_pad|>", "<|placeholder|>" * token_count, 1)
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

    if video_grid_thw is not None:
        merge_length = processor.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while "<|video_pad|>" in text[i]:
                token_count = video_grid_thw[index].prod() // merge_length
                text[i] = text[i].replace("<|video_pad|>", "<|placeholder|>" * token_count, 1)
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|video_pad|>")

    text_inputs = processor.tokenizer(
        text,
        return_tensors=return_tensors,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
    )
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    labels = torch.full_like(text_inputs.input_ids, -100)
    assistant_marker = "<|im_start|>assistant\n"
    im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_tokens = processor.tokenizer(assistant_marker, add_special_tokens=False).input_ids

    for i, sample_text in enumerate(text):
        parts = sample_text.split(assistant_marker)
        num_left_pads = 0
        for token_id in text_inputs.input_ids[i]:
            if token_id == pad_token_id:
                num_left_pads += 1
            else:
                break
        current_pos = num_left_pads
        assistant_regions = []
        for j, part in enumerate(parts):
            part_tokens = processor.tokenizer(part, add_special_tokens=False).input_ids
            if j == 0:
                current_pos += len(part_tokens)
                continue
            for k in range(current_pos + 1, len(text_inputs.input_ids[i])):
                if text_inputs.input_ids[i][k] == im_end_token_id:
                    assistant_regions.append((current_pos + len(assistant_tokens), k + 2))
                    break
            current_pos += len(part_tokens) + 3
        for start, end in assistant_regions:
            labels[i][start:end] = text_inputs.input_ids[i][start:end]

    action_token_id = processor.tokenizer.encode("<|action|>")[0]
    propri_token_id = processor.tokenizer.encode("<|propri|>")[0]
    labels[labels == action_token_id] = -100
    labels[labels == propri_token_id] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    text_inputs["labels"] = labels if (labels != -100).any().item() else None
    return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})
