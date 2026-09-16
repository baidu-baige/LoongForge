# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""reconstruct funcs"""

import torch
from megatron.training import get_args


def reconstruct_llm_for_internvl(tensor):
    """Reshape packed LLM tensors to (1, seq_len) for InternVL."""
    return tensor.reshape(1, -1)


def reconstruct_image_flags_for_internvl(tensor):
    """Image flag metadata; no reconstruction needed."""
    return tensor


def reconstruct_pixel_values_for_internvl(tensor):
    """Reshape flattened pixels to (N, 3, H, W) using force_image_size."""
    img_size = get_args().force_image_size
    return tensor.reshape(-1, 3, img_size, img_size)


def reconstruct_llm_for_vlm(tensor):
    """Reshape packed LLM tensors to (1, seq_len) for VLM."""
    return tensor.reshape(1, -1)


def reconstruct_visual_grid_thw_for_vlm(tensor):
    """Reshape visual grid metadata to (num_visuals, 3)."""
    return tensor.reshape(-1, 3)


def reconstruct_visual_for_vlm(tensor, visual_size):
    """Reshape flattened visual embeddings to (num_tokens, visual_size)."""
    return tensor.reshape(-1, visual_size)


def reconstruct_position_ids_for_vlm(tensor):
    """Reshape position IDs to (3, 1, seq_len) for InternVL-style VLM."""
    return tensor.reshape(3, 1, -1)

