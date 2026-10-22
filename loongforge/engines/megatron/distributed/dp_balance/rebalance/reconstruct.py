# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""reconstruct funcs"""


def reconstruct_llm_for_vlm(tensor):
    """Reshape packed LLM tensors to (1, seq_len) for VLM."""
    return tensor.reshape(1, -1)


def reconstruct_visual_grid_thw_for_vlm(tensor):
    """Reshape visual grid metadata to (num_visuals, 3)."""
    return tensor.reshape(-1, 3)


def reconstruct_visual_for_vlm(tensor, visual_size):
    """Reshape flattened visual embeddings to (num_tokens, visual_size)."""
    return tensor.reshape(-1, visual_size)
