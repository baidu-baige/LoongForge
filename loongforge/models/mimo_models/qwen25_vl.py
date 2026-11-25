# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5-VL vision components adapted to the MCore MiMo contract."""

from __future__ import annotations

from typing import Optional

import torch

from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules


def normalize_encoder_inputs(encoder_inputs: dict) -> dict:
    """Accept Mimo's named-encoder payload and LoongForge's flat payload."""
    if "images" in encoder_inputs:
        return encoder_inputs
    try:
        return encoder_inputs["image_encoder"]
    except KeyError as exc:
        raise KeyError(
            "Qwen2.5-VL image inputs must contain 'images' and "
            "'image_grid_thw', either directly or under 'image_encoder'"
        ) from exc


def split_image_inputs_for_fan_in(image_inputs, slot, scale, spatial_merge_size):
    """Assign contiguous whole images to source DP slots in gather order."""
    images = image_inputs["images"]
    image_grid_thw = image_inputs["image_grid_thw"]
    patch_counts = image_grid_thw.to(torch.int64).prod(dim=1).tolist()
    patches_per_token = spatial_merge_size**2
    if sum(patch_counts) != images.size(0) or any(
        count <= 0 or count % patches_per_token for count in patch_counts
    ):
        raise ValueError("Qwen2.5-VL image patches do not match image_grid_thw")

    images_per_slot, remainder = divmod(len(patch_counts), scale)
    image_offsets = [
        index * images_per_slot + min(index, remainder) for index in range(scale + 1)
    ]
    patch_offsets = [sum(patch_counts[:offset]) for offset in image_offsets]
    fan_in_sizes = tuple(
        (end - start) // patches_per_token
        for start, end in zip(patch_offsets, patch_offsets[1:])
    )
    if fan_in_sizes[slot] == 0:
        return None, fan_in_sizes
    return {
        "images": images[patch_offsets[slot] : patch_offsets[slot + 1]],
        "image_grid_thw": image_grid_thw[image_offsets[slot] : image_offsets[slot + 1]],
    }, fan_in_sizes


class Qwen25VisionModalitySubmodules(VisionModalitySubmodules):
    """Run the existing Qwen vision encoder and projector as one MiMo modality."""

    @classmethod
    def from_spec(cls, module_spec, is_first_stage=True, is_last_stage=True):
        from loongforge.train.initialize import change_parallel_state

        change_parallel_state("image_encoder")
        try:
            return super().from_spec(module_spec, is_first_stage, is_last_stage)
        finally:
            change_parallel_state("text_decoder")

    def forward(
        self,
        encoder_inputs: Optional[dict] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if encoder_inputs is not None and hidden_states is not None:
            raise ValueError("encoder_inputs and hidden_states are mutually exclusive")
        if hidden_states is not None:
            return hidden_states
        if encoder_inputs is None:
            return None

        from loongforge.train.initialize import change_parallel_state

        encoder_inputs = normalize_encoder_inputs(encoder_inputs)
        images = encoder_inputs["images"]
        image_grid_thw = encoder_inputs["image_grid_thw"]
        encoder = self.encoders["image_encoder"]
        change_parallel_state("image_encoder")
        try:
            embeddings, window_index, _ = encoder(images, image_grid_thw=image_grid_thw)
            if self.input_projections:
                embeddings = self.input_projections[0](embeddings, window_index)
        finally:
            change_parallel_state("text_decoder")

        if isinstance(embeddings, (list, tuple)):
            embeddings = torch.cat(
                [embedding.reshape(-1, embedding.shape[-1]) for embedding in embeddings],
                dim=0,
            )
        return embeddings
