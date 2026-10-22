# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Frozen data-recipe config for VLM datasets, loaders, and task encoders."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class VLMDataConfig:
    """Data-recipe fields needed to build the Energon datasets and task encoders.

    Built by the training method; the data layer never reads backend arguments.
    """

    data_path: list
    micro_batch_size: int
    num_workers: int
    packing_buffer_size: int
    data_shuffle_buffer_size: int = 0
    data_max_samples_per_sequence: int = 0
    valid_data_path: Optional[list] = None
    add_question_in_pretrain: Any = None
    conv_style: Any = None
    down_sample_ratio: Any = None
    dynamic_image_size: Any = None
    enable_discard_sample: Any = None
    energon_pack_algo: Any = None
    ernie_chat_template: Any = None
    force_image_size: Any = None
    fps: Any = None
    fps_max_frames: Any = None
    fps_min_frames: Any = None
    frame_max_pixels: Any = None
    frame_min_pixels: Any = None
    group_by_length: Any = None
    hf_tokenizer_path: Any = None
    image_dtype: Any = None
    image_resolution: Any = None
    image_token_len: Any = None
    loss_reduction: Any = None
    max_buffer_size: Any = None
    max_dynamic_patch: Any = None
    max_num_frame: Any = None
    max_packed_tokens: Any = None
    max_pixels: Any = None
    min_dynamic_patch: Any = None
    min_num_frame: Any = None
    min_pixels: Any = None
    normalize_type: Any = None
    num_images_expected: Any = None
    packing_pretrain_data: Any = None
    packing_sft_data: Any = None
    pad: Any = None
    pad2square: Any = None
    sample_type: Any = None
    seed: Any = None
    seq_length: Any = None
    sft_dataset: Any = None
    sft_dataset_config: Any = None
    sft_image_normalize: Any = None
    sft_image_rescale: Any = None
    sft_replace_ids: Any = None
    sft_shift_by_one: Any = None
    strict_mode: Any = None
    task_encoder: Any = None
    training_phase: Any = None
    training_rice_vl_max_answer_length: Any = None
    use_thumbnail: Any = None
    video_max_pixels: Any = None
