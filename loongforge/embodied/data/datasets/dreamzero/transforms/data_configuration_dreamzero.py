# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero DataConfig for YAML ``data:`` sections."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DreamZeroDataConfig:
    """Data-processing and sampler config for DreamZero LeRobot training."""

    # Canonical dataset/schema tag used for modality and projector routing.
    embodiment_tag: str = "oxe_droid"

    # Video pipeline and tokenizer preprocessing.
    image_height: int = 176
    image_width: int = 320
    crop_scale: float = 0.95
    enable_color_jitter: bool = True
    use_sample_transform_seed: bool = False
    sample_transform_seed: int = 0
    max_text_length: int = 512

    # Dataset metadata and action normalization.
    use_global_metadata: bool = False
    metadata_version: str | None = "0221"
    relative_action: bool = False
    relative_action_keys: list[str] | None = None
    relative_action_per_horizon: bool = False
    language_chunk_sampling: bool = False
    require_full_language_chunks: bool = False

    # Mixture dataset and sampling.
    use_mixture_dataset: bool = False
    balance_dataset_weights: bool = False
    balance_trajectory_weights: bool = True
    sampler_type: str = "distributed"
    sampler_seed: int | None = None
    mixture_sampling_seed: int | None = None
    sampler_worker_batching: str = "none"
    shard_sampling_rate: float = 0.1
    num_steps_per_shard: int = 10_000
    num_shards_to_sample: int = 1024
    allow_padding_at_end: bool = False

    def __post_init__(self) -> None:
        if self.embodiment_tag not in {"oxe_droid", "libero_sim", "agibot", "yam"}:
            raise ValueError(
                f"Unsupported DreamZero embodiment_tag: {self.embodiment_tag!r}"
            )
        if self.sampler_type not in {"distributed", "sharded", "global_batch_shard"}:
            raise ValueError(
                "DreamZero sampler_type must be one of "
                "distributed, sharded, global_batch_shard"
            )
        if self.sampler_worker_batching not in {"none", "upstream_iterable"}:
            raise ValueError(
                "DreamZero sampler_worker_batching must be one of "
                "none, upstream_iterable"
            )
        if (not self.use_mixture_dataset) and self.mixture_sampling_seed is not None:
            raise ValueError(
                "mixture_sampling_seed is only valid when use_mixture_dataset=true"
            )
        if self.require_full_language_chunks and not self.language_chunk_sampling:
            raise ValueError(
                "require_full_language_chunks=true requires language_chunk_sampling=true"
            )
        if self.require_full_language_chunks and self.sampler_type != "sharded":
            raise ValueError(
                "require_full_language_chunks=true is only supported with sampler_type=sharded"
            )
        if self.sampler_type == "sharded" and self.use_mixture_dataset:
            raise ValueError(
                "sampler_type=sharded expects use_mixture_dataset=false"
            )
        if self.sampler_worker_batching != "none" and self.sampler_type != "sharded":
            raise ValueError(
                "sampler_worker_batching is only supported with sampler_type=sharded"
            )
