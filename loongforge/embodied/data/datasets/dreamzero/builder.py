# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero dataset strategy and sampler construction."""

from __future__ import annotations

import logging
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from loongforge.embodied.data.datasets.dreamzero.transforms.dreamzero_collator import (
    DreamTransform,
)
from loongforge.embodied.data.datasets.dreamzero.dataset.datasets import (
    DreamZeroLeRobotMixtureDataset,
    DreamZeroLeRobotDataset,
)
from loongforge.embodied.data.datasets.dreamzero.dataset.modality_configs import (
    EMBODIMENT_BUILDERS,
    EMBODIMENT_TAG_TO_ID,
)
from loongforge.embodied.data.datasets.dreamzero.sampler import (
    DreamZeroShardedSampler,
)
from loongforge.embodied.data.datasets.dreamzero.transforms.base import (
    ComposedModalityTransform,
)
from loongforge.embodied.data.datasets.dreamzero.transforms.concat import (
    ConcatTransform,
)
from loongforge.embodied.data.datasets.dreamzero.transforms.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from loongforge.embodied.data.datasets.dreamzero.transforms.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from loongforge.embodied.distributed import DistributedContext
from loongforge.embodied.model.dreamzero.precomputed_cache import (
    DreamZeroPrecomputedCacheConfig,
    build_precomputed_cache_config,
)

logger = logging.getLogger(__name__)


class _StatefulIndexIterator(Iterator[int]):
    """Iterator state used by torchdata's StatefulDataLoader."""

    def __init__(self, sampler: "GlobalBatchShardSampler") -> None:
        self._sampler = sampler
        self._yielded = 0

    def __iter__(self) -> "_StatefulIndexIterator":
        return self

    def __next__(self) -> int:
        if self._yielded >= len(self._sampler._indices):
            raise StopIteration
        value = self._sampler._indices[self._yielded]
        self._yielded += 1
        return value

    def state_dict(self) -> dict[str, int]:
        """Return the iterator state for checkpointing."""
        return {"yielded": self._yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the iterator position from a checkpoint."""
        yielded = int(state_dict.get("yielded", 0))
        if yielded < 0 or yielded > len(self._sampler._indices):
            raise ValueError(
                f"Cannot restore DreamZero sampler iterator yielded={yielded}; "
                f"current sampler only has {len(self._sampler._indices)} indices"
            )
        self._yielded = yielded


class GlobalBatchShardSampler(Sampler[int]):
    """Shard a global index order by whole local batches across DP ranks."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank} for world_size {num_replicas}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._indices = self._build_indices()

    def _build_indices(self) -> list[int]:
        length = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(length, generator=generator).tolist()
        else:
            indices = list(range(length))

        if self.drop_last:
            usable = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:usable]

        out: list[int] = []
        global_batch = self.batch_size * self.num_replicas
        local_start = self.rank * self.batch_size
        for start in range(local_start, len(indices), global_batch):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                out.extend(batch)
        return out

    def __iter__(self) -> Iterator[int]:
        return _StatefulIndexIterator(self)

    def __len__(self) -> int:
        return len(self._indices)

    def set_epoch(self, epoch: int) -> None:
        """Rebuild this rank's indices for a new epoch."""
        self.epoch = int(epoch)
        self._indices = self._build_indices()

    def state_dict(self) -> dict[str, Any]:
        """Return sampler state for checkpointing."""
        return {"version": 1, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore sampler state from a checkpoint."""
        self.set_epoch(int(state_dict.get("epoch", 0)))


def _build_dreamzero_transforms(
    *,
    modality_configs,
    image_height: int,
    image_width: int,
    crop_scale: float,
    enable_color_jitter: bool,
    max_state_dim: int,
    max_action_dim: int,
    state_horizon: int,
    action_horizon: int,
    max_text_length: int,
    tokenizer_path: str,
    precomputed_cache_config: DreamZeroPrecomputedCacheConfig,
) -> ComposedModalityTransform:
    video_keys = modality_configs["video"].modality_keys
    state_keys = modality_configs["state"].modality_keys
    action_keys = modality_configs["action"].modality_keys

    pipeline = [
        VideoToTensor(apply_to=video_keys),
        VideoCrop(apply_to=video_keys, scale=crop_scale),
        VideoResize(
            apply_to=video_keys,
            height=image_height,
            width=image_width,
            interpolation="linear",
        ),
    ]
    if enable_color_jitter:
        pipeline.append(
            VideoColorJitter(
                apply_to=video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            )
        )
    pipeline += [
        VideoToNumpy(apply_to=video_keys),
        StateActionToTensor(apply_to=state_keys + action_keys),
        StateActionTransform(
            apply_to=state_keys + action_keys,
            normalization_modes={k: "q99" for k in state_keys + action_keys},
        ),
        ConcatTransform(
            video_concat_order=video_keys,
            state_concat_order=state_keys,
            action_concat_order=action_keys,
        ),
        DreamTransform(
            default_instruction="Perform the default behavior.",
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            max_length=max_text_length,
            state_horizon=state_horizon,
            action_horizon=action_horizon,
            embodiment_tag_mapping=EMBODIMENT_TAG_TO_ID,
            tokenizer_path=tokenizer_path,
            precomputed_video_latents=precomputed_cache_config.video_latents.enabled,
            precomputed_video_latents_key=precomputed_cache_config.video_latents.batch_key,
            precomputed_first_frame_latents=(
                precomputed_cache_config.first_frame_latents.enabled
            ),
            precomputed_first_frame_latents_key=(
                precomputed_cache_config.first_frame_latents.batch_key
            ),
        ),
    ]
    return ComposedModalityTransform(transforms=pipeline)


def build_dreamzero_sampler(
    *,
    train_dataset,
    base_dataset,
    sampler_type: str,
    ctx: DistributedContext,
    batch_size: int,
    num_workers: int,
    sampler_seed: int,
    data_cfg,
) -> tuple[Sampler | None, bool]:
    """Select the DreamZero sampler for ``sampler_type``.

    This is the model-specific implementation behind the framework sampler
    registry. It returns ``(sampler, shuffle)`` for compatibility with the
    original DreamZero sampler semantics.

    Naming convention (aligned with the generic framework samplers):
    * ``<Descriptor>Sampler`` PascalCase; model prefix only when the sampler is
      genuinely model-specific.
    * ``global_batch_shard`` -> ``GlobalBatchShardSampler`` (dataset-agnostic DP
      index layout; no prefix, mirrors the framework's ``_BlockShardSampler``).
    * ``sharded`` -> ``DreamZeroShardedSampler`` (LeRobot-trajectory specific, so
      it keeps the ``DreamZero`` prefix).
    * ``distributed`` -> torchdata ``StatefulDistributedSampler``.

    Cross-axis note: DreamZero ``sampler_type`` values map onto the generic
    ``distributed_sampler_mode`` vocabulary as ``distributed`` == ``cyclic`` and
    ``global_batch_shard`` ~= ``block`` (near-equivalent; differs only at the
    tail/remainder and resume-state handling).
    """
    ddp_active = ctx.is_distributed and ctx.world_size > 1
    replicas = ctx.world_size if ddp_active else 1
    rank = ctx.rank if ddp_active else 0
    sampler_type = str(sampler_type or "distributed").strip().lower()
    require_full_language_chunks = bool(data_cfg.require_full_language_chunks)
    sampler_worker_batching = str(data_cfg.sampler_worker_batching or "none").strip().lower()

    if sampler_type == "global_batch_shard":
        sampler = GlobalBatchShardSampler(
            train_dataset,
            batch_size=batch_size,
            num_replicas=replicas,
            rank=rank,
            seed=sampler_seed,
            shuffle=True,
            drop_last=False,
        )
        if rank == 0:
            logger.info(
                "[dreamzero-data] using global batch-shard sampler "
                "seed=%s micro_batch=%s replicas=%s dataset_len=%s",
                sampler_seed,
                batch_size,
                replicas,
                len(train_dataset),
            )
        return sampler, False
    if sampler_type == "sharded":
        sampler = DreamZeroShardedSampler(
            base_dataset,
            num_replicas=replicas,
            rank=rank,
            seed=sampler_seed,
            shard_sampling_rate=float(data_cfg.shard_sampling_rate),
            num_steps_per_shard=int(data_cfg.num_steps_per_shard),
            num_shards_to_sample=int(data_cfg.num_shards_to_sample),
            training=True,
            allow_padding_at_end=bool(data_cfg.allow_padding_at_end),
            require_full_language_chunks=require_full_language_chunks,
            worker_batching_mode=sampler_worker_batching,
            dataloader_num_workers=num_workers,
            micro_batch_size=batch_size,
        )
        return sampler, False
    if sampler_type == "distributed":
        if ddp_active:
            sampler = StatefulDistributedSampler(
                train_dataset,
                num_replicas=replicas,
                rank=rank,
                shuffle=True,
                seed=sampler_seed,
                drop_last=False,
            )
            return sampler, False
        return None, True
    raise ValueError(
        "Unknown DreamZero sampler_type="
        f"{sampler_type!r}; expected distributed, sharded, or global_batch_shard"
    )


def build_dreamzero_dataset(model_cfg, data_cfg, training_args):
    """Build the DreamZero LeRobot dataset (single or mixture).

    The framework selects this through ``--dataset-strategy dreamzero``. Sampler
    configuration is retained on the returned dataset for the model sampler
    registry; modality transforms remain dataset-owned to preserve deterministic
    per-sample augmentation.
    """
    dataset_path = training_args.dataset_path
    if dataset_path is None:
        raise ValueError("DreamZero requires --dataset-path to point to a LeRobot dataset root")

    embodiment_tag = data_cfg.embodiment_tag
    if embodiment_tag not in EMBODIMENT_BUILDERS:
        raise ValueError(
            f"unknown embodiment_tag={embodiment_tag!r}; supported: {sorted(EMBODIMENT_BUILDERS)}"
        )

    language_chunk_sampling = bool(data_cfg.language_chunk_sampling)
    modality_video_frames = 25 if language_chunk_sampling else int(model_cfg.num_frames)
    modality_chunk_size = 1 if language_chunk_sampling else int(model_cfg.max_chunk_size)
    modality_configs = EMBODIMENT_BUILDERS[embodiment_tag](
        num_video_frames=modality_video_frames,
        action_horizon=int(model_cfg.action_horizon),
        state_horizon=int(model_cfg.state_horizon),
        max_chunk_size=modality_chunk_size,
    )

    tokenizer_path = training_args.tokenizer_path
    if not tokenizer_path:
        raise ValueError("DreamZero requires --tokenizer-path for text tokenization")
    video_backend = training_args.video_backend or "decord"
    precomputed_cache_config = build_precomputed_cache_config(model_cfg)
    transforms = _build_dreamzero_transforms(
        modality_configs=modality_configs,
        image_height=int(data_cfg.image_height),
        image_width=int(data_cfg.image_width),
        crop_scale=float(data_cfg.crop_scale),
        enable_color_jitter=bool(data_cfg.enable_color_jitter),
        max_state_dim=int(model_cfg.max_state_dim),
        max_action_dim=int(model_cfg.max_action_dim),
        state_horizon=int(model_cfg.state_horizon),
        action_horizon=int(model_cfg.action_horizon),
        max_text_length=int(data_cfg.max_text_length),
        tokenizer_path=tokenizer_path,
        precomputed_cache_config=precomputed_cache_config,
    )
    use_sample_transform_seed = bool(data_cfg.use_sample_transform_seed)
    sample_transform_seed = int(data_cfg.sample_transform_seed)

    base_dataset = DreamZeroLeRobotDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
        use_global_metadata=bool(data_cfg.use_global_metadata),
        metadata_version=data_cfg.metadata_version,
        video_backend=video_backend,
        transforms=transforms,
        max_chunk_size=int(model_cfg.max_chunk_size),
        relative_action=bool(data_cfg.relative_action),
        relative_action_keys=data_cfg.relative_action_keys,
        relative_action_per_horizon=bool(data_cfg.relative_action_per_horizon),
        language_chunk_sampling=language_chunk_sampling,
        use_sample_transform_seed=use_sample_transform_seed,
        sample_transform_seed=sample_transform_seed,
        precomputed_cache_config=precomputed_cache_config,
    )

    run_seed = int(training_args.seed or 0)
    use_mixture_dataset = bool(data_cfg.use_mixture_dataset)
    mixture_sampling_seed_cfg = data_cfg.mixture_sampling_seed
    mixture_sampling_seed = (
        run_seed if mixture_sampling_seed_cfg is None else int(mixture_sampling_seed_cfg)
    )
    if use_mixture_dataset:
        train_dataset = DreamZeroLeRobotMixtureDataset(
            data_mixture=[(base_dataset, 1.0)],
            training=True,
            balance_dataset_weights=bool(data_cfg.balance_dataset_weights),
            balance_trajectory_weights=bool(data_cfg.balance_trajectory_weights),
            seed=mixture_sampling_seed,
            allow_padding_at_end=bool(data_cfg.allow_padding_at_end),
            use_sample_transform_seed=use_sample_transform_seed,
            sample_transform_seed=sample_transform_seed,
        )
    else:
        train_dataset = base_dataset

    # The sampler registry receives the dataset but not the typed data config.
    train_dataset._dreamzero_sampler_cfg = data_cfg
    train_dataset._dreamzero_base_dataset = base_dataset
    return train_dataset
