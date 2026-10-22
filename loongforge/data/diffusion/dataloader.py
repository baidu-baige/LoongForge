# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion dataloader entries: latent (per-sample) and packed (sequence packing)."""

import torch

from .latent_dataset import LatentDataset
from .packed_dataset import PackedDataset


def build_latent_dataloader(
    data_path, steps_per_epoch, seed, keep_keys, num_workers, dp_rank, dp_size, consumed_samples=0
):
    """Build the dataloader over pre-cached latents, one sample per batch.

    ``consumed_samples`` is the global sample count already trained. The offset is
    aligned to ``dp_size`` so each rank keeps the same shard assignment.
    """
    start_index = (consumed_samples // dp_size) * dp_size if dp_size > 0 else 0
    start_index = min(start_index, max(0, steps_per_epoch - dp_size))
    dataset = LatentDataset(
        data_path,
        steps_per_epoch,
        seed=seed,
        keep_keys=keep_keys,
        data_parallel_size=dp_size,
        start_index=start_index,
    )
    sampler = torch.utils.data.DistributedSampler(
        dataset, shuffle=False, num_replicas=dp_size,
        rank=dp_rank, drop_last=True,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
    )


def build_packed_dataloader(
    data_path, steps_per_epoch, scheduler, packing_buffer_size, seq_length,
    dp_rank, dp_size, cp_size, context_max_len, model_name, seed,
    max_timestep_boundary, min_timestep_boundary, global_batch_size, consumed_samples=0,
):
    """Build the dataloader over packed bins (PackedDataset yields merged batches)."""
    consumed_bins = consumed_samples // global_batch_size if global_batch_size else 0
    dataset = PackedDataset(
        data_path=data_path,
        steps_per_epoch=steps_per_epoch,
        scheduler=scheduler,
        cp_world_size=cp_size,
        packing_buffer_size=packing_buffer_size,
        seq_length=seq_length,
        dp_rank=dp_rank,
        dp_world_size=dp_size,
        context_max_len=context_max_len,
        model_name=model_name,
        seed=seed,
        max_timestep_boundary=max_timestep_boundary,
        min_timestep_boundary=min_timestep_boundary,
        consumed_bins=consumed_bins,
    )
    # batch_size=None: DataLoader yields each item as-is (no batching /
    # collation).  PackedDataset already produces fully-merged packed
    # batches so no further assembly is needed.
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=True,
    )
