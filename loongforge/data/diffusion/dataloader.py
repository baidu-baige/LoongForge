# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion dataloader entries: latent (per-sample) and packed (sequence packing)."""

import torch

from .latent_dataset import LatentDataset
from .packed_dataset import PackedDataset


def build_latent_dataloader(
    data_path, steps_per_epoch, seed, keep_keys, num_workers, dp_rank, dp_size
):
    """Build the dataloader over pre-cached latents, one sample per batch."""
    dataset = LatentDataset(
        data_path,
        steps_per_epoch,
        seed=seed,
        keep_keys=keep_keys,
        data_parallel_size=dp_size,
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
    max_timestep_boundary, min_timestep_boundary,
):
    """Build the dataloader over packed bins (PackedDataset yields merged batches)."""
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
