# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion dataloader entries: latent (per-sample) and packed (sequence packing)."""

import torch
from megatron.core import parallel_state

from .diffusion.latent_dataset import LatentDataset
from .diffusion.packed_dataset import PackedDataset


def build_latent_dataloader(data_path, steps_per_epoch, seed, keep_keys, num_workers):
    """Build the dataloader over pre-cached latents, one sample per batch."""
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()
    dataset = LatentDataset(
        data_path,
        steps_per_epoch,
        seed=seed,
        keep_keys=keep_keys,
        data_parallel_size=dp_world_size,
    )
    sampler = torch.utils.data.DistributedSampler(
        dataset, shuffle=False, num_replicas=dp_world_size,
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
    data_path, steps_per_epoch, args, scheduler, packing_buffer_size, seq_length
):
    """Build the dataloader over packed bins (PackedDataset yields merged batches)."""
    dataset = PackedDataset(
        data_path=data_path,
        steps_per_epoch=steps_per_epoch,
        args=args,
        scheduler=scheduler,
        cp_world_size=parallel_state.get_context_parallel_world_size(),
        packing_buffer_size=packing_buffer_size,
        seq_length=seq_length,
        dp_rank=parallel_state.get_data_parallel_rank(),
        dp_world_size=parallel_state.get_data_parallel_world_size(),
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
