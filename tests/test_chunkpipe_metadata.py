# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ChunkPipe sampler metadata travels with its batch through the DataLoader."""

import pytest
import torch
from torch.utils.data import DataLoader

from loongforge.engines.megatron.distributed.chunkpipe import ChunkPipeGroupBatchSampler

# Source groups of 1, 2 and 3 chunks; every group fits a step of 4 micro-batches.
GROUP_SIZES = [1, 2, 2, 3, 3, 3, 1, 1, 2, 2, 1, 3, 3, 3, 1, 2, 2, 1, 1, 1]
NUM_MICROBATCHES = 4


class _ChunkDataset:
    def __init__(self, group_sizes):
        self._sizes = group_sizes

    def __len__(self):
        return len(self._sizes)

    def __getitem__(self, key):
        if key == "chunk_group_size":
            return self._sizes
        return {"idx": key}


def _collate(items):
    return {"idx": torch.tensor([item["idx"] for item in items])}


def _sampler(consumed_samples=0):
    dataset = _ChunkDataset(GROUP_SIZES)
    return dataset, ChunkPipeGroupBatchSampler(
        dataset,
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=1,
        data_parallel_rank=0,
        data_parallel_size=1,
        num_microbatches=NUM_MICROBATCHES,
        seed=7,
    )


def _reference(epochs, consumed_samples=0):
    """(indices, step groups, composite) per micro-batch, read straight off the sampler."""
    _, sampler = _sampler(consumed_samples)
    out = []
    for _ in range(epochs):
        for indices in sampler:
            out.append((list(indices), sampler._step_g_queue[-1], list(sampler._composite_queue[-1])))
    return out


def _through_dataloader(epochs, num_workers, consumed_samples=0):
    dataset, sampler = _sampler(consumed_samples)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=_collate, num_workers=num_workers)

    def epochs_iter():
        for _ in range(epochs):
            yield from loader

    out = []
    for batch in sampler.attach_step_metadata(epochs_iter()):
        out.append((batch["idx"].tolist(), batch["chunkpipe_step_num_groups"], batch["chunkpipe_component_sizes"]))
    # Every appended entry was consumed by exactly one batch.
    assert not sampler._step_g_queue and not sampler._composite_queue
    return out


@pytest.mark.parametrize("num_workers", [0, 2])
def test_metadata_matches_its_batch_across_epochs(num_workers):
    expected = _reference(epochs=2)
    assert len(expected) > len(GROUP_SIZES)  # crosses the epoch boundary
    assert _through_dataloader(epochs=2, num_workers=num_workers) == expected


@pytest.mark.parametrize("num_workers", [0, 2])
def test_metadata_after_resume_skip(num_workers):
    full = _reference(epochs=1)
    skipped_steps = 2
    consumed = skipped_steps * NUM_MICROBATCHES
    resumed = _through_dataloader(epochs=1, num_workers=num_workers, consumed_samples=consumed)
    assert resumed == _reference(epochs=1, consumed_samples=consumed)
    # Skipped steps leave no queue entries: resuming continues the uninterrupted sequence.
    assert 0 < len(resumed) < len(full)
    assert resumed == full[len(full) - len(resumed):]
