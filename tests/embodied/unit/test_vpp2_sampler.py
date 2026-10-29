# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check full-rank tail geometry against the community's Accelerate carrier."""

import pytest
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from loongforge.embodied.data.datasets.vpp2.samplers import VPP2BlockShardSampler

BatchSamplerShard = pytest.importorskip("accelerate.data_loader").BatchSamplerShard


def test_epoch_only_checkpoint_cannot_silently_replay_consumed_batches():
    sampler = VPP2BlockShardSampler(
        range(579), batch_size=32, num_replicas=8, rank=0, shuffle=True, seed=42,
    )
    with pytest.raises(ValueError, match="no sample cursor"):
        sampler.load_state_dict({"epoch": 0})


@pytest.mark.parametrize("length", [0, 1, 31, 32, 33, 255, 256, 257, 512, 579, 768])
@pytest.mark.parametrize("world_size", [1, 2, 8])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_all_ranks_match_accelerate(length, world_size, drop_last, shuffle):
    dataset = range(length)
    for epoch in (0, 1, 7):
        rank_lengths = []
        for rank in range(world_size):
            sampler = VPP2BlockShardSampler(
                dataset, batch_size=32, num_replicas=world_size, rank=rank,
                shuffle=shuffle, seed=42, drop_last=drop_last,
            )
            sampler.set_epoch(epoch)
            reference_sampler = (
                RandomSampler(dataset, generator=torch.Generator().manual_seed(42 + epoch))
                if shuffle and length else SequentialSampler(dataset)
            )
            reference = BatchSampler(reference_sampler, 32, drop_last)
            if world_size > 1:
                reference = BatchSamplerShard(
                    reference, num_processes=world_size, process_index=rank, even_batches=True,
                )
            actual = list(BatchSampler(sampler, 32, drop_last))
            assert actual == list(reference), (length, world_size, rank, epoch)
            assert len(sampler) == sum(len(batch) for batch in actual)
            assert len(actual) == len(reference)
            rank_lengths.append(len(actual))
        assert len(set(rank_lengths)) == 1


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_resume_preserves_padded_tail_and_next_epoch(rank, num_workers):
    def make_loader():
        sampler = VPP2BlockShardSampler(
            range(579), batch_size=32, num_replicas=8, rank=rank,
            shuffle=True, seed=42,
        )
        return StatefulDataLoader(range(579), sampler=sampler, batch_size=32, num_workers=num_workers)

    uninterrupted = make_loader()
    uninterrupted.sampler.set_epoch(3)
    iterator = iter(uninterrupted)
    next(iterator)
    next(iterator)
    state = uninterrupted.state_dict()
    expected = list(iterator)
    resumed = make_loader()
    resumed.load_state_dict(state)
    actual = list(resumed)
    assert len(actual) == len(expected) == 1
    assert torch.equal(actual[0], expected[0])
    assert len(actual[0]) == 32
    assert resumed.sampler.epoch == 3
    for loader in (uninterrupted, resumed):
        loader.sampler.set_epoch(4)
    assert all(torch.equal(a, b) for a, b in zip(uninterrupted, resumed, strict=True))
