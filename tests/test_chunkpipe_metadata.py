# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ChunkPipe sampler metadata stays aligned with prefetched and resumed batches."""

import ast
from collections import deque, namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader


GROUP_SIZES = [1, 2, 2, 3, 3, 3, 1, 1, 2, 2, 1, 3, 3, 3, 1, 2, 2, 1, 1, 1]
NUM_MICROBATCHES = 4


@pytest.fixture
def sampler_api():
    path = Path(__file__).resolve().parents[1] / "loongforge/data/utils.py"
    names = {"ChunkPipeGroupBatchSampler", "_bind_chunkpipe_queue_iter"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == names
    args = SimpleNamespace()
    namespace = dict(
        torch=torch, deque=deque, Slot=namedtuple("Slot", ["chunks", "components"]),
        get_args=lambda: args,
    )
    # Run the sampler and iterator without unrelated training imports.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(
        sampler=namespace["ChunkPipeGroupBatchSampler"],
        bind=namespace["_bind_chunkpipe_queue_iter"],
        args=args,
    )


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


def _sampler(api, consumed_samples=0):
    dataset = _ChunkDataset(GROUP_SIZES)
    return dataset, api.sampler(
        dataset,
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=1,
        data_parallel_rank=0,
        data_parallel_size=1,
        num_microbatches=NUM_MICROBATCHES,
        seed=7,
    )


def _reference(api, epochs, consumed_samples=0):
    _, sampler = _sampler(api, consumed_samples)
    out = []
    for _ in range(epochs):
        for indices in sampler:
            out.append((list(indices), sampler._step_g_queue[-1], list(sampler._composite_queue[-1])))
    return out


def _through_dataloader(api, epochs, num_workers, consumed_samples=0):
    dataset, sampler = _sampler(api, consumed_samples)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=_collate, num_workers=num_workers)

    def epochs_iter():
        for _ in range(epochs):
            yield from loader

    out = []
    for batch in api.bind(epochs_iter(), sampler._step_g_queue, sampler._composite_queue):
        out.append((
            batch["idx"].tolist(),
            api.args.chunkpipe_step_g_queue.popleft(),
            list(api.args.chunkpipe_composite_queue.popleft()),
        ))
    assert not sampler._step_g_queue and not sampler._composite_queue
    return out


@pytest.mark.parametrize("num_workers", [0, 2])
def test_metadata_matches_its_batch_across_epochs(sampler_api, num_workers):
    expected = _reference(sampler_api, epochs=2)
    assert len(expected) > len(GROUP_SIZES)
    assert _through_dataloader(sampler_api, epochs=2, num_workers=num_workers) == expected


@pytest.mark.parametrize("num_workers", [0, 2])
def test_metadata_after_resume_skip(sampler_api, num_workers):
    full = _reference(sampler_api, epochs=1)
    consumed = 2 * NUM_MICROBATCHES
    resumed = _through_dataloader(sampler_api, epochs=1, num_workers=num_workers, consumed_samples=consumed)
    assert resumed == _reference(sampler_api, epochs=1, consumed_samples=consumed)
    assert 0 < len(resumed) < len(full)
    assert resumed == full[len(full) - len(resumed):]
