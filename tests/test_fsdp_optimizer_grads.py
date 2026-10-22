# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

from loongforge.engines.torch.distributed.fsdp_utils.optimizer_grads import (
    clean_optimizer_gradients,
    clip_optimizer_gradients,
)


def _check_layout(mesh, placements):
    tensors = [torch.arange(1.0, 17.0), torch.tensor([3.0, -4.0, 0.0, 1.0])]
    params = [torch.nn.Parameter(distribute_tensor(torch.zeros_like(t), mesh, placements)) for t in tensors]
    for parameter, tensor in zip(params, tensors):
        parameter.grad = distribute_tensor(tensor, mesh, placements)
    # Shared parameters must be clipped once even if two groups list them.
    optimizer = SimpleNamespace(param_groups=[{"params": params}, {"params": [params[0]]}])
    dense = [torch.nn.Parameter(torch.zeros_like(t)) for t in tensors]
    for parameter, tensor in zip(dense, tensors):
        parameter.grad = tensor.clone()
    expected_norm = torch.nn.utils.clip_grad_norm_(dense, 2.0)
    norm = clip_optimizer_gradients(optimizer, 2.0)
    assert norm == pytest.approx(expected_norm.item(), rel=1e-6)
    for parameter, reference in zip(params, dense):
        torch.testing.assert_close(parameter.grad.full_tensor(), reference.grad)
    for parameter in params:
        parameter.grad = None
    assert clip_optimizer_gradients(optimizer, 2.0) == 0.0
    params[0].grad = distribute_tensor(torch.full_like(tensors[0], float("nan")), mesh, placements)
    clean_optimizer_gradients(optimizer)
    assert torch.count_nonzero(params[0].grad.to_local()) == 0


def _distributed_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60),
    )
    try:
        fsdp = init_device_mesh("cpu", (4,))
        _check_layout(fsdp, [Shard(0)])
        hsdp = init_device_mesh("cpu", (2, 2), mesh_dim_names=("replica", "shard"))
        _check_layout(hsdp, [Replicate(), Shard(0)])
        _check_layout(fsdp, [Replicate()])
        assert clip_optimizer_gradients(SimpleNamespace(param_groups=[]), 1.0) == 0.0
        partial = torch.nn.Parameter(DTensor.from_local(torch.ones(4), fsdp, [Partial()]))
        with pytest.raises(RuntimeError, match="reduced Shard/Replicate"):
            clip_optimizer_gradients(SimpleNamespace(param_groups=[{"params": [partial]}]), 1.0)
        params = [
            torch.nn.Parameter(distribute_tensor(torch.zeros(8), fsdp, placements))
            for placements in ([Shard(0)], [Replicate()])
        ]
        with pytest.raises(RuntimeError, match="one shared DTensor mesh"):
            clip_optimizer_gradients(SimpleNamespace(param_groups=[{"params": params}]), 1.0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_fsdp_and_hsdp_clipping_match_dense_gradients(tmp_path):
    mp.spawn(_distributed_worker, args=(f"file://{tmp_path / 'rdzv'}",), nprocs=4, join=True)
