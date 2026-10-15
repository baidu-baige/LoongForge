# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU (gloo, world_size=2) tests for ``CustomDpOptimizer``."""

from __future__ import annotations

import json
import os
import socket
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from loongforge.embodied.distributed.custom_dp_utils.precision_policy import (
    DefaultParameterPolicy,
    ParameterClassPolicy,
)
from loongforge.embodied.train.trainers.custom.dp_training.custom_dp_optimizer import (
    CollectiveConfig,
    CustomDpOptimizer,
)
from loongforge.embodied.train.trainers.custom.dp_training.optimizer_builder import (
    build_muon_optimizer,
)

WORLD_SIZE = 2
GRAD_ACCUM = 2
LR = 1e-2


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run(worker, *args):
    """Run ``worker(rank, *args)`` on two gloo ranks; fork keeps it CPU-only."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    mp.start_processes(
        _bootstrap, args=(worker, args), nprocs=WORLD_SIZE, start_method="fork"
    )


def _bootstrap(rank, worker, args):
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        worker(rank, *args)
    finally:
        dist.destroy_process_group()


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 4))


def _batch(step, micro, rank):
    generator = torch.Generator().manual_seed(1000 * step + 10 * micro + rank)
    return torch.randn(3, 8, generator=generator), torch.randn(3, 4, generator=generator)


def _adamw_factory(dp):
    return torch.optim.AdamW(
        [p for _, p in dp.optimizer_view().named_parameters()], lr=LR, foreach=False
    )


def _muon_factory(dp):
    muon, adamw = dp.parameter_plan.split_masters_by_optimizer()
    return build_muon_optimizer(muon, lr=LR, adamw_parameters=adamw)


def _build(model, factory, policy=None, **config):
    config.setdefault("gradient_reduce_dtype", "fp32")
    return CustomDpOptimizer(
        model,
        policy or DefaultParameterPolicy(),
        factory,
        rank=dist.get_rank(),
        world_size=WORLD_SIZE,
        collective_config=CollectiveConfig(**config),
    )


def _train(model, optimizer, steps, start=0, clip=1.0):
    norms = []
    dtype = next(model.parameters()).dtype
    for step in range(start, start + steps):
        optimizer.zero_grad()
        for micro in range(GRAD_ACCUM):
            if micro == GRAD_ACCUM - 1:
                optimizer.prepare_for_backward()
            x, y = _batch(step, micro, dist.get_rank())
            loss = nn.functional.mse_loss(model(x.to(dtype)).float(), y) / GRAD_ACCUM
            loss.backward()
        optimizer.scrub_nan_gradients()
        norms.append(optimizer.clip_grad_norm_(clip))
        optimizer.step()
    return norms


def _replicas(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _assert_replicas_agree(model):
    for name, parameter in model.named_parameters():
        gathered = [torch.empty_like(parameter) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered, parameter.detach().contiguous())
        assert torch.equal(gathered[0], gathered[1]), name


# -- data-parallel equivalence ------------------------------------------------


def _worker_matches_reference(rank, overlap):
    model = _model()
    optimizer = _build(model, _adamw_factory, grad_overlap=overlap, param_overlap=overlap)
    norms = _train(model, optimizer, steps=3, clip=0.0)

    # Reference: one process, full fp32 AdamW over the mean gradient of every
    # rank's micro-batches.
    reference = _model()
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=LR, foreach=False)
    ref_norms = []
    for step in range(3):
        ref_opt.zero_grad()
        for micro in range(GRAD_ACCUM):
            for source_rank in range(WORLD_SIZE):
                x, y = _batch(step, micro, source_rank)
                loss = nn.functional.mse_loss(reference(x), y)
                (loss / (GRAD_ACCUM * WORLD_SIZE)).backward()
        ref_norms.append(
            float(torch.norm(torch.stack([p.grad.norm() for p in reference.parameters()])))
        )
        ref_opt.step()

    assert norms == pytest.approx(ref_norms, rel=1e-5)
    for (name, got), want in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6, msg=name)
    _assert_replicas_agree(model)


@pytest.mark.parametrize("overlap", [True, False])
def test_matches_single_process_reference(overlap):
    _run(_worker_matches_reference, overlap)


# -- Muon route, bf16 compute, LR scheduler ------------------------------------


def _worker_muon_bf16(rank):
    model = _model()
    # No markers: bf16 compute replicas, matrices on Muon, 1-D tensors on AdamW.
    optimizer = _build(
        model, _muon_factory, ParameterClassPolicy(),
        gradient_reduce_dtype="bf16", parameter_sync_precision="bf16",
    )
    assert optimizer.in_step_names == set(optimizer.parameter_plan.muon_parameter_names())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** step)
    before = _replicas(model)
    for step in range(3):
        _train(model, optimizer, steps=1, start=step)
        scheduler.step()
    assert all(group["lr"] == LR * 0.125 for group in optimizer.optim.param_groups)
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters())
    for name, master in optimizer.master.items():
        replica = dict(model.named_parameters())[name]
        assert torch.equal(master.to(replica.dtype), replica.detach()), name
    _assert_replicas_agree(model)


def test_muon_route_publishes_bf16_replicas():
    _run(_worker_muon_bf16)


# -- rank-local checkpoint round trip ------------------------------------------


def _worker_checkpoint(rank, state_dir):
    ctx = SimpleNamespace(
        rank=rank, world_size=WORLD_SIZE, is_main=rank == 0, barrier=dist.barrier
    )
    metadata_path = os.path.join(state_dir, "metadata.json")

    model = _model()
    optimizer = _build(model, _muon_factory, ParameterClassPolicy())
    _train(model, optimizer, steps=2)
    optimizer.zero_checkpoint_io.save_local_state(state_dir, metadata_path, ctx)
    _train(model, optimizer, steps=2, start=2)
    uninterrupted = _replicas(model)

    resumed_model = _model()
    resumed = _build(resumed_model, _muon_factory, ParameterClassPolicy())
    with open(metadata_path, encoding="utf-8") as file:
        metadata = json.load(file)
    resumed.zero_checkpoint_io.load_local_state(state_dir, metadata, ctx)
    # Replicas come from the published masters, not from the module state.
    _train(resumed_model, resumed, steps=2, start=2)
    for name, parameter in resumed_model.named_parameters():
        assert torch.equal(parameter.detach(), uninterrupted[name]), name


def test_checkpoint_round_trip_is_bitwise():
    with tempfile.TemporaryDirectory() as state_dir:
        _run(_worker_checkpoint, state_dir)
