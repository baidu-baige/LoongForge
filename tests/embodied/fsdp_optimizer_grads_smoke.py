# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Four-GPU FSDP/HSDP step and checkpoint smoke test.

Run explicitly: python -m pytest -q tests/embodied/fsdp_optimizer_grads_smoke.py
"""

from datetime import timedelta
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp

from loongforge.engines.torch.distributed.fsdp_utils.optimizer_grads import (
    clip_optimizer_gradients,
    register_post_step_reshard,
)


def _gpu_worker(rank, rendezvous, checkpoint_dir):
    from torch.distributed.checkpoint import load, save
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.fsdp import fully_shard
    from loongforge.engines.torch.distributed.fsdp_utils.builders import build_fsdp_device_mesh

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=120),
    )
    try:
        for shape in ((4,), (2, 2)):
            mesh = build_fsdp_device_mesh(
                SimpleNamespace(hsdp_shard_size=None if len(shape) == 1 else 2),
                SimpleNamespace(world_size=4),
            )
            torch.manual_seed(123)
            model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4)).cuda()
            dense = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4)).cuda()
            dense.load_state_dict(model.state_dict())
            for layer in model:
                fully_shard(layer, mesh=mesh)
            fully_shard(model, mesh=mesh)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            reference = torch.optim.AdamW(dense.parameters(), lr=0.01)
            hook, count = register_post_step_reshard(model, optimizer)
            assert count == 3
            inputs = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8) / 32
            for step in range(4):
                optimizer.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                model(inputs).square().mean().backward()
                dense(inputs).square().mean().backward()
                expected = torch.nn.utils.clip_grad_norm_(dense.parameters(), 0.1)
                assert clip_optimizer_gradients(optimizer, 0.1) == pytest.approx(expected.item(), rel=1e-5)
                optimizer.step()
                reference.step()
                for parameter, target in zip(model.parameters(), dense.parameters()):
                    torch.testing.assert_close(parameter.full_tensor(), target)
                if step == 1:
                    model_state, optimizer_state = get_state_dict(model, optimizer)
                    state = {"model": model_state, "optimizer": optimizer_state}
                    path = f"{checkpoint_dir}/mesh_{len(shape)}d"
                    save(state, checkpoint_id=path)
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.zero_()
                    optimizer.state.clear()
                    load(state, checkpoint_id=path)
                    set_state_dict(
                        model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"],
                    )
            hook.remove()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA GPUs")
def test_fsdp_gpu_step_and_checkpoint(tmp_path):
    mp.spawn(
        _gpu_worker, args=(f"file://{tmp_path / 'gpu_rdzv'}", str(tmp_path)), nprocs=4, join=True,
    )
