"""Smoke tests for the PyTorch-internal DDP reducer binding."""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from groot_n1_7_op.groot_ddp_reducer_bucket_control import get_buckets, initialize_buckets


def test_reducer_bucket_control_single_process():
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    fd, init_file = tempfile.mkstemp(prefix="groot_n1_7_ddp_")
    os.close(fd)
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=0, world_size=1
        )
        model = DistributedDataParallel(torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)
        ))
        reducer = model.reducer
        initialize_buckets(reducer, [[0, 1], [2, 3]])
        buckets = get_buckets(reducer)
        assert len(buckets) == 2
        assert all(bucket.buffer().numel() > 0 for bucket in buckets)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        try:
            os.unlink(init_file)
        except FileNotFoundError:
            pass
