# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Distributed environment context, replacing HuggingFace Accelerator."""

import os

import torch
import torch.distributed as dist


class DistributedContext:
    """Global distributed context — single instance throughout training lifetime."""

    def __init__(self, backend: str = "cpu:gloo,cuda:nccl"):
        self.backend = backend
        self.rank: int = 0
        self.local_rank: int = 0
        self.world_size: int = 1
        self.device: torch.device = torch.device("cpu")
        self._initialized: bool = False

    def init(self):
        """Initialize from torchrun environment variables."""
        if "RANK" not in os.environ:
            # Single-card mode
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(self.device)
            return

        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend=self.backend, device_id=self.device)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._initialized = True

    @property
    def is_main(self) -> bool:
        """Whether current process is rank 0."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """Whether distributed training is active."""
        return self._initialized

    def barrier(self):
        """Synchronize all processes."""
        if self._initialized:
            dist.barrier()

    def destroy(self):
        """Clean up distributed process group."""
        if self._initialized:
            dist.destroy_process_group()
            self._initialized = False

    def all_reduce_mean(self, value):
        """Average a scalar across all ranks.

        Collective op — EVERY rank must call this, otherwise NCCL will hang.
        On single-card / non-distributed runs it is a pass-through.

        Args:
            value: A python number or a 0-dim / single-element tensor.

        Returns:
            The cross-rank mean as a python float.
        """
        if not self._initialized or self.world_size == 1:
            return float(value.item()) if isinstance(value, torch.Tensor) else float(value)

        if isinstance(value, torch.Tensor):
            t = value.detach().to(self.device, dtype=torch.float32)
        else:
            t = torch.tensor(float(value), dtype=torch.float32, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= self.world_size
        return t.item()
