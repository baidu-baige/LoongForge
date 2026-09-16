# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Cross-engine process identity and metric reductions.

Engines keep ownership of their process groups: this module only reports the
launcher-provided identity and provides detached reductions. DDP/FSDP live in
``loongforge.engine.torch.distributed``, TP/PP/CP/VPP in ``loongforge.engine.mcore``.
"""

import os


def _env_value(name, mpi_name, default):
    mpi_value = os.environ.get(mpi_name, "-1")
    return int(os.environ.get(name, default) if mpi_value == "-1" else mpi_value)


def rank():
    """Rank of this process, from torch.distributed when initialized."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _env_value("RANK", "OMPI_COMM_WORLD_RANK", "0")


def world_size():
    """Number of processes, from torch.distributed when initialized."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return _env_value("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "1")


def local_rank():
    """Rank of this process on its node, as exported by the launcher."""
    return _env_value("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "0")


def is_rank_zero():
    """Whether this process is the first rank of its world."""
    return rank() == 0


def all_reduce_mean(value, group=None):
    """Mean of ``value`` over ``group``, detached from the autograd graph."""
    import torch.distributed as dist

    result = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, group=group)
        result /= dist.get_world_size(group)
    return result
