# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Global process identity; engines retain ownership of process groups."""

import os


def _env_value(name, mpi_name, default):
    mpi_value = os.environ.get(mpi_name, "-1")
    return int(os.environ.get(name, default) if mpi_value == "-1" else mpi_value)


def rank():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _env_value("RANK", "OMPI_COMM_WORLD_RANK", "0")


def world_size():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return _env_value("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "1")


def local_rank():
    return _env_value("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "0")


def is_rank_zero():
    return rank() == 0
