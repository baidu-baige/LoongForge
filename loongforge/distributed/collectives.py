# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Detached metric reductions over an engine-selected process group."""

import torch.distributed as dist


def all_reduce_mean(value, group=None):
    result = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, group=group)
        result /= dist.get_world_size(group)
    return result
