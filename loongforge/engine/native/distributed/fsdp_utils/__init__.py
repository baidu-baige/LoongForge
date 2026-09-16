# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 wrapping steps, split by concern.

The wrap pass itself lives in ``distributed.parallel._wrap_fsdp``, which drives
these modules in order:

* ``units``      — which modules become FSDP units, grouped into prefetch runs
* ``context``    — per-pass configuration shared by every group
* ``builders``   — device mesh, mixed precision policy, ignored params
* ``sharding``   — creation of one group, plus its reshard policy
* ``mixed_dtype``— splitting minority-dtype subtrees into their own groups
* ``inspection`` — read-only module/parameter queries used by the passes above
* ``prefetch``   — prefetch edges along each run
"""

from .builders import build_fsdp_device_mesh, build_ignored_params, build_mp_policy
from .context import FSDPWrapContext
from .inspection import (
    find_fsdp_root_module,
    get_fsdp_root_sharded_params,
    group_numel_by_dtype,
    is_valid_fsdp_wrap_target,
    managed_param_numel,
)
from .prefetch import configure_prefetch
from .sharding import fully_shard_unit, resolve_reshard_policy
from .units import FSDPWrapRun, resolve_wrap_runs

__all__ = [
    "FSDPWrapContext",
    "FSDPWrapRun",
    "build_fsdp_device_mesh",
    "build_ignored_params",
    "build_mp_policy",
    "configure_prefetch",
    "find_fsdp_root_module",
    "fully_shard_unit",
    "get_fsdp_root_sharded_params",
    "group_numel_by_dtype",
    "is_valid_fsdp_wrap_target",
    "managed_param_numel",
    "resolve_reshard_policy",
    "resolve_wrap_runs",
]
