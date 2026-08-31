# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 delta-FP8 AllGather patch, installed from TrainingArgs.

Replaces ``foreach_all_gather`` so BF16 FSDP units communicate a quantized
delta against a persistent unsharded reference instead of the full weight.
Falls back to the stock implementation for world_size 1, non-BF16 units, and
non-default all-gather comms.
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from itertools import chain

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _collectives
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _param_group
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGatherResult,
    DefaultAllGather,
)

from .delta_fp8_comm import (
    DEFAULT_BLOCK,
    dequantize_add,
    quantize_delta_into,
    validate_runtime,
)

logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIGINAL_FOREACH_ALL_GATHER = None
_GROUP_CONFIGS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_STATES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class _GroupConfig:
    """Delta-FP8 settings owned by one FSDP parameter group."""

    block: int
    prime_steps: int
    reprime_interval: int


class _CombinedWork(dist.distributed_c10d.Work):
    """Work handle that completes both delta payload and scale collectives."""

    def __init__(self, works):
        super().__init__()
        self._works = tuple(work for work in works if work is not None)

    def wait(self, timeout=None):
        """Wait for all payload and scale collectives to complete."""
        result = None
        for work in self._works:
            result = work.wait() if timeout is None else work.wait(timeout)
        return result

    def is_completed(self):
        """Return whether every combined collective has completed."""
        return all(work.is_completed() for work in self._works)


def _combine_works(*works):
    """Return one FSDP-compatible handle for the launched collectives."""
    pending = tuple(work for work in works if work is not None)
    if not pending:
        return None
    if len(pending) == 1:
        return pending[0]
    if all(
        isinstance(work, dist.distributed_c10d.Work) for work in pending
    ):
        return _CombinedWork(pending)
    # This branch is only useful for lightweight test doubles. Real NCCL
    # collectives always return dist.Work when async_op=True.
    return pending[0]


class _GroupState:
    """Persistent per-FSDP-unit buffers for delta communication."""

    __slots__ = (
        "shard_numel",
        "world_size",
        "block",
        "num_blocks",
        "reference",
        "shard_buffer",
        "quantized_local",
        "quantized_all",
        "scales_local",
        "scales_all",
        "gathers",
    )

    def __init__(self, shard_numel, world_size, device, block):
        self.shard_numel = shard_numel
        self.world_size = world_size
        self.block = block
        self.num_blocks = (shard_numel + block - 1) // block
        # These persistent buffers avoid reallocating every step, but the
        # unsharded reference plus quantized gather storage increases VRAM.
        self.reference = torch.empty(
            shard_numel * world_size, dtype=torch.bfloat16, device=device
        )
        self.shard_buffer = torch.empty(
            shard_numel, dtype=torch.bfloat16, device=device
        )
        self.quantized_local = torch.empty(
            shard_numel, dtype=torch.uint8, device=device
        )
        self.quantized_all = torch.empty(
            shard_numel * world_size, dtype=torch.uint8, device=device
        )
        self.scales_local = torch.empty(
            self.num_blocks, dtype=torch.float32, device=device
        )
        self.scales_all = torch.empty(
            self.num_blocks * world_size, dtype=torch.float32, device=device
        )
        self.gathers = 0


def _state_for(fsdp_params, shard_numel, world_size, device, block):
    key = fsdp_params[0]
    state = _STATES.get(key)
    if (
        state is None
        or state.shard_numel != shard_numel
        or state.world_size != world_size
        or state.block != block
        or state.reference.device != device
    ):
        state = _GroupState(shard_numel, world_size, device, block)
        _STATES[key] = state
        logger.warning(
            "delta-fp8 all-gather buffers allocated: units=%d shard_numel=%d",
            len(_STATES),
            shard_numel,
        )
    return state


def _delta_foreach_all_gather(
    fsdp_params,
    group,
    async_op,
    all_gather_copy_in_stream,
    all_gather_stream,
    device,
    all_gather_comm,
):
    world_size, rank = group.size(), group.rank()
    config = _GROUP_CONFIGS.get(fsdp_params[0])
    # Preserve native FSDP behavior for unregistered models and unsupported
    # communication paths. Registration is per group, so installing this
    # process-level hook does not enable Delta-FP8 for later FSDP models.
    if (
        config is None
        or world_size == 1
        or type(all_gather_comm) is not DefaultAllGather
    ):
        return _ORIGINAL_FOREACH_ALL_GATHER(
            fsdp_params,
            group,
            async_op,
            all_gather_copy_in_stream,
            all_gather_stream,
            device,
            all_gather_comm,
        )
    block = config.block
    prime_steps = config.prime_steps
    reprime_interval = config.reprime_interval
    device_handle = _get_device_handle(device.type)
    with device_handle.stream(all_gather_copy_in_stream):
        param_all_gather_inputs = _collectives._get_param_all_gather_inputs(fsdp_params)
        (
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            dtype,
        ) = _collectives._get_all_gather_input_metadatas(param_all_gather_inputs)
        if dtype is not torch.bfloat16:
            del param_all_gather_inputs
            return _ORIGINAL_FOREACH_ALL_GATHER(
                fsdp_params,
                group,
                async_op,
                all_gather_copy_in_stream,
                all_gather_stream,
                device,
                all_gather_comm,
            )
        all_gather_inputs = [*chain.from_iterable(param_all_gather_inputs)]
        inp_split_sizes = [t.numel() for t in all_gather_inputs]
        shard_numel = sum(inp_split_sizes)
        state = _state_for(fsdp_params, shard_numel, world_size, device, block)
        torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs, state.shard_buffer, inp_split_sizes, shard_numel, 0
        )
        del param_all_gather_inputs
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with device_handle.stream(all_gather_stream):
        # Prime with a full BF16 gather first; later gathers transmit only the
        # local delta from the rank's slice of the reconstructed reference.
        # Even with prime_steps=0, the first gather must initialize the
        # reference with exact BF16 data before any delta is quantized.
        use_delta = state.gathers > 0 and state.gathers >= prime_steps
        if use_delta and reprime_interval > 0:
            use_delta = state.gathers % reprime_interval != 0
        if use_delta:
            quantize_delta_into(
                state.shard_buffer,
                state.reference.narrow(0, rank * shard_numel, shard_numel),
                state.quantized_local,
                state.scales_local,
                block,
            )
            all_gather_work = all_gather_comm(
                output_tensor=state.quantized_all,
                input_tensor=state.quantized_local,
                group=group,
                async_op=async_op,
            )
            # Scales are gathered separately because the payload is uint8.
            scale_work = dist.all_gather_into_tensor(
                state.scales_all,
                state.scales_local,
                group=group,
                async_op=async_op,
            )
            dequantize_add(
                state.reference,
                state.quantized_all,
                state.scales_all,
                shard_numel,
                world_size,
                block,
            )
            all_gather_work = _combine_works(all_gather_work, scale_work)
        else:
            # This is the initial prime, or an explicit periodic re-prime.
            all_gather_work = all_gather_comm(
                output_tensor=state.reference,
                input_tensor=state.shard_buffer,
                group=group,
                async_op=async_op,
            )
        state.gathers += 1
        all_gather_event = all_gather_stream.record_event()
    return AllGatherResult(
        state.reference,
        all_gather_event,
        all_gather_work,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        inp_split_sizes,
    )


def _validate_config(block: int, prime_steps: int, reprime_interval: int) -> None:
    if block <= 0 or block & (block - 1):
        raise ValueError(
            "fsdp_delta_fp8_block must be a positive power of two, "
            f"got {block}"
        )
    if block > 1 << 20:
        raise ValueError(
            "fsdp_delta_fp8_block must be <= 1048576 for Triton tl.arange, "
            f"got {block}"
        )
    if prime_steps < 0:
        raise ValueError(
            f"fsdp_delta_fp8_prime_steps must be >= 0, got {prime_steps}"
        )
    if reprime_interval < 0:
        raise ValueError(
            "fsdp_delta_fp8_reprime_interval must be >= 0, got "
            f"{reprime_interval}"
        )


def _install_delta_fp8_allgather() -> None:
    """Install the process-level dispatch hook once."""
    global _INSTALLED, _ORIGINAL_FOREACH_ALL_GATHER
    if _INSTALLED:
        return
    _ORIGINAL_FOREACH_ALL_GATHER = _collectives.foreach_all_gather
    _collectives.foreach_all_gather = _delta_foreach_all_gather
    _param_group.foreach_all_gather = _delta_foreach_all_gather
    _INSTALLED = True


def register_delta_fp8_allgather(
    model,
    *,
    block: int = DEFAULT_BLOCK,
    prime_steps: int = 1,
    reprime_interval: int = 0,
) -> int:
    """Enable Delta-FP8 only for the FSDP groups owned by ``model``."""
    block = int(block)
    prime_steps = int(prime_steps)
    reprime_interval = int(reprime_interval)
    _validate_config(block, prime_steps, reprime_interval)
    config = _GroupConfig(block, prime_steps, reprime_interval)
    param_groups = []
    seen_states: set[int] = set()
    for module in model.modules():
        get_state = getattr(module, "_get_fsdp_state", None)
        if not callable(get_state):
            continue
        state = get_state()
        if id(state) in seen_states:
            continue
        seen_states.add(id(state))
        for param_group in state._fsdp_param_groups:
            if not param_group.fsdp_params:
                continue
            param_groups.append(param_group)

    if not param_groups:
        raise RuntimeError(
            "Cannot enable Delta-FP8 AllGather: model has no FSDP parameter groups"
        )
    for param_group in param_groups:
        process_group = param_group._all_gather_process_group
        try:
            backend = dist.get_backend(process_group)
        except Exception as exc:
            raise RuntimeError(
                "Unable to determine the FSDP all-gather backend for "
                f"--fsdp-delta-fp8-allgather (device={param_group.device}, "
                f"process_group={type(process_group).__name__})"
            ) from exc
        validate_runtime(param_group.device, backend)

    _install_delta_fp8_allgather()
    for param_group in param_groups:
        _GROUP_CONFIGS[param_group.fsdp_params[0]] = config
    registered = len(param_groups)
    logger.warning(
        "FSDP2 delta-FP8 all-gather registered for %d groups "
        "(block=%d, prime_steps=%d, reprime_interval=%d)",
        registered,
        block,
        prime_steps,
        reprime_interval,
    )
    return registered


def uninstall_delta_fp8_allgather() -> None:
    """Restore the stock FSDP2 ``foreach_all_gather``. Tests only."""
    global _INSTALLED, _ORIGINAL_FOREACH_ALL_GATHER
    if not _INSTALLED:
        return
    _collectives.foreach_all_gather = _ORIGINAL_FOREACH_ALL_GATHER
    _param_group.foreach_all_gather = _ORIGINAL_FOREACH_ALL_GATHER
    _ORIGINAL_FOREACH_ALL_GATHER = None
    _GROUP_CONFIGS.clear()
    _STATES.clear()
    _INSTALLED = False
