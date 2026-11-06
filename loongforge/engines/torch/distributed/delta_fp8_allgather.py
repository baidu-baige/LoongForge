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
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _collectives
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _param_group
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGatherResult,
    DefaultAllGather,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.fsdp._fully_shard._fsdp_common import _from_local_no_grad

from .delta_fp8_comm import (
    DEFAULT_BLOCK,
    dequantize_add,
    dequantize_add_param_major,
    quantize_delta_into,
    quantize_delta_param_major_into,
    require_triton,
    validate_runtime,
)

logger = logging.getLogger(__name__)

_CONFIG = {
    "block": DEFAULT_BLOCK,
    "prime_steps": 1,
    "reprime_interval": 0,
}
_INSTALLED = False
_ORIGINAL_FOREACH_ALL_GATHER = None
_ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT = None
_ORIGINAL_FREE_UNSHARDED_PARAM = None
_ORIGINAL_INIT_UNSHARDED_PARAM = None
_STATES: dict[int, "_GroupState"] = {}
_SCRATCH_STATES: dict[tuple[str, int | None, int], "_ScratchState"] = {}
_ALIASED_RESULTS: dict[int, "_GroupState"] = {}
_GROUP_CONFIGS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_MODEL_SCOPED = False


@dataclass(frozen=True)
class _GroupConfig:
    """Delta-FP8 settings owned by one FSDP parameter group."""

    block: int
    prime_steps: int
    reprime_interval: int


class _GroupState:
    """Persistent per-FSDP-unit buffers for delta communication."""

    __slots__ = (
        "shard_numel",
        "world_size",
        "num_blocks",
        "block",
        "reference",
        "block_metadata",
        "param_shard_numels",
        "fsdp_params",
        "shard_buffer",
        "gathers",
    )

    def __init__(
        self,
        shard_numel,
        world_size,
        device,
        block,
        fsdp_params=(),
        param_shard_numels=(),
    ):
        self.shard_numel = shard_numel
        self.world_size = world_size
        self.block = block
        self.param_shard_numels = tuple(param_shard_numels)
        self.fsdp_params = tuple(fsdp_params) if self.param_shard_numels else ()
        self.block_metadata = None
        if self.fsdp_params:
            self.num_blocks = sum(
                (numel + block - 1) // block for numel in self.param_shard_numels
            )
        else:
            self.num_blocks = (shard_numel + block - 1) // block
        self.reference = torch.empty(
            shard_numel * world_size, dtype=torch.bfloat16, device=device
        )
        if self.fsdp_params:
            self.block_metadata = _build_param_major_block_metadata(
                self.param_shard_numels,
                world_size,
                block,
                device,
            )
            reference_offset = 0
            for fsdp_param, param_shard_numel in zip(
                self.fsdp_params, self.param_shard_numels
            ):
                output_numel = param_shard_numel * world_size
                fsdp_param.all_gather_outputs = [
                    self.reference.narrow(0, reference_offset, output_numel)
                ]
                fsdp_param._delta_fp8_persistent_reference = True
                reference_offset += output_numel
            _ALIASED_RESULTS[id(self.reference)] = self
        # The copy buffer is allocated lazily only when FSDP does not expose a
        # contiguous flat input that can be reused directly.
        self.shard_buffer = None
        self.gathers = 0


class _ScratchState:
    """Reusable quantization buffers for one device and AllGather stream."""

    __slots__ = (
        "stream",
        "quantized_local",
        "quantized_all",
        "scales_local",
        "scales_all",
    )

    def __init__(self, stream):
        self.stream = stream
        self.quantized_local = None
        self.quantized_all = None
        self.scales_local = None
        self.scales_all = None

    def ensure(self, shard_numel, world_size, num_blocks, device):
        """Grow buffers on demand; callers use narrow views for each unit."""
        if (
            self.quantized_local is None
            or self.quantized_local.numel() < shard_numel
        ):
            self.quantized_local = torch.empty(
                shard_numel, dtype=torch.uint8, device=device
            )
        if (
            self.quantized_all is None
            or self.quantized_all.numel() < shard_numel * world_size
        ):
            self.quantized_all = torch.empty(
                shard_numel * world_size, dtype=torch.uint8, device=device
            )
        if self.scales_local is None or self.scales_local.numel() < num_blocks:
            self.scales_local = torch.empty(
                num_blocks, dtype=torch.float32, device=device
            )
        if (
            self.scales_all is None
            or self.scales_all.numel() < num_blocks * world_size
        ):
            self.scales_all = torch.empty(
                num_blocks * world_size, dtype=torch.float32, device=device
            )

    def views(self, shard_numel, world_size, num_blocks):
        """Return exact-size contiguous views for the current FSDP unit."""
        return (
            self.quantized_local.narrow(0, 0, shard_numel),
            self.quantized_all.narrow(0, 0, shard_numel * world_size),
            self.scales_local.narrow(0, 0, num_blocks),
            self.scales_all.narrow(0, 0, num_blocks * world_size),
        )


def _build_param_major_block_metadata(
    param_shard_numels,
    world_size,
    block,
    device,
):
    """Map local parameter blocks to their offsets in parameter-major output."""
    reference_numel = sum(param_shard_numels) * world_size
    if reference_numel > torch.iinfo(torch.int32).max:
        raise ValueError(
            "delta-fp8 parameter-major reference exceeds int32 indexing: "
            f"{reference_numel} elements"
        )
    num_blocks = sum((numel + block - 1) // block for numel in param_shard_numels)
    metadata = torch.empty((num_blocks, 3), dtype=torch.int32)
    block_cursor = 0
    shard_offset = 0
    reference_offset = 0
    for param_shard_numel in param_shard_numels:
        param_blocks = (param_shard_numel + block - 1) // block
        offsets = torch.arange(0, param_shard_numel, block, dtype=torch.int32)
        block_slice = metadata.narrow(0, block_cursor, param_blocks)
        block_slice[:, 0] = offsets + shard_offset
        block_slice[:, 1] = offsets + reference_offset
        block_slice[:, 2] = param_shard_numel
        block_cursor += param_blocks
        shard_offset += param_shard_numel
        reference_offset += param_shard_numel * world_size
    return metadata.to(device=device)


def _scratch_key(stream, device):
    stream_id = getattr(stream, "stream_id", None)
    if stream_id is None:
        stream_id = id(stream)
    return (device.type, device.index, int(stream_id))


def _scratch_for(stream, shard_numel, world_size, num_blocks, device):
    key = _scratch_key(stream, device)
    scratch = _SCRATCH_STATES.get(key)
    if scratch is None or scratch.stream is not stream:
        scratch = _ScratchState(stream)
        _SCRATCH_STATES[key] = scratch
    scratch.ensure(shard_numel, world_size, num_blocks, device)
    return scratch.views(shard_numel, world_size, num_blocks)


def _as_contiguous_flat_input(all_gather_inputs, shard_numel):
    """Alias inputs that are adjacent views of one contiguous allocation."""
    if not all_gather_inputs or shard_numel <= 0:
        return None
    first = all_gather_inputs[0]
    if not first.is_contiguous():
        return None
    storage = first.untyped_storage()
    storage_ptr = storage.data_ptr()
    expected_offset = first.storage_offset()
    dtype = first.dtype
    device = first.device
    for tensor in all_gather_inputs:
        if (
            tensor.dtype is not dtype
            or tensor.device != device
            or not tensor.is_contiguous()
            or tensor.untyped_storage().data_ptr() != storage_ptr
            or tensor.storage_offset() != expected_offset
        ):
            return None
        expected_offset += tensor.numel()
    if expected_offset - first.storage_offset() != shard_numel:
        return None
    try:
        return first.as_strided(
            (shard_numel,), (1,), storage_offset=first.storage_offset()
        )
    except RuntimeError:
        return None


def _get_shard_input(state, all_gather_inputs, inp_split_sizes, device):
    flat_input = _as_contiguous_flat_input(all_gather_inputs, state.shard_numel)
    if flat_input is not None:
        return flat_input, True
    if state.shard_buffer is None:
        state.shard_buffer = torch.empty(
            state.shard_numel, dtype=torch.bfloat16, device=device
        )
        logger.info(
            "delta-fp8 flat-input reuse fell back to a copy buffer: shard_numel=%d",
            state.shard_numel,
        )
    torch.ops.fsdp.all_gather_copy_in(
        all_gather_inputs,
        state.shard_buffer,
        inp_split_sizes,
        state.shard_numel,
        0,
    )
    return state.shard_buffer, False


def _block_current_stream_on_work(work) -> None:
    """Make the current stream wait for an async collective without a CPU wait."""
    if work is None:
        return
    block_current_stream = getattr(work, "block_current_stream", None)
    if block_current_stream is not None:
        block_current_stream()
    else:
        # Keep mocked/legacy Work implementations correct even without the
        # stream-aware API. Real c10d Work objects take the non-blocking path.
        work.wait()


def _launch_delta_collectives(
    all_gather_comm,
    quantized_all,
    quantized_local,
    scales_all,
    scales_local,
    group,
    async_op,
):
    """Launch payload and scale collectives and order the stream-side decode."""
    payload_work = all_gather_comm(
        output_tensor=quantized_all,
        input_tensor=quantized_local,
        group=group,
        async_op=async_op,
    )
    scale_work = dist.all_gather_into_tensor(
        scales_all,
        scales_local,
        group=group,
        async_op=async_op,
    )
    # ``dequantize_add`` is queued below on this stream. Both collectives must
    # therefore be ordered on the stream before the kernel reads their output.
    # Return only payload_work because FSDP2's AllGatherResult has one Work
    # slot; the scale work is covered by this stream dependency and event.
    _block_current_stream_on_work(payload_work)
    _block_current_stream_on_work(scale_work)
    return payload_work


def _supports_unsharded_param_reuse(
    fsdp_params,
    param_all_gather_inputs,
    param_all_gather_input_dtypes,
    param_all_gather_input_numels,
):
    """Return whether this group has the standard dim-0 FSDP tensor layout."""
    for fsdp_param, inputs, dtypes, numels in zip(
        fsdp_params,
        param_all_gather_inputs,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
    ):
        if (
            len(inputs) != 1
            or len(dtypes) != 1
            or len(numels) != 1
            or dtypes[0] is not torch.bfloat16
            or fsdp_param.fsdp_placement.dim != 0
            or fsdp_param.post_forward_mesh_info is not None
            or len(fsdp_param.all_gather_outputs) != 0
            or hasattr(fsdp_param._sharded_local_tensor, "fsdp_post_all_gather")
        ):
            return False
    return True


def _launch_param_major_prime(
    all_gather_comm,
    group,
    shard_input,
    parameter_major,
    param_shard_numels,
    world_size,
    async_op,
):
    """Gather parameter shards directly into persistent parameter-major storage."""
    last_work = None
    reference_offset = 0
    for input_tensor, param_shard_numel in zip(
        shard_input.split(param_shard_numels), param_shard_numels
    ):
        output_tensor = parameter_major.narrow(
            0, reference_offset, param_shard_numel * world_size
        )
        last_work = all_gather_comm(
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            group=group,
            async_op=async_op,
        )
        _block_current_stream_on_work(last_work)
        reference_offset += param_shard_numel * world_size
    return last_work


def _delta_foreach_all_gather_copy_out(all_gather_result, fsdp_params, group):
    """Skip the stock copy when the result already owns FSDP output storage."""
    state = _ALIASED_RESULTS.get(id(all_gather_result.all_gather_output))
    if state is None or not state.fsdp_params:
        return _ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT(
            all_gather_result, fsdp_params, group
        )
    if not fsdp_params or id(fsdp_params[0]) != id(state.fsdp_params[0]):
        raise RuntimeError("delta-fp8 aliased AllGatherResult used by another FSDP group")
    device_handle = _get_device_handle(all_gather_result.all_gather_output.device.type)
    if all_gather_result.all_gather_event is not None:
        device_handle.current_stream().wait_event(all_gather_result.all_gather_event)
    work = all_gather_result.all_gather_work
    if isinstance(work, dist.distributed_c10d.Work):
        work.wait()


def _delta_free_unsharded_param(fsdp_param):
    """Keep aliased FSDP outputs allocated while switching back to shards."""
    if getattr(fsdp_param, "_delta_fp8_persistent_reference", False):
        return
    return _ORIGINAL_FREE_UNSHARDED_PARAM(fsdp_param)


def _delta_init_unsharded_param(fsdp_param):
    """Preserve each output view's offset inside the shared reference storage."""
    was_initialized = hasattr(fsdp_param, "_unsharded_param")
    result = _ORIGINAL_INIT_UNSHARDED_PARAM(fsdp_param)
    if (
        was_initialized
        or not getattr(fsdp_param, "_delta_fp8_persistent_reference", False)
    ):
        return result
    output = fsdp_param.all_gather_outputs[0]
    unsharded_tensor = torch.as_strided(
        output,
        fsdp_param._orig_size,
        fsdp_param._contiguous_orig_stride,
        storage_offset=output.storage_offset(),
    )
    if fsdp_param._unsharded_dtensor_spec is not None:
        unsharded_tensor = _from_local_no_grad(
            unsharded_tensor, fsdp_param._unsharded_dtensor_spec
        )
    fsdp_param._unsharded_param = nn.Parameter(
        unsharded_tensor,
        requires_grad=fsdp_param.sharded_param.requires_grad,
    )
    return result


def _state_for(
    fsdp_params,
    param_all_gather_inputs,
    param_all_gather_input_dtypes,
    param_all_gather_input_numels,
    shard_numel,
    world_size,
    device,
    block,
):
    key = id(fsdp_params[0])
    state = _STATES.get(key)
    if (
        state is None
        or state.shard_numel != shard_numel
        or state.world_size != world_size
        or state.block != block
        or state.reference.device != device
    ):
        alias_supported = _supports_unsharded_param_reuse(
            fsdp_params,
            param_all_gather_inputs,
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
        )
        if not alias_supported:
            logger.warning(
                "delta-fp8 unsharded-parameter reuse is unsupported for this "
                "FSDP group; falling back to a separate reference"
            )
        param_shard_numels = tuple(
            numels[0] for numels in param_all_gather_input_numels
        ) if alias_supported else ()
        state = _GroupState(
            shard_numel,
            world_size,
            device,
            block,
            fsdp_params if alias_supported else (),
            param_shard_numels if alias_supported else (),
        )
        _STATES[key] = state
        logger.warning(
            "delta-fp8 all-gather buffers allocated: units=%d shard_numel=%d "
            "memory_optimizations=default parameter_major_reference=%s",
            len(_STATES),
            shard_numel,
            bool(state.fsdp_params),
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
        world_size == 1
        or type(all_gather_comm) is not DefaultAllGather
        or (_MODEL_SCOPED and config is None)
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
    if config is None:
        block = _CONFIG["block"]
        prime_steps = _CONFIG["prime_steps"]
        reprime_interval = _CONFIG["reprime_interval"]
    else:
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
        state = _state_for(
            fsdp_params,
            param_all_gather_inputs,
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            shard_numel,
            world_size,
            device,
            block,
        )
        shard_input, reused_flat_input = _get_shard_input(
            state, all_gather_inputs, inp_split_sizes, device
        )
        if reused_flat_input and shard_input.device.type == "cuda":
            # _get_param_all_gather_inputs allocates this buffer on the copy-in
            # stream. Its consumers run on the AllGather stream after this
            # function returns, so retain the allocation until that stream is done.
            shard_input.record_stream(all_gather_stream)
        del param_all_gather_inputs
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with device_handle.stream(all_gather_stream):
        use_delta = state.gathers >= prime_steps
        if use_delta and reprime_interval > 0:
            use_delta = state.gathers % reprime_interval != 0
        if use_delta:
            (
                quantized_local,
                quantized_all,
                scales_local,
                scales_all,
            ) = _scratch_for(
                all_gather_stream,
                shard_numel,
                world_size,
                state.num_blocks,
                device,
            )
            if state.block_metadata is not None:
                quantize_delta_param_major_into(
                    shard_input,
                    state.reference,
                    quantized_local,
                    scales_local,
                    state.block_metadata,
                    rank,
                    block,
                )
            else:
                quantize_delta_into(
                    shard_input,
                    state.reference.narrow(0, rank * shard_numel, shard_numel),
                    quantized_local,
                    scales_local,
                    block,
                )
            payload_work = _launch_delta_collectives(
                all_gather_comm,
                quantized_all,
                quantized_local,
                scales_all,
                scales_local,
                group,
                async_op,
            )
            if state.block_metadata is not None:
                dequantize_add_param_major(
                    state.reference,
                    quantized_all,
                    scales_all,
                    state.block_metadata,
                    shard_numel,
                    world_size,
                    block,
                )
            else:
                dequantize_add(
                    state.reference,
                    quantized_all,
                    scales_all,
                    shard_numel,
                    world_size,
                    block,
                )
        else:
            if state.block_metadata is not None:
                payload_work = _launch_param_major_prime(
                    all_gather_comm,
                    group,
                    shard_input,
                    state.reference,
                    state.param_shard_numels,
                    world_size,
                    async_op,
                )
            else:
                payload_work = all_gather_comm(
                    output_tensor=state.reference,
                    input_tensor=shard_input,
                    group=group,
                    async_op=async_op,
                )
        state.gathers += 1
        all_gather_event = all_gather_stream.record_event()
    return AllGatherResult(
        state.reference,
        all_gather_event,
        payload_work,
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


def install_delta_fp8_allgather(
    *,
    block: int = DEFAULT_BLOCK,
    prime_steps: int = 1,
    reprime_interval: int = 0,
) -> None:
    """Patch FSDP2 ``foreach_all_gather`` to communicate quantized deltas."""
    global _INSTALLED, _ORIGINAL_FOREACH_ALL_GATHER
    global _ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT, _ORIGINAL_FREE_UNSHARDED_PARAM
    global _ORIGINAL_INIT_UNSHARDED_PARAM, _MODEL_SCOPED
    require_triton()
    _validate_config(block, prime_steps, reprime_interval)
    _CONFIG["block"] = int(block)
    _CONFIG["prime_steps"] = int(prime_steps)
    _CONFIG["reprime_interval"] = int(reprime_interval)
    # Direct installation retains the historical process-wide behavior.
    # ``register_delta_fp8_allgather`` narrows the hook back to model-owned groups.
    _MODEL_SCOPED = False
    if _INSTALLED:
        logger.warning(
            "FSDP2 delta-FP8 all-gather already installed; updated "
            "(block=%d, prime_steps=%d, reprime_interval=%d, "
            "memory_optimizations=default)",
            _CONFIG["block"],
            _CONFIG["prime_steps"],
            _CONFIG["reprime_interval"],
        )
        return
    _ORIGINAL_FOREACH_ALL_GATHER = _collectives.foreach_all_gather
    _ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT = _collectives.foreach_all_gather_copy_out
    _ORIGINAL_FREE_UNSHARDED_PARAM = FSDPParam.free_unsharded_param
    _ORIGINAL_INIT_UNSHARDED_PARAM = FSDPParam.init_unsharded_param
    _collectives.foreach_all_gather = _delta_foreach_all_gather
    _param_group.foreach_all_gather = _delta_foreach_all_gather
    _collectives.foreach_all_gather_copy_out = _delta_foreach_all_gather_copy_out
    _param_group.foreach_all_gather_copy_out = _delta_foreach_all_gather_copy_out
    FSDPParam.free_unsharded_param = _delta_free_unsharded_param
    FSDPParam.init_unsharded_param = _delta_init_unsharded_param
    _INSTALLED = True
    logger.warning(
        "FSDP2 delta-FP8 all-gather installed "
        "(block=%d, prime_steps=%d, reprime_interval=%d, "
        "memory_optimizations=default)",
        _CONFIG["block"],
        _CONFIG["prime_steps"],
        _CONFIG["reprime_interval"],
    )


def register_delta_fp8_allgather(
    model,
    *,
    block: int = DEFAULT_BLOCK,
    prime_steps: int = 1,
    reprime_interval: int = 0,
) -> int:
    """Validate and enable optimized Delta-FP8 for a wrapped FSDP model."""
    block = int(block)
    prime_steps = int(prime_steps)
    reprime_interval = int(reprime_interval)
    _validate_config(block, prime_steps, reprime_interval)
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
            if param_group.fsdp_params:
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

    install_delta_fp8_allgather(
        block=block,
        prime_steps=prime_steps,
        reprime_interval=reprime_interval,
    )
    config = _GroupConfig(block, prime_steps, reprime_interval)
    for param_group in param_groups:
        _GROUP_CONFIGS[param_group.fsdp_params[0]] = config
    global _MODEL_SCOPED
    _MODEL_SCOPED = True
    registered = len(param_groups)
    logger.warning(
        "FSDP2 delta-FP8 all-gather registered for %d groups "
        "(block=%d, prime_steps=%d, reprime_interval=%d, "
        "memory_optimizations=default)",
        registered,
        block,
        prime_steps,
        reprime_interval,
    )
    return registered


def uninstall_delta_fp8_allgather() -> None:
    """Restore the stock FSDP2 ``foreach_all_gather``. Tests only."""
    global _INSTALLED, _ORIGINAL_FOREACH_ALL_GATHER
    global _ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT, _ORIGINAL_FREE_UNSHARDED_PARAM
    global _ORIGINAL_INIT_UNSHARDED_PARAM, _MODEL_SCOPED
    if not _INSTALLED:
        return
    _collectives.foreach_all_gather = _ORIGINAL_FOREACH_ALL_GATHER
    _param_group.foreach_all_gather = _ORIGINAL_FOREACH_ALL_GATHER
    _collectives.foreach_all_gather_copy_out = _ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT
    _param_group.foreach_all_gather_copy_out = _ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT
    FSDPParam.free_unsharded_param = _ORIGINAL_FREE_UNSHARDED_PARAM
    FSDPParam.init_unsharded_param = _ORIGINAL_INIT_UNSHARDED_PARAM
    _ORIGINAL_FOREACH_ALL_GATHER = None
    _ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT = None
    _ORIGINAL_FREE_UNSHARDED_PARAM = None
    _ORIGINAL_INIT_UNSHARDED_PARAM = None
    for state in _STATES.values():
        for fsdp_param in state.fsdp_params:
            fsdp_param._delta_fp8_persistent_reference = False
    _STATES.clear()
    _SCRATCH_STATES.clear()
    _ALIASED_RESULTS.clear()
    _GROUP_CONFIGS.clear()
    _MODEL_SCOPED = False
    _INSTALLED = False
