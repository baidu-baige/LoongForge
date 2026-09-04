# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Replicated-compute ZeRO-1 parameter and optimizer-state ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import zlib

import torch
import torch.distributed as dist
from torch import nn


class MasterParameterView:
    """Minimal module-like view consumed by the vendored optimizer builder."""

    def __init__(self, named_parameters):
        self._named_parameters = tuple(named_parameters)

    def named_parameters(self):
        """Yield the ``(name, parameter)`` pairs this view was built from."""
        return iter(self._named_parameters)

    def named_modules(self):
        """Yield nothing; the view has no module tree."""
        return iter(())


@dataclass(frozen=True)
class ParameterOwnership:
    """Which rank owns a parameter, and how experts are split when sharded."""

    name: str
    shape: tuple[int, ...]
    owner: int | None
    expert_counts: tuple[int, ...] = ()

    @property
    def is_expert_shard(self) -> bool:
        """True when the parameter has no single owner and is split by expert."""
        # Sharded means "no single owner", which is what assign_parameter_owners
        # records. Keying off len(shape) == 3 instead would disagree with the
        # policy the moment sharding is disabled: the tensor gets a real owner
        # but expert_counts stays empty, and every rank dies with an IndexError
        # in Zero1ParameterManager.__init__.
        return self.owner is None


@dataclass
class GradientSyncEntry:
    """One in-flight gradient collective and the buffers it borrows."""

    kind: str
    items: tuple[ParameterOwnership, ...]
    owner: int | None = None
    payload: torch.Tensor | None = None
    buffer: torch.Tensor | None = None
    input_buffer: torch.Tensor | None = None
    input_pooled: bool = False
    nbytes: int = 0
    work: object | None = None
    dtype: torch.dtype = torch.float32


_UNKNOWN_ORDINAL = 1 << 30

# Overlap scheduling constants, deliberately not exposed as configuration:
# deeper parameter-sync queues (depth 4 and 6) benchmarked at 0.94x on 8 GPUs
# at GBS80, so the depth below is the retained optimum. The parameter in-flight
# cap bounds peak memory rather than throughput.
_PARAM_SYNC_DEPTH = 2
_PARAM_INFLIGHT_BYTES = 2048 * 1024 * 1024
# Default bucket sizes, used when the model config does not pin one. Overlapped
# sync wants finer buckets so collectives start early; the serial path prefers
# fewer, larger ones.
_BUCKET_MB_OVERLAP = 256
_BUCKET_MB_SERIAL = 1024


@dataclass
class ParameterSyncEntry:
    """One in-flight parameter-publish collective and its scheduling position."""

    kind: str
    items: tuple[ParameterOwnership, ...]
    owner: int | None
    is_early: bool
    position: int
    payload: object | None = None
    nbytes: int = 0
    work: object | None = None


def assign_parameter_owners(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    world_size: int,
    parameter_policy=None,
):
    """Assign full tensors to owners and policy-approved expert shards on dim 0."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if parameter_policy is None:
        raise ValueError("parameter_policy is required")
    policy = parameter_policy
    items = list(named_parameters)
    loads = [0] * world_size
    result = []
    for name, parameter in items:
        shape = tuple(parameter.shape)
        if policy.is_expert_shard(name, parameter):
            base, remainder = divmod(shape[0], world_size)
            counts = tuple(base + (rank < remainder) for rank in range(world_size))
            for rank, count in enumerate(counts):
                loads[rank] += count * parameter[0].numel() * 4 if shape[0] else 0
            result.append(ParameterOwnership(name, shape, None, counts))
        else:
            owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
            loads[owner] += parameter.numel() * 4
            result.append(ParameterOwnership(name, shape, owner))
    return result


class Zero1ParameterManager:
    """Own FP32 master/state shards while retaining complete compute replicas."""

    def __init__(
        self,
        module: nn.Module,
        group=None,
        rank=None,
        world_size=None,
        parameter_policy=None,
        grad_reduce_dtype: str | None = None,
        param_sync_dtype: str | None = None,
        grad_overlap: bool = True,
        param_overlap: bool = True,
        bucket_mb: int | None = None,
        grad_inflight_mb: int = 3072,
    ):
        self.module = module
        self.group = group
        self.rank = dist.get_rank(group) if rank is None else rank
        self.world_size = dist.get_world_size(group) if world_size is None else world_size
        if parameter_policy is None:
            raise ValueError("parameter_policy is required")
        self.parameter_policy = parameter_policy
        self.compute = {name: parameter for name, parameter in module.named_parameters() if parameter.requires_grad}
        # The ready order is measured, never configured: iteration 1 buckets by
        # reverse registration order while the gradient hooks record the true
        # backward-ready sequence, and iteration 2 rebuilds the plan from it.
        # Owner assignment deliberately stays on the validated greedy scheme --
        # reordering owners changes Muon's same-shape megabatch grouping
        # (measured up to 2% grad norm drift), while ready-order bucketing alone
        # is bit-identical.
        self._ready_order = None
        self.ownership = assign_parameter_owners(
            self.compute.items(), self.world_size, self.parameter_policy
        )
        self.specs = {item.name: item for item in self.ownership}
        # named_parameters() follows the forward pass, so its reverse approximates
        # backward readiness well enough for the first iteration.
        self._reverse_position = {
            item.name: -index for index, item in enumerate(self.ownership)
        }
        self.master: dict[str, nn.Parameter] = {}
        self._expert_offsets = {}
        self._pending = []
        # Collective precision, configured from the model YAML
        # (``model.grad_reduce_dtype`` / ``model.param_sync_dtype``); override per
        # run on the command line, e.g. ``model.param_sync_dtype=bf16``. Gradient
        # reduction and parameter publication are downcast per parameter, never per
        # collective: the parameters the policy marks precision-critical (MoE
        # router/gate, 1-D tensors) always travel at full precision, and buckets
        # are split so a critical parameter can never be dragged into a downcast
        # payload.
        self._grad_comm_mode = (grad_reduce_dtype or "fp32").lower()
        self._param_comm_mode = (param_sync_dtype or "compute").lower()
        for name, value, allowed in (
            ("grad_reduce_dtype", self._grad_comm_mode, ("fp32", "bf16", "mixed", "compute")),
            ("param_sync_dtype", self._param_comm_mode, ("compute", "bf16")),
        ):
            if value not in allowed:
                raise ValueError(
                    f"invalid {name}={value!r}; expected one of {', '.join(allowed)}"
                )
        self._comm_critical = {
            item.name
            for item in self.ownership
            if self.parameter_policy.is_comm_precision_critical(
                item.name, self.compute[item.name]
            )
        }
        self._grad_overlap_enabled = bool(grad_overlap)
        self._bucket_mb = int(
            bucket_mb
            if bucket_mb
            else (_BUCKET_MB_OVERLAP if grad_overlap else _BUCKET_MB_SERIAL)
        )
        self._grad_sync_entries: list[GradientSyncEntry] = []
        self._grad_sync_by_name: dict[str, GradientSyncEntry] = {}
        self._grad_sync_next = 0
        self._grad_sync_handles = []
        self._grad_sync_inflight: list[GradientSyncEntry] = []
        self._grad_inflight_limit = int(grad_inflight_mb) * 1024 * 1024
        self._grad_inflight_bytes = 0
        self._gradient_overlap_active = False
        self._grad_present_names: set[str] = set()
        self._grad_ready_names: set[str] = set()
        self._buffer_pool: dict[tuple, list[torch.Tensor]] = {}
        self._record_order = True
        self._pending_ready_order = None
        self._ready_ordinals: dict[str, int] = {}
        self._ready_counter = 0
        # Overlap owner-side optimizer updates with the replica parameter sync.
        self._param_overlap_enabled = bool(param_overlap)
        self._param_sync_entries: list[ParameterSyncEntry] = []
        self._param_sync_inflight: list[ParameterSyncEntry] = []
        self._param_sync_next = 0
        self._param_inflight_bytes = 0
        self._param_inflight_limit = _PARAM_INFLIGHT_BYTES
        self._master_ids: dict[int, str] = {}
        self._master_updated: set[str] = set()
        self._muon_managed: set[str] | None = None

        for item in self.ownership:
            compute = self.compute[item.name]
            if item.is_expert_shard:
                start = sum(item.expert_counts[: self.rank])
                count = item.expert_counts[self.rank]
                self._expert_offsets[item.name] = (start, count)
                value = compute.detach()[start : start + count].float().clone()
                self.master[item.name] = nn.Parameter(value, requires_grad=True)
            elif item.owner == self.rank:
                self.master[item.name] = nn.Parameter(compute.detach().float().clone(), requires_grad=True)

            compute_dtype = self.parameter_policy.compute_dtype(item.name, compute)
            if compute.dtype != compute_dtype:
                compute.data = compute.data.to(compute_dtype)

        if self._grad_overlap_enabled:
            self._build_gradient_overlap_plan()

    def optimizer_view(self):
        """Return a module-like view over the fp32 master parameters."""
        return MasterParameterView(self.master.items())

    @property
    def grad_reduce_mode(self) -> str:
        """Resolved gradient-reduction precision mode."""
        return self._grad_comm_mode

    @property
    def param_sync_mode(self) -> str:
        """Resolved parameter-publication precision mode."""
        return self._param_comm_mode

    @property
    def bucket_mb(self) -> int:
        """Resolved collective bucket size in MiB."""
        return self._bucket_mb

    def named_master_parameters(self):
        """Return the ``(name, master parameter)`` pairs owned by this rank."""
        return list(self.master.items())

    def _bucket_items(self, items, size_mb=None):
        limit = int(size_mb or self._bucket_mb) * 1024 * 1024
        buckets = []
        current = []
        current_bytes = 0
        for item in items:
            size = self.compute[item.name].numel() * 4
            if current and current_bytes + size > limit:
                buckets.append(current)
                current = []
                current_bytes = 0
            current.append(item)
            current_bytes += size
            if current_bytes >= limit:
                buckets.append(current)
                current = []
                current_bytes = 0
        if current:
            buckets.append(current)
        return buckets

    def _grad_reduce_dtype(self, item, kind="ordinary"):
        """Pick the wire dtype for one parameter's gradient reduction."""
        mode = self._grad_comm_mode
        if mode == "fp32" or item.name in self._comm_critical:
            return torch.float32
        if kind == "expert":
            # Expert shards only ever downcast under the explicit bf16 mode; the
            # "mixed"/"compute" modes keep the reduce-scatter in fp32.
            return torch.bfloat16 if mode == "bf16" else torch.float32
        if mode in ("bf16", "mixed"):
            return torch.bfloat16
        if mode == "compute":
            return (
                torch.bfloat16
                if self.compute[item.name].dtype == torch.bfloat16
                else torch.float32
            )
        return torch.float32

    def _param_sync_dtype(self, item):
        """Pick the wire dtype for publishing one updated parameter to replicas."""
        if item.name in self._comm_critical:
            return torch.float32
        compute_dtype = self.compute[item.name].dtype
        if compute_dtype in (torch.float16, torch.bfloat16):
            return compute_dtype
        # fp32 compute parameters (the action expert) can still be published in
        # bf16: the fp32 master keeps the update precision, replicas just receive
        # a rounded copy -- and all ranks, owner included, read the same rounded
        # bytes back, so the replicas stay bit-identical.
        if self._param_comm_mode == "bf16":
            return torch.bfloat16
        return torch.float32

    def _split_by_dtype(self, items, dtype_of):
        """Group items by wire dtype so a bucket never mixes precisions."""
        groups: dict[torch.dtype, list] = {}
        for item in items:
            groups.setdefault(dtype_of(item), []).append(item)
        return [groups[key] for key in sorted(groups, key=str)]

    def _ready_key(self, name):
        if self._ready_order:
            return (1, self._ready_order.get(name, _UNKNOWN_ORDINAL), name)
        # The first iteration has no observed ready order to build from. The
        # queue launches strictly in order, so a head that is not ready blocks
        # every later entry and forces all gradients to stay resident until
        # backward ends (+26GB/rank, OOM at GBS80). Reverse registration order
        # keeps the head genuinely ready first.
        return (0, self._reverse_position[name], name)

    def _build_gradient_overlap_plan(self, register_hooks=True):
        """Build a deterministic communication plan and optionally register hooks."""
        entries = []
        ordinary = [item for item in self.ownership if not item.is_expert_shard]
        ordinary = sorted(ordinary, key=lambda item: self._ready_key(item.name))
        for owner in range(self.world_size):
            owned = [item for item in ordinary if item.owner == owner]
            for group in self._split_by_dtype(owned, self._grad_reduce_dtype):
                for bucket in self._bucket_items(group):
                    entries.append(
                        GradientSyncEntry(
                            "ordinary",
                            tuple(bucket),
                            owner=owner,
                            dtype=self._grad_reduce_dtype(bucket[0]),
                        )
                    )
        for item in self.ownership:
            if item.is_expert_shard:
                entries.append(
                    GradientSyncEntry(
                        "expert",
                        (item,),
                        dtype=self._grad_reduce_dtype(item, "expert"),
                    )
                )
        entries.sort(key=lambda entry: max(self._ready_key(item.name) for item in entry.items))
        self._grad_sync_entries = entries
        self._grad_sync_by_name = {
            item.name: entry for entry in entries for item in entry.items
        }
        if register_hooks:
            self._register_gradient_ready_hooks()

    def _register_gradient_ready_hooks(self):
        for name, parameter in self.compute.items():
            if name not in self._grad_sync_by_name:
                continue
            handle = parameter.register_post_accumulate_grad_hook(
                lambda _parameter, param_name=name: self._on_gradient_ready(param_name)
            )
            self._grad_sync_handles.append(handle)

    def _acquire_buffer(self, numel, dtype, device):
        key = (numel, dtype, str(device))
        pool = self._buffer_pool.setdefault(key, [])
        if pool:
            return pool.pop()
        return torch.empty(numel, dtype=dtype, device=device)

    def _release_buffer(self, buffer):
        if buffer is None:
            return
        key = (buffer.numel(), buffer.dtype, str(buffer.device))
        self._buffer_pool.setdefault(key, []).append(buffer)

    def begin_gradient_overlap(self):
        """Arm the overlap plan so per-parameter hooks can launch collectives."""
        if not self._grad_overlap_enabled:
            return
        if self._gradient_overlap_active:
            raise RuntimeError("gradient overlap step was already started")
        if any(
            entry.work is not None or entry.payload is not None or entry.input_buffer is not None
            for entry in self._grad_sync_entries
        ):
            raise RuntimeError("previous gradient overlap step was not finished")
        if self._pending_ready_order is not None:
            self._ready_order = self._pending_ready_order
            self._pending_ready_order = None
            self._build_gradient_overlap_plan(register_hooks=False)
            self._assert_gradient_plan_matches_ranks()
            # Rebucketing produces new bucket sizes, and the pool is keyed by
            # exact numel, so every pooled buffer from the previous plan would
            # stay resident forever. Nothing is in flight here (the guard above
            # rejects an unfinished step), so the whole generation can go.
            self._buffer_pool.clear()
        self._grad_sync_next = 0
        self._grad_sync_inflight.clear()
        # Gradients from earlier accumulation micro-batches are already present,
        # but they are not safe to communicate until the same parameter's hook
        # runs in the final backward and adds its last contribution.
        self._grad_present_names = {
            name for name, parameter in self.compute.items() if parameter.grad is not None
        }
        self._grad_ready_names.clear()
        self._grad_inflight_bytes = 0
        for parameter in self.master.values():
            parameter.grad = None
        self._gradient_overlap_active = True

    def _on_gradient_ready(self, name):
        if self._grad_overlap_enabled and self._gradient_overlap_active:
            self._grad_present_names.add(name)
            self._grad_ready_names.add(name)
            if self._record_order and name not in self._ready_ordinals:
                self._ready_ordinals[name] = self._ready_counter
                self._ready_counter += 1
            self._launch_ready_gradient_entries()

    def _entry_is_ready(self, entry):
        return all(
            item.name in self._grad_ready_names
            and self.compute[item.name].grad is not None
            for item in entry.items
        )

    @torch.no_grad()
    def _reclaim_completed_gradient_entries(self):
        """Retire collectives that already finished, without stalling any stream."""
        while self._grad_sync_inflight:
            entry = self._grad_sync_inflight[0]
            if entry.work is not None and not entry.work.is_completed():
                break
            self._complete_gradient_entry(self._grad_sync_inflight.pop(0))

    @torch.no_grad()
    def _launch_ready_gradient_entries(self, force=False):
        while self._grad_sync_next < len(self._grad_sync_entries):
            entry = self._grad_sync_entries[self._grad_sync_next]
            if not force and not self._entry_is_ready(entry):
                break
            # Blocking on a NCCL event inside the autograd hook would stall every
            # later backward kernel on the compute stream, so in-flight collectives
            # are bounded by bytes and reclaimed only once they already completed --
            # never by waiting on a fixed queue depth.
            self._reclaim_completed_gradient_entries()
            while (
                self._grad_sync_inflight
                and self._grad_inflight_bytes >= self._grad_inflight_limit
            ):
                self._complete_gradient_entry(self._grad_sync_inflight.pop(0))
            self._launch_gradient_entry(entry)
            self._grad_sync_inflight.append(entry)
            self._grad_sync_next += 1

    def _gradient_or_zeros(self, item, dtype):
        grad = self.compute[item.name].grad
        if grad is not None:
            return grad.detach().to(dtype)
        return torch.zeros(item.shape, dtype=dtype, device=next(iter(self.compute.values())).device)

    @torch.no_grad()
    def _launch_gradient_entry(self, entry):
        device = next(iter(self.compute.values())).device
        if entry.kind == "ordinary":
            total = sum(self.compute[item.name].numel() for item in entry.items)
            flat = self._acquire_buffer(total, entry.dtype, device)
            offset = 0
            for item in entry.items:
                compute = self.compute[item.name]
                count = compute.numel()
                target = flat[offset : offset + count]
                if compute.grad is None:
                    target.zero_()
                else:
                    target.copy_(compute.grad.detach().reshape(-1))
                offset += count
            work = (
                dist.reduce(flat, dst=entry.owner, group=self.group, async_op=True)
                if self.world_size > 1
                else None
            )
            entry.buffer = flat
            entry.payload = flat
            entry.work = work
            entry.nbytes = flat.numel() * flat.element_size()
            self._grad_inflight_bytes += entry.nbytes
            for item in entry.items:
                self.compute[item.name].grad = None
            return

        item = entry.items[0]
        counts = item.expert_counts
        max_count = max(counts)
        rows = max_count * self.world_size
        row_numel = 1
        for dim in item.shape[1:]:
            row_numel *= dim
        grad = self.compute[item.name].grad
        even = rows == item.shape[0] and all(count == max_count for count in counts)
        if even and grad is not None and grad.dtype == entry.dtype:
            # 32 experts over 8 ranks needs no padding, so the reduce-scatter can
            # read the gradient in place instead of copying the full tensor.
            padded = grad.detach().contiguous()
            entry.input_buffer = padded
            entry.input_pooled = False
            padded_bytes = 0
        else:
            padded_flat = self._acquire_buffer(rows * row_numel, entry.dtype, device)
            padded = padded_flat.view(rows, *item.shape[1:])
            padded.zero_()
            if grad is not None:
                full = grad.detach().to(entry.dtype)
                cursor = 0
                for owner, count in enumerate(counts):
                    padded[owner * max_count : owner * max_count + count].copy_(
                        full[cursor : cursor + count]
                    )
                    cursor += count
            entry.input_buffer = padded_flat
            entry.input_pooled = True
            padded_bytes = padded_flat.numel() * padded_flat.element_size()
        local_flat = self._acquire_buffer(max_count * row_numel, entry.dtype, device)
        local = local_flat.view(max_count, *item.shape[1:])
        work = (
            dist.reduce_scatter_tensor(local, padded, group=self.group, async_op=True)
            if self.world_size > 1
            else None
        )
        if self.world_size == 1:
            local.copy_(padded)
        entry.buffer = local_flat
        entry.payload = local
        entry.work = work
        entry.nbytes = local_flat.numel() * local_flat.element_size() + padded_bytes
        self._grad_inflight_bytes += entry.nbytes
        self.compute[item.name].grad = None

    @torch.no_grad()
    def _complete_gradient_entry(self, entry):
        if entry.work is not None:
            entry.work.wait()
        payload = entry.payload
        if entry.kind == "ordinary":
            if self.rank == entry.owner:
                offset = 0
                for item in entry.items:
                    master = self.master[item.name]
                    count = master.numel()
                    # .float() is a no-op for an fp32 payload and .div() always
                    # allocates, so the pooled buffer is never written in place.
                    master.grad = (
                        payload[offset : offset + count]
                        .view_as(master)
                        .float()
                        .div(self.world_size)
                    )
                    offset += count
        else:
            item = entry.items[0]
            count = item.expert_counts[self.rank]
            self.master[item.name].grad = payload[:count].float().div(self.world_size)
        self._grad_inflight_bytes -= entry.nbytes
        self._release_buffer(entry.buffer)
        if entry.input_pooled:
            self._release_buffer(entry.input_buffer)
        entry.payload = None
        entry.buffer = None
        entry.input_buffer = None
        entry.input_pooled = False
        entry.nbytes = 0
        entry.work = None

    @torch.no_grad()
    def finish_gradient_overlap(self):
        """Drain the in-flight gradient collectives, or reduce serially if off."""
        if not self._grad_overlap_enabled:
            self.reduce_gradients_to_owners()
            return
        if not self._gradient_overlap_active:
            raise RuntimeError("gradient overlap step was not started")
        self._launch_ready_gradient_entries(force=True)
        while self._grad_sync_inflight:
            self._complete_gradient_entry(self._grad_sync_inflight.pop(0))
        present = torch.tensor(
            [item.name in self._grad_present_names for item in self.ownership],
            device=next(iter(self.compute.values())).device,
            dtype=torch.int32,
        )
        if self.world_size > 1:
            dist.all_reduce(present, op=dist.ReduceOp.MAX, group=self.group)
        for item, has_grad in zip(self.ownership, present.tolist()):
            if not has_grad and item.name in self.master:
                self.master[item.name].grad = None
        for parameter in self.compute.values():
            parameter.grad = None
        if self._record_order:
            self._capture_ready_order()
        self._gradient_overlap_active = False

    @torch.no_grad()
    def _assert_plan_matches_ranks(self, plan, error_message):
        if self.world_size <= 1:
            return
        signature = zlib.crc32(repr(plan).encode("utf-8"))
        probe = torch.tensor(
            [signature, -signature],
            device=next(iter(self.compute.values())).device,
            dtype=torch.int64,
        )
        dist.all_reduce(probe, op=dist.ReduceOp.MAX, group=self.group)
        if int(probe[0]) != -int(probe[1]):
            raise RuntimeError(error_message)

    def _assert_gradient_plan_matches_ranks(self):
        plan = [
            (entry.kind, entry.owner, str(entry.dtype), tuple(item.name for item in entry.items))
            for entry in self._grad_sync_entries
        ]
        self._assert_plan_matches_ranks(
            plan, "gradient overlap plan differs across ranks"
        )

    @torch.no_grad()
    def _capture_ready_order(self):
        """Publish the measured backward-ready order for the next iteration."""
        self._record_order = False
        unknown = len(self.ownership) + 1
        ordinals = torch.tensor(
            [self._ready_ordinals.get(item.name, unknown) for item in self.ownership],
            device=next(iter(self.compute.values())).device,
            dtype=torch.int32,
        )
        if self.world_size > 1:
            # Agree on one observation so every rank rebuilds the same plan;
            # _assert_gradient_plan_matches_ranks then verifies it did.
            dist.all_reduce(ordinals, op=dist.ReduceOp.MAX, group=self.group)
        self._pending_ready_order = {
            item.name: int(value) for item, value in zip(self.ownership, ordinals.tolist())
        }

    @torch.no_grad()
    def reduce_gradients_to_owners(self):
        """Average gradients directly onto owners using owner-aligned buckets."""
        device = next(iter(self.compute.values())).device
        present = torch.tensor(
            [self.compute[item.name].grad is not None for item in self.ownership],
            device=device,
            dtype=torch.int32,
        )
        if self.world_size > 1:
            dist.all_reduce(present, op=dist.ReduceOp.MAX, group=self.group)
        any_present = present.tolist()

        # Whole-tensor parameters are packed by owner. This preserves Muon's
        # complete-matrix ownership while replacing O(parameters) reductions
        # with O(owners * buckets) reductions.
        for owner in range(self.world_size):
            owned = [
                item
                for item, has_grad in zip(self.ownership, any_present)
                if has_grad and not item.is_expert_shard and item.owner == owner
            ]
            owned_groups = self._split_by_dtype(owned, self._grad_reduce_dtype)
            for owned_group in owned_groups:
                for bucket in self._bucket_items(owned_group):
                    bucket_reduce_dtype = self._grad_reduce_dtype(bucket[0])
                    tensors = []
                    for item in bucket:
                        grad = self.compute[item.name].grad
                        tensors.append(
                            grad.detach().to(bucket_reduce_dtype).reshape(-1)
                            if grad is not None
                            else torch.zeros(
                                self.compute[item.name].numel(), device=device, dtype=bucket_reduce_dtype
                            )
                        )
                    flat = torch.cat(tensors)
                    if self.world_size > 1:
                        dist.reduce(flat, dst=owner, group=self.group)
                    if self.rank == owner:
                        flat = flat.float().div_(self.world_size)
                        offset = 0
                        for item in bucket:
                            master = self.master[item.name]
                            count = master.numel()
                            master.grad = flat[offset : offset + count].view_as(master).clone()
                            offset += count

        # Fused experts remain partitioned only along dim 0 so every owner sees
        # complete expert matrices for Newton-Schulz.
        for item, has_grad in zip(self.ownership, any_present):
            if not item.is_expert_shard or not has_grad:
                continue
            compute = self.compute[item.name]
            expert_reduce_dtype = self._grad_reduce_dtype(item, "expert")
            full_grad = (
                compute.grad.detach().to(expert_reduce_dtype).contiguous()
                if compute.grad is not None
                else torch.zeros_like(compute, dtype=expert_reduce_dtype)
            )
            max_count = max(item.expert_counts)
            padded = torch.zeros(
                (max_count * self.world_size, *item.shape[1:]),
                dtype=expert_reduce_dtype,
                device=device,
            )
            cursor = 0
            for owner, count in enumerate(item.expert_counts):
                padded[owner * max_count : owner * max_count + count].copy_(
                    full_grad[cursor : cursor + count]
                )
                cursor += count
            local = torch.empty(
                (max_count, *item.shape[1:]), dtype=expert_reduce_dtype, device=device
            )
            if self.world_size > 1:
                dist.reduce_scatter_tensor(local, padded, group=self.group)
            else:
                local.copy_(padded)
            local = local.float().div_(self.world_size)
            count = item.expert_counts[self.rank]
            self.master[item.name].grad = local[:count].clone()

        for item, has_grad in zip(self.ownership, any_present):
            if not has_grad and item.name in self.master:
                self.master[item.name].grad = None
            self.compute[item.name].grad = None

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> float:
        """Clip owned master gradients by global norm and return that norm."""
        # Keep the fp64 per-element accumulation. Measured and rejected:
        # ``torch._foreach_norm`` is 1.019x end-to-end but accumulates in fp32
        # opmath, moving loss up to 0.963% and grad norm up to 3.65% over 20 steps.
        # The clipping coefficient feeds every parameter, so the accumulation
        # dtype has to be the widest one in the step, not the fastest.
        local_sq = torch.zeros((), dtype=torch.float64, device=next(iter(self.compute.values())).device)
        for parameter in self.master.values():
            if parameter.grad is not None:
                local_sq.add_(parameter.grad.detach().double().square().sum())
        if self.world_size > 1:
            dist.all_reduce(local_sq, group=self.group)
        norm = local_sq.sqrt()
        if max_norm > 0:
            coefficient = max_norm / (norm + 1e-6)
            if coefficient < 1:
                for parameter in self.master.values():
                    if parameter.grad is not None:
                        parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
        return float(norm.item())

    def _finish_sync_entry(self, entry):
        kind, items, payload, work = entry
        if work is not None:
            work.wait()
        if kind == "expert":
            item = items[0]
            full = torch.cat(
                [value[:count] for value, count in zip(payload, item.expert_counts)], dim=0
            )
            self.compute[item.name].copy_(full)
            return
        offset = 0
        for item in items:
            compute = self.compute[item.name]
            count = compute.numel()
            compute.copy_(payload[offset : offset + count].view_as(compute))
            offset += count

    def _enqueue_sync(self, entry):
        self._pending.append(entry)
        if len(self._pending) > _PARAM_SYNC_DEPTH:
            self._finish_sync_entry(self._pending.pop(0))

    @torch.no_grad()
    def _build_expert_sync(self, item):
        """AllGather one 3-D expert stack from its dim-0 owners into all replicas."""
        device = next(iter(self.compute.values())).device
        max_count = max(item.expert_counts)
        sync_dtype = self._param_sync_dtype(item)
        send = torch.zeros((max_count, *item.shape[1:]), dtype=sync_dtype, device=device)
        count = item.expert_counts[self.rank]
        send[:count].copy_(self.master[item.name])
        gathered = [torch.empty_like(send) for _ in range(self.world_size)]
        work = (
            dist.all_gather(gathered, send, group=self.group, async_op=True)
            if self.world_size > 1
            else None
        )
        nbytes = send.numel() * send.element_size() * (self.world_size + 1)
        return ("expert", [item], gathered, work, nbytes)

    @torch.no_grad()
    def _build_ordinary_sync(self, bucket, owner):
        """Broadcast one owner-aligned bucket of whole parameters in compute dtype."""
        device = next(iter(self.compute.values())).device
        total = sum(self.compute[item.name].numel() for item in bucket)
        # Buckets are built per wire dtype, but stay defensive: one fp32 member
        # promotes the whole payload rather than silently rounding it.
        sync_dtype = torch.bfloat16
        for item in bucket:
            item_dtype = self._param_sync_dtype(item)
            if item_dtype == torch.float32:
                sync_dtype = torch.float32
                break
            sync_dtype = item_dtype
        if self.rank == owner:
            buffer = torch.cat([
                self.master[item.name].detach().to(sync_dtype).reshape(-1)
                for item in bucket
            ])
        else:
            buffer = torch.empty(total, dtype=sync_dtype, device=device)
        work = (
            dist.broadcast(buffer, src=owner, group=self.group, async_op=True)
            if self.world_size > 1
            else None
        )
        return ("ordinary", bucket, buffer, work, buffer.numel() * buffer.element_size())

    def _ordinary_sync_buckets(self):
        """Owner-aligned ordinary buckets, split by which optimizer updates them.

        Muon-managed masters finish incrementally during ``optimizer.step()`` and
        can start syncing early; AdamW-managed masters are only updated after the
        Muon inner optimizer returns, so mixing them into the same bucket would
        block the whole in-order launch queue.
        """
        ordinary = [item for item in self.ownership if not item.is_expert_shard]
        groups = []
        for owner in range(self.world_size):
            owned = [item for item in ordinary if item.owner == owner]
            if self._muon_managed is None:
                groups.append((owner, owned, True))
                continue
            early = [item for item in owned if item.name in self._muon_managed]
            late = [item for item in owned if item.name not in self._muon_managed]
            groups.append((owner, early, True))
            groups.append((owner, late, False))
        for owner, owned, is_early in groups:
            for dtype_group in self._split_by_dtype(owned, self._param_sync_dtype):
                for bucket in self._bucket_items(dtype_group):
                    yield owner, bucket, is_early

    def set_muon_managed(self, names):
        """Record which masters the Muon inner optimizer updates in-step."""
        self._muon_managed = set(names)
        self._param_sync_entries = []

    def _build_parameter_sync_plan(self):
        positions = {item.name: index for index, item in enumerate(self.ownership)}
        entries = []
        for item in self.ownership:
            if item.is_expert_shard:
                entries.append(
                    ParameterSyncEntry(
                        "expert", (item,), None, True, positions[item.name]
                    )
                )
        for owner, bucket, is_early in self._ordinary_sync_buckets():
            entries.append(
                ParameterSyncEntry(
                    "ordinary",
                    tuple(bucket),
                    owner,
                    is_early,
                    max(positions[item.name] for item in bucket),
                )
            )
        # Late (AdamW) entries cannot be ready before the Muon step returns, so
        # they go last and never block the early ones.
        entries.sort(key=lambda entry: (not entry.is_early, entry.position))
        self._param_sync_entries = entries
        self._master_ids = {id(value): name for name, value in self.master.items()}
        self._assert_param_plan_matches_ranks()

    def _assert_param_plan_matches_ranks(self):
        """Fail fast instead of deadlocking if ranks disagree on the sync order."""
        plan = [
            (
                entry.kind,
                entry.owner,
                entry.is_early,
                entry.items[0].name,
                len(entry.items),
            )
            for entry in self._param_sync_entries
        ]
        self._assert_plan_matches_ranks(
            plan,
            "ZeRO-1 parameter-sync plan differs across ranks; refusing to run "
            "overlapped parameter sync",
        )

    def begin_parameter_sync_overlap(self):
        """Reset the publish plan so optimizer updates can start collectives."""
        if not self._param_overlap_enabled:
            return
        if not self._param_sync_entries:
            self._build_parameter_sync_plan()
        if self._param_sync_inflight:
            raise RuntimeError("previous parameter sync overlap was not finished")
        self._param_sync_next = 0
        self._param_inflight_bytes = 0
        self._master_updated.clear()

    def on_master_updated(self, parameter):
        """Mark a master parameter as updated and launch any ready publishes."""
        if not self._param_overlap_enabled:
            return
        name = self._master_ids.get(id(parameter))
        if name is None:
            return
        self._master_updated.add(name)
        self._launch_ready_param_entries()

    def _param_entry_ready(self, entry):
        for item in entry.items:
            master = self.master.get(item.name)
            if master is None:
                # Not owned here; this rank only receives, so nothing to wait for.
                continue
            if item.name in self._master_updated:
                continue
            if master.grad is None:
                # The optimizer skips grad-less masters, so it never reports them.
                continue
            return False
        return True

    @torch.no_grad()
    def _reclaim_completed_param_entries(self):
        while self._param_sync_inflight:
            entry = self._param_sync_inflight[0]
            if entry.work is not None and not entry.work.is_completed():
                break
            self._finish_param_entry(self._param_sync_inflight.pop(0))

    @torch.no_grad()
    def _launch_ready_param_entries(self, force=False):
        while self._param_sync_next < len(self._param_sync_entries):
            entry = self._param_sync_entries[self._param_sync_next]
            if not force and not (entry.is_early and self._param_entry_ready(entry)):
                break
            self._reclaim_completed_param_entries()
            while (
                self._param_sync_inflight
                and self._param_inflight_bytes >= self._param_inflight_limit
            ):
                self._finish_param_entry(self._param_sync_inflight.pop(0))
            if entry.kind == "expert":
                built = self._build_expert_sync(entry.items[0])
            else:
                built = self._build_ordinary_sync(list(entry.items), entry.owner)
            entry.payload = built[2]
            entry.work = built[3]
            entry.nbytes = built[4]
            self._param_inflight_bytes += entry.nbytes
            self._param_sync_inflight.append(entry)
            self._param_sync_next += 1

    @torch.no_grad()
    def _finish_param_entry(self, entry):
        self._finish_sync_entry((entry.kind, entry.items, entry.payload, entry.work))
        self._param_inflight_bytes -= entry.nbytes
        entry.payload = None
        entry.work = None
        entry.nbytes = 0

    @torch.no_grad()
    def finish_parameter_sync_overlap(self):
        """Drain the in-flight publishes, or publish serially if overlap is off."""
        if not self._param_overlap_enabled:
            self.start_parameter_sync()
            self.finish_parameter_sync()
            return
        self._launch_ready_param_entries(force=True)
        while self._param_sync_inflight:
            self._finish_param_entry(self._param_sync_inflight.pop(0))

    @torch.no_grad()
    def start_parameter_sync(self):
        """Launch bounded owner-to-replica parameter synchronization."""
        if self._pending:
            raise RuntimeError("parameter synchronization is already pending")

        # Keep expert collectives separate to preserve dim-0 ownership, but cap
        # in-flight buffers so the full model is never materialized twice.
        for item in self.ownership:
            if not item.is_expert_shard:
                continue
            built = self._build_expert_sync(item)
            self._enqueue_sync(built[:4])

        # Whole parameters are broadcast in owner-aligned buckets, split by wire
        # dtype: VLM bf16 traffic stays bf16, the FP32 action-expert path stays
        # FP32 unless ``model.param_sync_dtype=bf16``, and precision-critical
        # parameters (MoE router/gate, 1-D tensors) always stay FP32.
        ordinary = [item for item in self.ownership if not item.is_expert_shard]
        for owner in range(self.world_size):
            owned = [item for item in ordinary if item.owner == owner]
            for dtype_group in self._split_by_dtype(owned, self._param_sync_dtype):
                for bucket in self._bucket_items(dtype_group):
                    built = self._build_ordinary_sync(bucket, owner)
                    self._enqueue_sync(built[:4])

    @torch.no_grad()
    def finish_parameter_sync(self):
        """Wait for every queued parameter broadcast to land."""
        while self._pending:
            self._finish_sync_entry(self._pending.pop(0))

    def state_dict(self):
        """Return ownership metadata plus the fp32 master weights on CPU."""
        return {
            "world_size": self.world_size,
            "ownership": [item.__dict__ for item in self.ownership],
            "master": {name: value.detach().cpu().clone() for name, value in self.master.items()},
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        """Restore the fp32 masters, rejecting any ownership-schema mismatch."""
        if state["world_size"] != self.world_size:
            raise RuntimeError(
                f"ZeRO-1 checkpoint world_size={state['world_size']} does not match {self.world_size}"
            )
        expected = [item.__dict__ for item in self.ownership]
        if state["ownership"] != expected:
            raise RuntimeError("ZeRO-1 parameter ownership schema does not match checkpoint")
        if set(state["master"]) != set(self.master):
            raise RuntimeError("ZeRO-1 FP32 master keys do not match checkpoint")
        for name, value in state["master"].items():
            target = self.master[name]
            if value.shape != target.shape or value.dtype != torch.float32:
                raise RuntimeError(f"invalid FP32 master tensor for {name}")
            target.copy_(value.to(target.device))


__all__ = [
    "MasterParameterView",
    "ParameterOwnership",
    "Zero1ParameterManager",
    "assign_parameter_owners",
]
