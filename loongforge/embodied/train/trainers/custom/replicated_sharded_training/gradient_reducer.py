# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Reduce gradients onto their owners, overlapped with backward or serially.

Both paths share one plan builder, so the overlap switch can only change when a
collective is issued, never which parameters travel together or in what dtype.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch
import torch.distributed as dist

# Shared with the parameter publisher. Both paths must bucket and probe a plan
# the same way, so the overlap switch can only change when a collective is
# issued, never which parameters travel together.
WHOLE_TENSOR = "whole_tensor"
_UNKNOWN_ORDINAL = 1 << 30


def split_by_dtype(items, dtype_of):
    """Group items by wire dtype so a bucket never mixes precisions."""
    groups: dict[torch.dtype, list] = {}
    for item in items:
        groups.setdefault(dtype_of(item), []).append(item)
    return [groups[key] for key in sorted(groups, key=str)]


def bucket_by_bytes(items, limit_bytes, size_of):
    """Split an ordered sequence into buckets of at most ``limit_bytes``.

    ``size_of`` reports the fp32 master footprint, not the wire footprint: the
    measured bucket layout the overlap schedule was tuned against uses that
    accounting, so a bf16 bucket really is half its nominal size.
    """
    buckets = []
    current = []
    current_bytes = 0
    for item in items:
        size = size_of(item)
        if current and current_bytes + size > limit_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += size
        if current_bytes >= limit_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
    if current:
        buckets.append(current)
    return buckets


class BufferPool:
    """Reuse flat collective buffers, keyed by exact ``(numel, dtype, device)``."""

    def __init__(self):
        """Start with an empty pool."""
        self._pool: dict[tuple, list[torch.Tensor]] = {}

    def acquire(self, numel, dtype, device):
        """Return a buffer of exactly ``numel`` elements, reusing one if possible."""
        pool = self._pool.setdefault((numel, dtype, str(device)), [])
        if pool:
            return pool.pop()
        return torch.empty(numel, dtype=dtype, device=device)

    def release(self, buffer):
        """Return a buffer to the pool; ``None`` is accepted and ignored."""
        if buffer is None:
            return
        key = (buffer.numel(), buffer.dtype, str(buffer.device))
        self._pool.setdefault(key, []).append(buffer)

    def clear(self):
        """Drop every pooled buffer.

        Required whenever the plan is rebucketed: the key is the exact element
        count, so buffers from the previous generation would stay resident for
        the rest of training without ever being handed out again.
        """
        self._pool.clear()

    def __len__(self):
        """Return how many buffers are currently pooled."""
        return sum(len(buffers) for buffers in self._pool.values())


def assert_plan_matches_ranks(plan, group, world_size, device, message):
    """Fail fast, instead of deadlocking, when ranks disagree on a plan.

    The probe sends ``(signature, -signature)`` through a MAX all-reduce: the two
    halves only stay symmetric if every rank contributed the same signature.
    """
    if world_size <= 1:
        return
    signature = zlib.crc32(repr(plan).encode("utf-8"))
    probe = torch.tensor([signature, -signature], device=device, dtype=torch.int64)
    dist.all_reduce(probe, op=dist.ReduceOp.MAX, group=group)
    if int(probe[0]) != -int(probe[1]):
        raise RuntimeError(message)


@dataclass
class GradientSyncEntry:
    """One in-flight gradient collective and the buffers it borrows."""

    kind: str
    records: tuple
    owner: int | None = None
    output_buffer: torch.Tensor | None = None
    output_view: torch.Tensor | None = None
    input_buffer: torch.Tensor | None = None
    input_pooled: bool = False
    nbytes: int = 0
    work: object | None = None
    dtype: torch.dtype = torch.float32

    @property
    def names(self) -> tuple[str, ...]:
        """Return the parameter names this collective carries."""
        return tuple(record.name for record in self.records)


class GradientReducer:
    """Average gradients directly onto the ranks that own the fp32 masters."""

    def __init__(
        self,
        registry,
        group,
        rank,
        world_size,
        bucket_mb,
        inflight_bytes,
        overlap=True,
    ):
        """Build the plan and, when overlapping, arm the per-parameter hooks."""
        self._registry = registry
        self._group = group
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._bucket_bytes = int(bucket_mb) * 1024 * 1024
        self._inflight_limit = int(inflight_bytes)
        self._overlap = bool(overlap)

        self.pool = BufferPool()
        self._entries: list[GradientSyncEntry] = []
        self._entry_by_name: dict[str, GradientSyncEntry] = {}
        self._next = 0
        self._inflight: list[GradientSyncEntry] = []
        self._inflight_bytes = 0
        self._handles = []
        self._active = False
        self._present_names: set[str] = set()
        self._ready_names: set[str] = set()
        # The ready order is measured, never configured: the first step buckets
        # by reverse registration order while the hooks record the true
        # backward-ready sequence, and a later step rebuilds the plan from it.
        self._ready_order = None
        self._pending_ready_order = None
        self._record_order = True
        self._ready_ordinals: dict[str, int] = {}
        self._ready_counter = 0

        if self._overlap:
            self._build_plan(register_hooks=True)

    @property
    def entries(self):
        """Return the current gradient plan."""
        return self._entries

    def describe_plan(self):
        """Render the plan as comparable plain data."""
        return [
            (entry.kind, entry.owner, str(entry.dtype), entry.names)
            for entry in self._entries
        ]

    def _ready_key(self, record):
        """Sort key that prefers the measured ready order once one exists."""
        if self._ready_order:
            return (1, self._ready_order.get(record.name, _UNKNOWN_ORDINAL), record.name)
        # The first iteration has no observed ready order to build from. The
        # queue launches strictly in order, so a head that is not ready blocks
        # every later entry and forces all gradients to stay resident until
        # backward ends (+26GB/rank, OOM at GBS80). Reverse registration order
        # keeps the head genuinely ready first.
        return (0, record.reverse_position, record.name)

    def _bucket(self, records):
        """Split records into owner-aligned buckets of bounded master bytes."""
        return bucket_by_bytes(
            records, self._bucket_bytes, lambda record: record.master_bytes
        )

    def _build_plan(self, register_hooks=True):
        """Build a deterministic communication plan and optionally register hooks."""
        entries = []
        whole_tensor = sorted(self._registry.whole_tensor_records(), key=self._ready_key)
        for owner in range(self._world_size):
            owned = [record for record in whole_tensor if record.owner == owner]
            for group in split_by_dtype(owned, lambda record: record.grad_wire_dtype):
                for bucket in self._bucket(group):
                    entries.append(
                        GradientSyncEntry(
                            WHOLE_TENSOR,
                            tuple(bucket),
                            owner=owner,
                            dtype=bucket[0].grad_wire_dtype,
                        )
                    )
        entries.sort(key=lambda entry: max(self._ready_key(r) for r in entry.records))
        self._entries = entries
        self._entry_by_name = {
            record.name: entry for entry in entries for record in entry.records
        }
        if register_hooks:
            self._register_hooks()

    def _register_hooks(self):
        """Attach a post-accumulate-grad hook to every planned parameter."""
        for record in self._registry.records:
            if record.name not in self._entry_by_name:
                continue
            handle = record.compute.register_post_accumulate_grad_hook(
                lambda _parameter, name=record.name: self._on_gradient_ready(name)
            )
            self._handles.append(handle)

    def _assert_plan_matches_ranks(self):
        """Refuse to run a plan the ranks do not agree on."""
        assert_plan_matches_ranks(
            self.describe_plan(),
            self._group,
            self._world_size,
            self._registry.device,
            "gradient overlap plan differs across ranks",
        )

    def begin(self):
        """Arm the overlap plan so per-parameter hooks can launch collectives."""
        if not self._overlap:
            return
        if self._active:
            raise RuntimeError("gradient overlap step was already started")
        if any(
            entry.work is not None
            or entry.output_view is not None
            or entry.input_buffer is not None
            for entry in self._entries
        ):
            raise RuntimeError("previous gradient overlap step was not finished")
        if self._pending_ready_order is not None:
            self._ready_order = self._pending_ready_order
            self._pending_ready_order = None
            self._build_plan(register_hooks=False)
            self._assert_plan_matches_ranks()
            # Rebucketing produces new bucket sizes, and the pool is keyed by
            # exact numel, so every pooled buffer from the previous plan would
            # stay resident forever. Nothing is in flight here (the guard above
            # rejects an unfinished step), so the whole generation can go.
            self.pool.clear()
        self._next = 0
        self._inflight.clear()
        # Gradients from earlier accumulation micro-batches are already present,
        # but they are not safe to communicate until the same parameter's hook
        # runs in the final backward and adds its last contribution.
        self._present_names = {
            record.name
            for record in self._registry.records
            if record.compute.grad is not None
        }
        self._ready_names.clear()
        self._inflight_bytes = 0
        for parameter in self._registry.master.values():
            parameter.grad = None
        self._active = True

    def _on_gradient_ready(self, name):
        """Record readiness and launch every collective that became complete."""
        if self._overlap and self._active:
            self._present_names.add(name)
            self._ready_names.add(name)
            if self._record_order and name not in self._ready_ordinals:
                self._ready_ordinals[name] = self._ready_counter
                self._ready_counter += 1
            self._launch_ready(force=False)

    def _entry_is_ready(self, entry):
        """True once every member's final gradient contribution has landed."""
        return all(
            record.name in self._ready_names and record.compute.grad is not None
            for record in entry.records
        )

    @torch.no_grad()
    def _reclaim_completed(self):
        """Retire collectives that already finished, without stalling any stream."""
        while self._inflight:
            entry = self._inflight[0]
            if entry.work is not None and not entry.work.is_completed():
                break
            self._complete(self._inflight.pop(0))

    @torch.no_grad()
    def _launch_ready(self, force=False):
        """Issue collectives in plan order for as long as the head is ready."""
        while self._next < len(self._entries):
            entry = self._entries[self._next]
            if not force and not self._entry_is_ready(entry):
                break
            # Blocking on a NCCL event inside the autograd hook would stall every
            # later backward kernel on the compute stream, so in-flight collectives
            # are bounded by bytes and reclaimed only once they already completed --
            # never by waiting on a fixed queue depth.
            self._reclaim_completed()
            while self._inflight and self._inflight_bytes >= self._inflight_limit:
                self._complete(self._inflight.pop(0))
            self._launch(entry)
            self._inflight.append(entry)
            self._next += 1

    @torch.no_grad()
    def _launch(self, entry):
        """Pack one entry into a flat buffer and start its collective."""
        self._launch_whole_tensor(entry)

    @torch.no_grad()
    def _launch_whole_tensor(self, entry):
        """Reduce a bucket of whole tensors onto its owner."""
        device = self._registry.device
        total = sum(record.numel for record in entry.records)
        flat = self.pool.acquire(total, entry.dtype, device)
        offset = 0
        for record in entry.records:
            target = flat[offset : offset + record.numel]
            if record.compute.grad is None:
                target.zero_()
            else:
                target.copy_(record.compute.grad.detach().reshape(-1))
            offset += record.numel
        entry.work = (
            dist.reduce(flat, dst=entry.owner, group=self._group, async_op=True)
            if self._world_size > 1
            else None
        )
        entry.output_buffer = flat
        entry.output_view = flat
        entry.nbytes = flat.numel() * flat.element_size()
        self._inflight_bytes += entry.nbytes
        # Releasing the compute gradient as soon as it is packed is what keeps the
        # backward peak flat.
        for record in entry.records:
            record.compute.grad = None

    @torch.no_grad()
    def _complete(self, entry):
        """Wait for one collective, write the master gradient and free buffers."""
        if entry.work is not None:
            entry.work.wait()
        output = entry.output_view
        if self._rank == entry.owner:
            offset = 0
            for record in entry.records:
                master = record.master
                count = master.numel()
                # .float() is a no-op for an fp32 output and .div() always
                # allocates, so the pooled buffer is never written in place.
                master.grad = (
                    output[offset : offset + count]
                    .view_as(master)
                    .float()
                    .div(self._world_size)
                )
                offset += count
        self._inflight_bytes -= entry.nbytes
        self.pool.release(entry.output_buffer)
        if entry.input_pooled:
            self.pool.release(entry.input_buffer)
        entry.output_view = None
        entry.output_buffer = None
        entry.input_buffer = None
        entry.input_pooled = False
        entry.nbytes = 0
        entry.work = None

    @torch.no_grad()
    def finish(self):
        """Drain the in-flight gradient collectives, or reduce serially if off."""
        if not self._overlap:
            self.reduce_serial()
            return
        if not self._active:
            raise RuntimeError("gradient overlap step was not started")
        self._launch_ready(force=True)
        while self._inflight:
            self._complete(self._inflight.pop(0))
        self._clear_absent_master_grads(self._present_mask(self._present_names))
        for record in self._registry.records:
            record.compute.grad = None
        if self._record_order:
            self._capture_ready_order()
        self._active = False

    @torch.no_grad()
    def clear_pending(self):
        """Abandon any half-issued step so a later one starts from a clean slate.

        Only for the failure path: the collectives are not waited on, because a
        rank that raised mid-backward has no partner to synchronize with.
        """
        for entry in self._entries:
            entry.output_view = None
            entry.output_buffer = None
            entry.input_buffer = None
            entry.input_pooled = False
            entry.nbytes = 0
            entry.work = None
        self._inflight.clear()
        self._inflight_bytes = 0
        self._next = 0
        self._active = False
        self.pool.clear()

    @torch.no_grad()
    def _present_mask(self, names):
        """Agree across ranks on which parameters received a gradient at all."""
        present = torch.tensor(
            [record.name in names for record in self._registry.records],
            device=self._registry.device,
            dtype=torch.int32,
        )
        if self._world_size > 1:
            dist.all_reduce(present, op=dist.ReduceOp.MAX, group=self._group)
        return present.tolist()

    @torch.no_grad()
    def _clear_absent_master_grads(self, mask):
        """Drop master gradients for parameters no rank produced one for."""
        for record, has_grad in zip(self._registry.records, mask):
            if not has_grad and record.master is not None:
                record.master.grad = None

    @torch.no_grad()
    def _capture_ready_order(self):
        """Publish the measured backward-ready order for the next iteration."""
        self._record_order = False
        unknown = len(self._registry.records) + 1
        ordinals = torch.tensor(
            [
                self._ready_ordinals.get(record.name, unknown)
                for record in self._registry.records
            ],
            device=self._registry.device,
            dtype=torch.int32,
        )
        if self._world_size > 1:
            # Agree on one observation so every rank rebuilds the same plan;
            # _assert_plan_matches_ranks then verifies it did.
            dist.all_reduce(ordinals, op=dist.ReduceOp.MAX, group=self._group)
        self._pending_ready_order = {
            record.name: int(value)
            for record, value in zip(self._registry.records, ordinals.tolist())
        }

    @torch.no_grad()
    def reduce_serial(self):
        """Average gradients onto owners without overlapping the backward pass."""
        device = self._registry.device
        mask = self._present_mask(
            {
                record.name
                for record in self._registry.records
                if record.compute.grad is not None
            }
        )
        present = dict(zip((record.name for record in self._registry.records), mask))

        # Whole-tensor parameters are packed by owner. This preserves Muon's
        # complete-matrix ownership while replacing O(parameters) reductions
        # with O(owners * buckets) reductions.
        for owner in range(self._world_size):
            owned = [
                record
                for record in self._registry.whole_tensor_records()
                if present[record.name] and record.owner == owner
            ]
            for group in split_by_dtype(owned, lambda record: record.grad_wire_dtype):
                for bucket in self._bucket(group):
                    self._reduce_serial_bucket(bucket, owner, device)

        for record in self._registry.records:
            if not present[record.name] and record.master is not None:
                record.master.grad = None
            record.compute.grad = None

    @torch.no_grad()
    def _reduce_serial_bucket(self, bucket, owner, device):
        """Reduce one owner-aligned bucket and write the owner's master gradients."""
        wire_dtype = bucket[0].grad_wire_dtype
        if any(record.grad_wire_dtype != wire_dtype for record in bucket):
            raise RuntimeError("gradient bucket mixes wire dtypes")
        tensors = []
        for record in bucket:
            grad = record.compute.grad
            tensors.append(
                grad.detach().to(wire_dtype).reshape(-1)
                if grad is not None
                else torch.zeros(record.numel, device=device, dtype=wire_dtype)
            )
        flat = torch.cat(tensors)
        if self._world_size > 1:
            dist.reduce(flat, dst=owner, group=self._group)
        if self._rank != owner:
            return
        flat = flat.float().div_(self._world_size)
        offset = 0
        for record in bucket:
            count = record.master.numel()
            record.master.grad = flat[offset : offset + count].view_as(record.master).clone()
            offset += count


__all__ = [
    "WHOLE_TENSOR",
    "BufferPool",
    "GradientReducer",
    "GradientSyncEntry",
    "assert_plan_matches_ranks",
    "bucket_by_bytes",
    "split_by_dtype",
]
