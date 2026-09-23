# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Publish owner-updated fp32 masters back to every compute replica.

Each parameter is broadcast from the rank that owns its fp32 master to every
compute replica.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from loongforge.embodied.train.trainers.custom.replicated_sharded_training.gradient_reducer import (
    WHOLE_TENSOR,
    assert_plan_matches_ranks,
    bucket_by_bytes,
    split_by_dtype,
)

E4M3_MAX = 448.0
DEFAULT_BLOCK = 256
NUM_BLOCKS_PER_TILE = 8

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - validated before enabling the path.
    triton = None
    tl = None


def validate_runtime(device: torch.device, backend: str) -> None:
    """Fail early unless the selected process group supports fused FP8."""
    device = torch.device(device)
    if device.type != "cuda" or "nccl" not in str(backend).lower():
        raise RuntimeError(
            "fp8_e4m3_delta parameter sync requires a CUDA device and NCCL"
        )
    if triton is None or tl is None or not hasattr(tl, "float8e4nv"):
        raise RuntimeError(
            "fp8_e4m3_delta parameter sync requires Triton tl.float8e4nv"
        )
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("fp8_e4m3_delta requires PyTorch E4M3 support")
    if torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError("fp8_e4m3_delta requires compute capability >= 8.9")


if triton is not None:

    @triton.jit
    def _quantize_kernel(
        X,
        Q,
        S,
        numel,
        BLOCK: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Quantize contiguous fp32 deltas to E4M3 with one scale per block."""
        pid = tl.program_id(0).to(tl.int64)
        block0 = pid * NB
        block_offsets = tl.arange(0, NB).to(tl.int64)
        element_offsets = tl.arange(0, BLOCK).to(tl.int64)
        offsets = (block0 + block_offsets)[:, None] * BLOCK + element_offsets[None, :]
        mask = offsets < numel
        block_mask = block0 + block_offsets < (numel + BLOCK - 1) // BLOCK
        values = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(values), axis=1)
        scale = amax / 448.0
        inverse = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        quantized = (values * inverse[:, None]).to(tl.float8e4nv)
        tl.store(Q + offsets, quantized.to(tl.uint8, bitcast=True), mask=mask)
        tl.store(S + block0 + block_offsets, scale, mask=block_mask)

    @triton.jit
    def _dequantize_kernel(
        Q,
        S,
        OUT,
        numel,
        BLOCK: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Dequantize E4M3 bytes into a contiguous fp32 delta buffer."""
        pid = tl.program_id(0).to(tl.int64)
        block0 = pid * NB
        block_offsets = tl.arange(0, NB).to(tl.int64)
        element_offsets = tl.arange(0, BLOCK).to(tl.int64)
        offsets = (block0 + block_offsets)[:, None] * BLOCK + element_offsets[None, :]
        mask = offsets < numel
        block_mask = block0 + block_offsets < (numel + BLOCK - 1) // BLOCK
        bits = tl.load(Q + offsets, mask=mask, other=0).to(tl.uint8)
        values = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scales = tl.load(
            S + block0 + block_offsets, mask=block_mask, other=0.0
        )
        tl.store(OUT + offsets, values * scales[:, None], mask=mask)


def _grid(numel: int, block: int) -> tuple[int]:
    tile = block * NUM_BLOCKS_PER_TILE
    return ((numel + tile - 1) // tile,)


def quantize_into(delta, quantized, scales, block=DEFAULT_BLOCK) -> None:
    """Quantize a contiguous fp32 delta into preallocated buffers."""
    if triton is None:
        raise RuntimeError("fp8_e4m3_delta requires Triton")
    if not delta.is_contiguous() or delta.dtype is not torch.float32:
        raise ValueError("FP8 parameter delta must be contiguous fp32")
    _quantize_kernel[_grid(delta.numel(), block)](
        delta,
        quantized,
        scales,
        delta.numel(),
        BLOCK=block,
        NB=NUM_BLOCKS_PER_TILE,
    )


def dequantize_into(quantized, scales, output, block=DEFAULT_BLOCK) -> None:
    """Dequantize a contiguous FP8 payload into preallocated fp32 storage."""
    if triton is None:
        raise RuntimeError("fp8_e4m3_delta requires Triton")
    _dequantize_kernel[_grid(output.numel(), block)](
        quantized,
        scales,
        output,
        output.numel(),
        BLOCK=block,
        NB=NUM_BLOCKS_PER_TILE,
    )


@dataclass
class ParameterSyncEntry:
    """One in-flight parameter-publish collective and its scheduling position."""

    kind: str
    records: tuple
    owner: int | None
    is_early: bool
    position: int
    payload: object | None = None
    nbytes: int = 0
    work: object | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Return the parameter names this collective carries."""
        return tuple(record.name for record in self.records)


class ParameterSynchronizer:
    """Move updated master values onto the compute replicas of every rank."""

    def __init__(
        self,
        registry,
        adapter,
        group,
        rank,
        world_size,
        bucket_mb,
        overlap=True,
        sync_depth=2,
        inflight_bytes=2048 * 1024 * 1024,
        compensation="none",
        quantization="none",
        fp8_block=256,
        fp8_reprime_interval=0,
    ):
        """Record the collective geometry; the plan is built on first use."""
        self._registry = registry
        self._adapter = adapter
        self._group = group
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._bucket_bytes = int(bucket_mb) * 1024 * 1024
        self._overlap = bool(overlap)
        self._sync_depth = int(sync_depth)
        self._inflight_limit = int(inflight_bytes)
        if compensation not in ("none", "error_feedback", "error_feedback_delta"):
            raise ValueError(
                "parameter-sync compensation must be 'none', 'error_feedback' "
                "or 'error_feedback_delta'"
            )
        self._compensation = compensation
        if quantization not in ("none", "fp8_e4m3_delta"):
            raise ValueError(
                "parameter-sync quantization must be 'none' or 'fp8_e4m3_delta'"
            )
        if quantization != "none" and not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("fp8_e4m3_delta requires torch.float8_e4m3fn")
        self._quantization = quantization
        self._fp8_block = int(fp8_block)
        self._fp8_reprime_interval = int(fp8_reprime_interval)
        if self._fp8_block <= 0 or self._fp8_reprime_interval < 0:
            raise ValueError("invalid FP8 block or reprime interval")
        self._sync_generation = 0
        if self._quantization != "none":
            backend = dist.get_backend(group) if self._world_size > 1 else "nccl"
            validate_runtime(self._registry.device, backend)

        self._entries: list[ParameterSyncEntry] = []
        self._inflight: list[ParameterSyncEntry] = []
        self._next = 0
        self._inflight_bytes = 0
        self._updated: set[str] = set()
        # Serial-path queue, bounded so the full model is never materialized twice.
        self._pending: list[tuple] = []
        self._residuals: dict[str, torch.Tensor] = {}
        # Per-rank fp32 shadows for bf16-compute FP8 records. A bf16 replica cannot
        # accumulate a small FP8 delta without dead-zoning at bf16 ULP, so the delta
        # is accumulated here in fp32 and cast down into the compute replica for the
        # forward. Lazily created from the (identical across ranks) initial replica,
        # so every rank's shadow stays bit-identical as it adds the same delta bytes.
        self._fp8_shadows: dict[str, torch.Tensor] = {}

    @property
    def entries(self):
        """Return the current publish plan."""
        return self._entries

    def describe_plan(self):
        """Render the plan as comparable plain data."""
        return [
            (entry.kind, entry.owner, entry.is_early, entry.names)
            for entry in self._entries
        ]

    def invalidate(self):
        """Drop the cached plan, forcing a rebuild on the next step."""
        self._entries = []

    def _bucket(self, records):
        """Split records into buckets of bounded fp32 master bytes."""
        return bucket_by_bytes(
            records, self._bucket_bytes, lambda record: record.master_bytes
        )

    def _owner_buckets(self, split_by_update_timing):
        """Yield ``(owner, bucket, is_early)`` for the whole-tensor parameters.

        When the optimizer reports which masters it updates in-step, those are
        bucketed separately from the ones only ready after ``step()`` returns:
        mixing them would block the whole in-order launch queue behind the late
        group.
        """
        whole_tensor = self._registry.whole_tensor_records()
        groups = []
        for owner in range(self._world_size):
            owned = [record for record in whole_tensor if record.owner == owner]
            if not split_by_update_timing or self._adapter.in_step_names is None:
                groups.append((owner, owned, True))
                continue
            early = [r for r in owned if self._adapter.updates_in_step(r)]
            late = [r for r in owned if not self._adapter.updates_in_step(r)]
            groups.append((owner, early, True))
            groups.append((owner, late, False))
        for owner, owned, is_early in groups:
            for dtype_group in split_by_dtype(
                owned, lambda record: record.param_wire_dtype
            ):
                for bucket in self._bucket(dtype_group):
                    yield owner, bucket, is_early

    def _build_plan(self):
        """Build the publish plan and verify every rank agrees on it."""
        positions = {
            record.name: index for index, record in enumerate(self._registry.records)
        }
        entries = []
        for owner, bucket, is_early in self._owner_buckets(split_by_update_timing=True):
            entries.append(
                ParameterSyncEntry(
                    WHOLE_TENSOR,
                    tuple(bucket),
                    owner,
                    is_early,
                    max(positions[record.name] for record in bucket),
                )
            )
        # Late entries cannot be ready before the optimizer step returns, so they
        # go last and never block the early ones.
        entries.sort(key=lambda entry: (not entry.is_early, entry.position))
        self._entries = entries
        self._adapter.refresh()
        assert_plan_matches_ranks(
            self.describe_plan(),
            self._group,
            self._world_size,
            self._registry.device,
            "replicated-sharded parameter-sync plan differs across ranks; refusing to run "
            "overlapped parameter sync",
        )

    def _is_delta_record(self, record):
        """True when this parameter publishes as a bf16 delta with implicit EF.

        Restricted to fp32-compute, non-critical parameters on a bf16 wire:
        those are the ones a plain absolute bf16 publish pushes into the dead
        zone, and their fp32 replica can accumulate a delta without loss. The
        replica itself is the running reconstruction, so the quantization
        residual folds forward on its own and never needs a separate buffer.
        """
        return (
            self._compensation == "error_feedback_delta"
            and record.param_wire_dtype == torch.bfloat16
            and not record.is_comm_critical
            and record.compute.dtype == torch.float32
        )

    def _is_fp8_record(self, record):
        """True for records the registry resolved to the FP8 delta payload.

        ``param_wire_dtype`` is the single source of truth: the registry only
        assigns uint8 to non-critical, non-sharded fp32-compute parameters under
        the fp8 knob (see ParameterRegistry._resolve_param_wire_dtype), so
        re-deriving those conditions here would just duplicate that assignment.
        """
        return record.param_wire_dtype == torch.uint8

    def _is_fp8_shadow_record(self, record):
        """True for FP8 records whose compute replica is bf16 (the VLM tower).

        The registry only assigns uint8 to a bf16-compute parameter under the
        opt-in bf16 shadow, so the compute dtype alone tells the two FP8 flavours
        apart: fp32-compute reconstructs the delta in the replica itself, while a
        bf16-compute record reconstructs it in a per-rank fp32 shadow.
        """
        return self._is_fp8_record(record) and record.compute.dtype == torch.bfloat16

    def _fp8_shadow(self, record):
        """Return this rank's fp32 shadow for a bf16-compute FP8 record.

        Created on first use from the current (still-initial, replica-identical)
        compute value, so the owner (during encode) and every receiver (during
        land) seed the same fp32 anchor before any delta is applied.
        """
        shadow = self._fp8_shadows.get(record.name)
        if shadow is None:
            shadow = record.compute.detach().float().clone()
            self._fp8_shadows[record.name] = shadow
        return shadow

    def _fp8_delta_source(self, record):
        """Return the fp32 (master - reconstruction) delta the owner quantizes.

        The reconstruction is the fp32 compute replica for an fp32-compute record,
        or the per-rank fp32 shadow for a bf16-compute one.
        """
        master = record.master.detach()
        if self._is_fp8_shadow_record(record):
            return master - self._fp8_shadow(record)
        return master - record.compute

    def _fp8_force_full(self):
        """Periodically re-anchor the replica with an absolute BF16 value."""
        return (
            self._fp8_reprime_interval > 0
            and self._sync_generation % self._fp8_reprime_interval == 0
        )

    def _fp8_layout(self, bucket):
        """Return quantized and scale byte counts for one FP8 bucket."""
        values = sum(record.numel for record in bucket)
        scales = sum(
            (record.numel + self._fp8_block - 1) // self._fp8_block
            for record in bucket
        )
        return values, scales

    def _fp8_encode(self, value):
        """Encode an fp32 delta as E4M3 bytes plus one fp32 scale per block."""
        flat = value.reshape(-1)
        group_count = (flat.numel() + self._fp8_block - 1) // self._fp8_block
        quantized = torch.empty(flat.numel(), dtype=torch.uint8, device=flat.device)
        scales = torch.empty(group_count, dtype=torch.float32, device=flat.device)
        quantize_into(flat, quantized, scales, self._fp8_block)
        return quantized, scales

    def _fp8_decode(self, quantized, scales, shape):
        """Decode E4M3 bytes with block scales into the compute dtype."""
        numel = quantized.numel()
        decoded = torch.empty(numel, dtype=torch.float32, device=quantized.device)
        dequantize_into(quantized, scales, decoded, self._fp8_block)
        return decoded.view(shape)

    @torch.no_grad()
    def _build_whole_tensor_collective(self, bucket, owner):
        """Broadcast one owner-aligned bucket of whole parameters."""
        device = self._registry.device
        wire_dtype = bucket[0].param_wire_dtype
        if any(record.param_wire_dtype != wire_dtype for record in bucket):
            raise RuntimeError(
                "parameter-sync bucket mixes wire dtypes; buckets must be split by "
                "dtype before a collective is built"
            )
        is_fp8 = all(self._is_fp8_record(record) for record in bucket)
        if any(self._is_fp8_record(record) != is_fp8 for record in bucket):
            raise RuntimeError("FP8 buckets must contain only FP8 delta records")
        value_bytes, scale_count = self._fp8_layout(bucket) if is_fp8 else (0, 0)
        if is_fp8:
            total = (
                value_bytes + scale_count * 4
                if not self._fp8_force_full()
                else sum(record.numel for record in bucket) * 2
            )
        else:
            total = sum(record.numel for record in bucket)
        if self._rank == owner:
            values = []
            if is_fp8 and not self._fp8_force_full():
                encoded_values = []
                encoded_scales = []
                for record in bucket:
                    quantized, scales = self._fp8_encode(
                        self._fp8_delta_source(record)
                    )
                    encoded_values.append(quantized)
                    encoded_scales.append(scales.view(torch.uint8).reshape(-1))
                buffer = torch.cat(encoded_values + encoded_scales)
            elif is_fp8:
                values = [
                    record.master.detach().to(torch.bfloat16).view(torch.uint8).reshape(-1)
                    for record in bucket
                ]
                buffer = torch.cat(values)
            else:
                for record in bucket:
                    master = record.master.detach()
                    if self._is_delta_record(record):
                        wire = (master - record.compute).to(wire_dtype)
                    elif (
                        self._compensation == "error_feedback"
                        and wire_dtype == torch.bfloat16
                        and not record.is_comm_critical
                        and record.compute.dtype == torch.float32
                    ):
                        residual = self._residuals.get(record.name)
                        if residual is None:
                            residual = torch.zeros_like(master)
                            self._residuals[record.name] = residual
                        corrected = master + residual
                        wire = corrected.to(wire_dtype)
                        residual.copy_(corrected - wire.to(master.dtype))
                    else:
                        wire = master.to(wire_dtype)
                    values.append(wire.reshape(-1))
                buffer = torch.cat(values)
        else:
            buffer = torch.empty(total, dtype=wire_dtype, device=device)
        work = (
            dist.broadcast(buffer, src=owner, group=self._group, async_op=True)
            if self._world_size > 1
            else None
        )
        return (
            WHOLE_TENSOR,
            tuple(bucket),
            buffer,
            work,
            buffer.numel() * buffer.element_size(),
        )

    def compensation_state_dict(self):
        """Return owner-local error-feedback residuals on CPU."""
        return {
            name: value.detach().cpu().clone()
            for name, value in self._residuals.items()
        }

    @torch.no_grad()
    def load_compensation_state_dict(self, state):
        """Restore owner-local residuals, accepting checkpoints without them."""
        self._residuals.clear()
        for name, value in (state or {}).items():
            record = self._registry.record(name)
            if record.master is None:
                continue
            if value.shape != record.master.shape or value.dtype != torch.float32:
                raise RuntimeError(f"invalid compensation residual for {name}")
            self._residuals[name] = value.to(record.master.device)

    @torch.no_grad()
    def _land(self, records, payload, work):
        """Wait for one collective and write its result into the compute replicas."""
        if work is not None:
            work.wait()
        offset = 0
        if records and self._is_fp8_record(records[0]):
            value_bytes, scale_count = self._fp8_layout(records)
            if self._fp8_force_full():
                offset = 0
                for record in records:
                    nbytes = record.numel * 2
                    chunk = payload[offset : offset + nbytes].view(torch.bfloat16)
                    record.compute.copy_(chunk.view_as(record.compute))
                    if self._is_fp8_shadow_record(record):
                        # Re-anchor the shadow to the same absolute value the
                        # replica just took, so the next delta is measured from it.
                        self._fp8_shadow(record).copy_(
                            chunk.view_as(record.compute).float()
                        )
                    offset += nbytes
                return
            value_offset = 0
            scale_offset = value_bytes
            for record in records:
                q_count = record.numel
                group_count = (record.numel + self._fp8_block - 1) // self._fp8_block
                quantized = payload[value_offset : value_offset + q_count]
                scales = payload[
                    scale_offset : scale_offset + group_count * 4
                ].view(torch.float32)
                delta = self._fp8_decode(quantized, scales, record.compute.shape)
                if self._is_fp8_shadow_record(record):
                    # Accumulate the delta in the fp32 shadow (every rank adds the
                    # same bytes onto the same shadow, so they stay identical), then
                    # cast the shadow down for the bf16 forward.
                    shadow = self._fp8_shadow(record)
                    shadow.add_(delta)
                    record.compute.copy_(shadow.to(record.compute.dtype))
                else:
                    record.compute.add_(delta.to(record.compute.dtype))
                value_offset += q_count
                scale_offset += group_count * 4
            return
        for record in records:
            chunk = payload[offset : offset + record.numel].view_as(record.compute)
            if self._is_delta_record(record):
                # The broadcast carries a delta, not an absolute value; add it
                # into the fp32 replica, which is the running reconstruction and
                # the same value every rank holds (identical wire bytes + start).
                record.compute.add_(chunk.to(record.compute.dtype))
            else:
                record.compute.copy_(chunk)
            offset += record.numel

    def begin(self):
        """Reset the publish plan so optimizer updates can start collectives."""
        self._sync_generation += 1
        if not self._overlap:
            return
        if not self._entries:
            self._build_plan()
        if self._inflight:
            raise RuntimeError("previous parameter sync overlap was not finished")
        self._next = 0
        self._inflight_bytes = 0
        self._updated.clear()

    def note_updated(self, name):
        """Mark one master as updated and launch any publish that became ready."""
        if not self._overlap or name is None:
            return
        self._updated.add(name)
        self._launch_ready(force=False)

    def _entry_is_ready(self, entry):
        """True once every locally owned master in the entry has been updated."""
        for record in entry.records:
            if record.master is None:
                # Not owned here; this rank only receives, so nothing to wait for.
                continue
            if record.name in self._updated:
                continue
            if record.master.grad is None:
                # The optimizer skips grad-less masters, so it never reports them.
                continue
            return False
        return True

    @torch.no_grad()
    def _reclaim_completed(self):
        """Retire publishes that already finished, without stalling any stream."""
        while self._inflight:
            entry = self._inflight[0]
            if entry.work is not None and not entry.work.is_completed():
                break
            self._finish_entry(self._inflight.pop(0))

    @torch.no_grad()
    def _launch_ready(self, force=False):
        """Issue publishes in plan order for as long as the head is ready."""
        while self._next < len(self._entries):
            entry = self._entries[self._next]
            if not force and not (entry.is_early and self._entry_is_ready(entry)):
                break
            self._reclaim_completed()
            while self._inflight and self._inflight_bytes >= self._inflight_limit:
                self._finish_entry(self._inflight.pop(0))
            built = self._build_whole_tensor_collective(list(entry.records), entry.owner)
            entry.payload = built[2]
            entry.work = built[3]
            entry.nbytes = built[4]
            self._inflight_bytes += entry.nbytes
            self._inflight.append(entry)
            self._next += 1

    @torch.no_grad()
    def _finish_entry(self, entry):
        """Land one in-flight publish and release its accounting."""
        self._land(entry.records, entry.payload, entry.work)
        self._inflight_bytes -= entry.nbytes
        entry.payload = None
        entry.work = None
        entry.nbytes = 0

    @torch.no_grad()
    def finish(self):
        """Drain the in-flight publishes, or publish serially if overlap is off."""
        if not self._overlap:
            self.start_serial()
            self.finish_serial()
            return
        self._launch_ready(force=True)
        while self._inflight:
            self._finish_entry(self._inflight.pop(0))

    def _enqueue_serial(self, entry):
        """Queue one serial collective, keeping at most ``sync_depth`` in flight."""
        self._pending.append(entry)
        if len(self._pending) > self._sync_depth:
            self._land(*self._pending.pop(0))

    @torch.no_grad()
    def start_serial(self):
        """Launch bounded owner-to-replica parameter synchronization."""
        if self._pending:
            raise RuntimeError("parameter synchronization is already pending")
        for owner, bucket, _ in self._owner_buckets(split_by_update_timing=False):
            self._enqueue_serial(self._build_whole_tensor_collective(bucket, owner)[1:4])

    @torch.no_grad()
    def finish_serial(self):
        """Wait for every queued parameter broadcast to land."""
        while self._pending:
            self._land(*self._pending.pop(0))

    @torch.no_grad()
    def clear_pending(self):
        """Abandon any half-issued publication so a later step starts clean."""
        for entry in self._entries:
            entry.payload = None
            entry.work = None
            entry.nbytes = 0
        self._inflight.clear()
        self._inflight_bytes = 0
        self._next = 0
        self._pending.clear()


__all__ = ["ParameterSyncEntry", "ParameterSynchronizer"]
