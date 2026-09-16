# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Stable parameter metadata for replicated-compute optimizer-state-shard.

The registry runs the setup in explicitly ordered phases. The previous version
depended on statement order inside one constructor: the compute dtype was
rewritten in the same loop that created the fp32 masters, while the wire dtypes
were resolved later from whatever dtype the compute parameter happened to hold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from loongforge.embodied.train.trainers.optimizer_state_shard.ownership import (
    OwnershipPlanner,
    ParameterOwnership,
)

logger = logging.getLogger(__name__)

_GRAD_REDUCE_MODES = ("fp32", "bf16", "mixed", "compute")
_PARAM_SYNC_PRECISIONS = ("fp32", "bf16", "bf16_ef", "bf16_ef_delta", "fp8_e4m3_delta")
# Each parameter-publish precision decomposes into the three internal levers the
# collective paths consume: the fp32-master publish dtype, the bf16 error-feedback
# variant, and the delta quantization. Restricting the public surface to these five
# named combinations makes conflicting settings (bf16 and fp8 at once) unrepresentable.
_PARAM_SYNC_DECOMPOSITION = {
    "fp32": ("compute", "none", "none"),
    "bf16": ("bf16", "none", "none"),
    "bf16_ef": ("bf16", "error_feedback", "none"),
    "bf16_ef_delta": ("bf16", "error_feedback_delta", "none"),
    "fp8_e4m3_delta": ("compute", "none", "fp8_e4m3_delta"),
}
# Human-readable labels for the resolved wire dtypes, used by precision_summary.
# uint8 only ever carries the fp8 E4M3 delta payload (see _resolve_param_wire_dtype).
_DTYPE_LABEL = {
    torch.float32: "fp32",
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.uint8: "fp8_e4m3_delta",
}


@dataclass
class ParameterRecord:
    """Everything the collective paths need to know about one parameter."""

    name: str
    compute: nn.Parameter
    ownership: ParameterOwnership
    numel: int
    optimizer_kind: str
    is_comm_critical: bool
    parameter_class: str
    master: nn.Parameter | None = None
    grad_wire_dtype: torch.dtype = torch.float32
    param_wire_dtype: torch.dtype = torch.float32
    reverse_position: int = 0

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the full parameter shape."""
        return self.ownership.shape

    @property
    def owner(self) -> int:
        """Return the owning rank."""
        return self.ownership.owner

    @property
    def master_bytes(self) -> int:
        """Return the fp32 master footprint used for bucket accounting."""
        return self.numel * 4


class ParameterRegistry:
    """Own the compute replicas, the fp32 masters and the resolved wire dtypes."""

    def __init__(
        self,
        module,
        rank,
        world_size,
        parameter_policy,
        grad_reduce_dtype=None,
        param_sync_precision=None,
        param_sync_fp8_block=256,
        param_sync_fp8_reprime_interval=0,
        param_sync_fp8_include=None,
        param_sync_bf16_with_fp8=False,
    ):
        """Enumerate, plan, cast, create masters and resolve dtypes, in that order."""
        if parameter_policy is None:
            raise ValueError("parameter_policy is required")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.policy = parameter_policy
        self.grad_reduce_mode = self._validate(
            "grad_reduce_dtype", grad_reduce_dtype or "fp32", _GRAD_REDUCE_MODES
        )
        self.param_sync_precision = self._validate(
            "param_sync_precision",
            param_sync_precision or "fp32",
            _PARAM_SYNC_PRECISIONS,
        )
        # Fan the single public knob out into the internal levers the collective
        # paths read. param_sync_mode still governs the VLM (bf16-compute) publish;
        # only the fp32 action-expert masters see the compensation/quantization.
        (
            self.param_sync_mode,
            self.param_sync_compensation,
            self.param_sync_quantization,
        ) = _PARAM_SYNC_DECOMPOSITION[self.param_sync_precision]
        self.param_sync_fp8_block = int(param_sync_fp8_block)
        if (
            self.param_sync_fp8_block <= 0
            or self.param_sync_fp8_block & (self.param_sync_fp8_block - 1)
            or self.param_sync_fp8_block > 1 << 20
        ):
            raise ValueError(
                "param_sync_fp8_block must be a positive power of two <= 1048576"
            )
        self.param_sync_fp8_reprime_interval = int(param_sync_fp8_reprime_interval)
        if self.param_sync_fp8_reprime_interval < 0:
            raise ValueError("param_sync_fp8_reprime_interval must be non-negative")
        # FP8 whitelist (pure opt-in): only eligible parameters whose name
        # contains one of these substrings publish as an FP8 delta. Empty means no
        # parameter is quantized -- you must name them explicitly.
        self.param_sync_fp8_include = tuple(param_sync_fp8_include or ())
        # Off by default: extend the FP8 delta to the bf16-compute tower (VLM) by
        # keeping a per-rank fp32 shadow the delta accumulates into. Only meaningful
        # under the fp8 quantization lever; ignored otherwise.
        self.param_sync_bf16_with_fp8 = bool(param_sync_bf16_with_fp8)

        # Phase 1: enumerate trainable parameters in registration order.
        self.compute = {
            name: parameter
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        # Guard against silent misclassification: a marker that matches no
        # parameter usually means a submodule was renamed, which would drop its
        # parameters into the default class (or leave them un-quantized) with no
        # error. Warn on the main rank only -- some markers (lm_head,
        # shared_expert_gate) are legitimately absent when their feature is off,
        # so this stays a warning rather than a hard failure.
        if self.rank == 0:
            self._warn_unmatched_markers(list(self.compute))
        # Phase 2: plan ownership before anything mutates the parameters.
        self.ownership = OwnershipPlanner(world_size).plan(
            self.compute.items()
        )
        self.specs = {item.name: item for item in self.ownership}

        # Phase 3: classify precision-critical parameters and optimizer kind from
        # the original tensors, before any dtype is rewritten.
        self.records: list[ParameterRecord] = []
        for index, item in enumerate(self.ownership):
            compute = self.compute[item.name]
            self.records.append(
                ParameterRecord(
                    name=item.name,
                    compute=compute,
                    ownership=item,
                    numel=compute.numel(),
                    optimizer_kind=parameter_policy.optimizer_kind(item.name, compute),
                    is_comm_critical=parameter_policy.is_comm_precision_critical(
                        item.name, compute
                    ),
                    parameter_class=parameter_policy.classify(item.name, compute),
                    # named_parameters() follows the forward pass, so its reverse
                    # order approximates the order gradients become ready in backward.
                    reverse_position=-index,
                )
            )
        self._by_name = {record.name: record for record in self.records}

        # Phase 4: create the fp32 masters from the pre-cast values, then apply the
        # policy compute dtype. This order is load bearing: a master built after a
        # bf16 downcast would lose the bits the optimizer needs.
        self.master: dict[str, nn.Parameter] = {}
        for record in self.records:
            self._create_master(record)
            self._apply_compute_dtype(record)

        # Phase 5: resolve wire dtypes now that every compute dtype is final.
        for record in self.records:
            record.grad_wire_dtype = self._resolve_grad_wire_dtype(record)
            record.param_wire_dtype = self._resolve_param_wire_dtype(record)

    @staticmethod
    def _validate(name, value, allowed):
        """Reject an unsupported precision mode instead of falling back silently."""
        lowered = str(value).lower()
        if lowered not in allowed:
            raise ValueError(
                f"invalid {name}={lowered!r}; expected one of {', '.join(allowed)}"
            )
        return lowered

    def _create_master(self, record):
        """Create this rank's fp32 master, if it owns the parameter."""
        compute = record.compute
        if record.owner == self.rank:
            value = compute.detach().float().clone()
        else:
            return
        record.master = nn.Parameter(value, requires_grad=True)
        self.master[record.name] = record.master

    def _apply_compute_dtype(self, record):
        """Cast the compute replica to the dtype the policy asks for."""
        compute_dtype = self.policy.compute_dtype(record.name, record.compute)
        if record.compute.dtype != compute_dtype:
            record.compute.data = record.compute.data.to(compute_dtype)

    def _resolve_grad_wire_dtype(self, record):
        """Pick the wire dtype for one parameter's gradient reduction.

        Parameters the policy marks precision-critical (MoE router/gate, 1-D
        tensors) always travel at full precision, and buckets are split by dtype
        so a critical parameter can never be dragged into a downcast payload.
        """
        mode = self.grad_reduce_mode
        if mode == "fp32" or record.is_comm_critical:
            return torch.float32
        if mode in ("bf16", "mixed"):
            return torch.bfloat16
        if mode == "compute":
            return (
                torch.bfloat16
                if record.compute.dtype == torch.bfloat16
                else torch.float32
            )
        return torch.float32

    def _resolve_param_wire_dtype(self, record):
        """Pick the wire dtype for publishing one updated parameter to replicas."""
        if record.is_comm_critical:
            return torch.float32
        compute_dtype = record.compute.dtype
        # Single fp8 assignment site: uint8 is produced here and nowhere else, so
        # param_wire_dtype==uint8 is the authoritative "publishes as fp8" marker
        # every collective path keys off. It is reached only by non-critical
        # parameters in the configured FP8 scope, and only when the
        # param_sync_precision knob selected the fp8 quantization lever. An
        # fp32-compute action-expert weight uses its fp32 replica as the delta
        # reconstruction directly; a bf16-compute VLM weight has no fp32 anchor on
        # the wire, so it qualifies only under the opt-in bf16 shadow (the collective
        # keeps a per-rank fp32 shadow to accumulate the delta -- see
        # ParameterSynchronizer). This check precedes the bf16/fp16 early return so
        # the tower can reach it; fp32-compute behaviour is unchanged either way.
        if (
            self.param_sync_quantization == "fp8_e4m3_delta"
            and self._fp8_in_scope(record.name)
            and (
                compute_dtype == torch.float32
                or (
                    compute_dtype == torch.bfloat16
                    and self.param_sync_bf16_with_fp8
                )
            )
        ):
            return torch.uint8
        if compute_dtype in (torch.float16, torch.bfloat16):
            return compute_dtype
        # An fp32 compute parameter can still be published in bf16: the fp32
        # master keeps the update precision, replicas just receive a rounded
        # copy -- and all ranks, owner included, read the same rounded bytes
        # back, so the replicas stay bit-identical.
        if self.param_sync_mode == "bf16":
            return torch.bfloat16
        return torch.float32

    def _fp8_in_scope(self, name):
        """True when ``name`` is on the FP8 whitelist.

        ``param_sync_fp8_include`` is the sole scope and a pure opt-in: FP8 applies
        only to eligible parameters whose name contains one of its substrings. An
        empty whitelist means no parameter is quantized -- you must name them.
        Anything not whitelisted (or not eligible) keeps its normal publish (bf16
        or fp32).
        """
        return any(token in name for token in self.param_sync_fp8_include)

    def _warn_unmatched_markers(self, names):
        """Warn about configured substrings that match no parameter name.

        Covers the policy's own marker lists (compute/comm/optimizer routing) plus
        the FP8 whitelist. A miss almost always means a renamed submodule silently
        changed a parameter's precision or routing, so surface it at startup.
        """
        misses = []
        if hasattr(self.policy, "validate_markers"):
            misses.extend(self.policy.validate_markers(names))
        misses.extend(
            ("param_sync_fp8_include", token)
            for token in self.param_sync_fp8_include
            if not any(token in name for name in names)
        )
        for list_name, token in misses:
            logger.warning(
                "optimizer-state-shard: %s marker %r matched no parameter name; a "
                "renamed submodule may have silently changed precision/routing",
                list_name,
                token,
            )
        if self.param_sync_quantization != "none" and not self.param_sync_fp8_include:
            logger.warning(
                "optimizer-state-shard: param_sync_precision selects FP8 but "
                "param_sync_fp8_include is empty, so no parameter will be quantized"
            )

    def record(self, name) -> ParameterRecord:
        """Return the record registered under ``name``."""
        return self._by_name[name]

    @property
    def device(self):
        """Return the device the compute replicas live on."""
        return next(iter(self.compute.values())).device

    def named_master_parameters(self):
        """Return the ``(name, master parameter)`` pairs owned by this rank."""
        return list(self.master.items())

    def whole_tensor_records(self):
        """Return the parameter records, in registration order."""
        return list(self.records)

    def precision_summary(self):
        """Group parameters by class and resolved dtypes, one row per combination.

        Reads the resolved records, so it reports what the collectives will
        actually send this run -- fp8 appears only when the param_sync knob
        selected it -- rather than a static per-name guess. Rows are sorted by
        descending numel so the dominant classes read first.
        """
        total = sum(record.numel for record in self.records) or 1
        groups: dict[tuple, dict] = {}
        for record in self.records:
            key = (
                record.parameter_class,
                _DTYPE_LABEL.get(record.compute.dtype, str(record.compute.dtype)),
                _DTYPE_LABEL.get(record.grad_wire_dtype, str(record.grad_wire_dtype)),
                _DTYPE_LABEL.get(record.param_wire_dtype, str(record.param_wire_dtype)),
            )
            entry = groups.setdefault(key, {"params": 0, "numel": 0})
            entry["params"] += 1
            entry["numel"] += record.numel
        rows = [
            {
                "parameter_class": parameter_class,
                "compute": compute,
                "grad_reduce": grad_reduce,
                "param_sync": param_sync,
                "params": entry["params"],
                "numel": entry["numel"],
                "numel_ratio": entry["numel"] / total,
            }
            for (parameter_class, compute, grad_reduce, param_sync), entry
            in groups.items()
        ]
        rows.sort(key=lambda row: row["numel"], reverse=True)
        return rows


class MasterParameterView:
    """Minimal module-like view consumed by the vendored optimizer builder."""

    def __init__(self, named_parameters):
        """Store the ``(name, parameter)`` pairs this view exposes."""
        self._named_parameters = tuple(named_parameters)

    def named_parameters(self):
        """Yield the ``(name, parameter)`` pairs this view was built from."""
        return iter(self._named_parameters)

    def named_modules(self):
        """Yield nothing; the view has no module tree."""
        return iter(())


__all__ = ["MasterParameterView", "ParameterRecord", "ParameterRegistry"]
