# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Stable parameter metadata for replicated-sharded training.

The registry runs the setup in explicitly ordered phases. The previous version
depended on statement order inside one constructor: the compute dtype was
rewritten in the same loop that created the fp32 masters, while the wire dtypes
were resolved later from whatever dtype the compute parameter happened to hold.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn

from loongforge.embodied.model.precision_policy import (
    POLICY_VERSION,
    PARAM_SYNC_DECOMPOSITION,
    DTYPES,
    build_parameter_policy,
    resolve_parameter_capabilities,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParameterOwnership:
    """Which rank owns a parameter's fp32 master.

    Kept as the on-the-wire ownership schema: ``state_dict`` serializes these
    fields, so renaming or reordering them would invalidate the checkpoint
    compatibility check.
    """

    name: str
    shape: tuple[int, ...]
    owner: int


def assign_parameter_owners(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    world_size: int,
):
    """Assign every full tensor to the least-loaded owner rank."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    items = list(named_parameters)
    loads = [0] * world_size
    result = []
    for name, parameter in items:
        shape = tuple(parameter.shape)
        owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
        loads[owner] += parameter.numel() * 4
        result.append(ParameterOwnership(name, shape, owner))
    return result


class OwnershipPlanner:
    """Turn named parameters into whole-tensor ownership records.

    Only the validated greedy scheme is implemented. Reordering owners changes
    Muon's same-shape megabatch grouping (measured up to 2% gradient-norm drift),
    so an alternative balancing scheme belongs behind a new mode, never as a
    tweak here.
    """

    def __init__(self, world_size):
        """Record the target world size."""
        self._world_size = int(world_size)

    def plan(self, named_parameters):
        """Return the ownership records, in registration order."""
        return assign_parameter_owners(named_parameters, self._world_size)

_GRAD_REDUCE_MODES = ("fp32", "bf16", "mixed", "compute")
_PARAM_SYNC_PRECISIONS = ("fp32", "bf16", "bf16_ef", "bf16_ef_delta", "fp8_e4m3_delta")
# Each parameter-publish precision decomposes into the three internal levers the
# collective paths consume: the fp32-master publish dtype, the bf16 error-feedback
# variant, and the delta quantization. Restricting the public surface to these five
# named combinations makes conflicting settings (bf16 and fp8 at once) unrepresentable.
_PARAM_SYNC_DECOMPOSITION = PARAM_SYNC_DECOMPOSITION
# Human-readable labels for the resolved wire dtypes, used by precision_summary.
# uint8 only carries the fp8 E4M3 delta payload resolved by WirePrecisionResolver.
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
        param_sync_bf16_with_fp8=None,
    ):
        """Resolve, plan, create masters from original values, then cast replicas."""
        if parameter_policy is None:
            raise ValueError("parameter_policy is required")
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be within a positive world_size")
        # The precision plan has a single source: the resolved collective config
        # the manager passes down as these concrete arguments. The model policy
        # only describes what each parameter *is* (compute dtype, comm-critical,
        # optimizer kind), so it is wrapped once here with that plan to produce the
        # capability-resolving adapter -- no reconciliation between two configs.
        runtime_config = dict(
            grad_reduce_dtype=grad_reduce_dtype or "fp32",
            param_sync_precision=param_sync_precision or "fp32",
            param_sync_fp8_include=param_sync_fp8_include,
            param_sync_bf16_with_fp8=param_sync_bf16_with_fp8,
        )
        if hasattr(parameter_policy, "compute_dtype"):
            parameter_policy = build_parameter_policy(
                runtime_config=runtime_config, legacy_policy=parameter_policy,
            )
        self.policy = parameter_policy
        self._module = module
        self.capabilities = resolve_parameter_capabilities(
            module,
            parameter_policy,
        )
        self._capability_by_name = {item.name: item for item in self.capabilities}
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
        # Public policies can supply capabilities directly, bypassing the legacy
        # wire resolver. Reject an unsupported payload before creating any master.
        # The converse is valid: FP8 mode also carries non-FP8/critical records.
        for cap in self.capabilities:
            if (
                cap.parameter_sync_dtype == "fp8_e4m3_delta"
                and self.param_sync_quantization != "fp8_e4m3_delta"
            ):
                raise ValueError(
                    f"FP8 capability for {cap.name} requires FP8 parameter-sync quantization"
                )
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
                    optimizer_kind=self._capability_by_name[item.name].optimizer_kind,
                    is_comm_critical=self._capability_by_name[item.name].communication_critical,
                    parameter_class=self._capability_by_name[item.name].parameter_class,
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

        # Phase 5: collectives consume the same validated capability as compute.
        for record in self.records:
            cap = self._capability_by_name[record.name]
            record.grad_wire_dtype = DTYPES[cap.grad_reduce_dtype]
            record.param_wire_dtype = DTYPES[cap.parameter_sync_dtype]

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
        compute_dtype = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self._capability_by_name[record.name].compute_dtype]
        if record.compute.dtype != compute_dtype:
            record.compute.data = record.compute.data.to(compute_dtype)

    def manifest(self):
        """Return the checkpoint contract for all parameters, including owner."""
        parameters = dict(self._module.named_parameters())
        if set(parameters) != set(self._capability_by_name):
            raise RuntimeError("model parameter names changed after setup")
        for name, parameter in parameters.items():
            cap = self._capability_by_name[name]
            if (
                tuple(parameter.shape) != cap.shape
                or parameter.dtype != DTYPES[cap.compute_dtype]
                or parameter.requires_grad != cap.requires_grad
                or (cap.requires_grad and parameter is not self.compute[name])
            ):
                raise RuntimeError(f"model parameter contract changed: {name}")
        for record in self.records:
            cap = self._capability_by_name[record.name]
            if (
                tuple(record.compute.shape) != cap.shape
                or record.compute.dtype != DTYPES[cap.compute_dtype]
                or record.grad_wire_dtype != DTYPES[cap.grad_reduce_dtype]
                or record.param_wire_dtype != DTYPES[cap.parameter_sync_dtype]
                or record.optimizer_kind != cap.optimizer_kind
            ):
                raise RuntimeError(f"parameter contract changed after setup: {record.name}")
            if record.master is not None and (
                record.master.dtype != torch.float32 or tuple(record.master.shape) != cap.shape
            ):
                raise RuntimeError(f"invalid FP32 master: {record.name}")
        return {
            "policy_version": POLICY_VERSION,
            "world_size": self.world_size,
            "wire_config": {
                "grad_reduce_dtype": self.grad_reduce_mode,
                "param_sync_precision": self.param_sync_precision,
                "fp8_block": self.param_sync_fp8_block,
                "fp8_reprime_interval": self.param_sync_fp8_reprime_interval,
            },
            "parameters": [
                {
                    **asdict(cap),
                    "shape": list(cap.shape),
                    "owner_rank": self.specs[cap.name].owner if cap.requires_grad else None,
                }
                for cap in self.capabilities
            ],
        }

    def optimizer_kind(self, name, parameter):
        """Legacy optimizer routing adapter backed only by resolved capabilities."""
        return self._capability_by_name[name].optimizer_kind

    def validate_manifest(self, saved, *, allow_reshard=False, saved_world_size=None):
        """Validate before loading tensors; only owners may change on reshard."""
        if saved is None:
            logger.warning("Legacy checkpoint has no parameter manifest; precision validation unavailable")
            return
        current = self.manifest()
        if not isinstance(saved, dict) or saved.get("policy_version") != POLICY_VERSION:
            raise RuntimeError("checkpoint parameter policy version mismatch")
        if saved.get("wire_config") != current["wire_config"]:
            raise RuntimeError("checkpoint wire configuration mismatch")
        world = saved.get("world_size")
        if type(world) is not int or world < 1:
            raise RuntimeError("invalid manifest world_size")
        if saved_world_size is not None and world != saved_world_size:
            raise RuntimeError("manifest/checkpoint world_size mismatch")
        if world != self.world_size and not allow_reshard:
            raise RuntimeError("manifest world_size mismatch")
        rows = saved.get("parameters")
        if not isinstance(rows, list) or len(rows) != len(current["parameters"]):
            raise RuntimeError("manifest parameter count mismatch")
        # Registration order is part of the greedy owner and optimizer contract.
        for row, expected in zip(rows, current["parameters"]):
            if not isinstance(row, dict):
                raise RuntimeError("invalid manifest parameter entry")
            owner = row.get("owner_rank")
            if expected["requires_grad"]:
                if type(owner) is not int or not 0 <= owner < world:
                    raise RuntimeError(f"invalid saved owner for {expected['name']}")
            elif owner is not None:
                raise RuntimeError(f"frozen parameter has owner: {expected['name']}")
            normalized = dict(row)
            if world != self.world_size and allow_reshard:
                normalized["owner_rank"] = expected["owner_rank"]
            if normalized != expected:
                raise RuntimeError(f"parameter manifest mismatch for {expected['name']}")

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
                "replicated-sharded: %s marker %r matched no parameter name; a "
                "renamed submodule may have silently changed precision/routing",
                list_name,
                token,
            )
        if self.param_sync_quantization != "none" and not self.param_sync_fp8_include:
            logger.warning(
                "replicated-sharded: param_sync_precision selects FP8 but "
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
    """Minimal module-like view consumed by ``split_muon_adamw_params``."""

    def __init__(self, named_parameters):
        """Store the ``(name, parameter)`` pairs this view exposes."""
        self._named_parameters = tuple(named_parameters)

    def named_parameters(self):
        """Yield the ``(name, parameter)`` pairs this view was built from."""
        return iter(self._named_parameters)

    def named_modules(self):
        """Yield nothing; the view has no module tree."""
        return iter(())


__all__ = [
    "MasterParameterView",
    "OwnershipPlanner",
    "ParameterOwnership",
    "ParameterRecord",
    "ParameterRegistry",
    "assign_parameter_owners",
]
