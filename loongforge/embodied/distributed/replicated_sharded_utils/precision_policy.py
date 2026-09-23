# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Config-driven parameter precision policy shared by replicated-sharded model integrations.

This is the contract between a model and the replicated-sharded training strategy
-- the model declares what each parameter *is*, the strategy decides how it moves
on the wire -- so it lives next to that strategy's config in
``replicated_sharded_utils``. ``model/`` holds model files only; a model
implementation supplies marker lists by subclassing ``MarkerParameterPolicy``.

The replicated-sharded registry is model-agnostic: it consumes a parameter policy through a
small duck-typed contract (``compute_dtype``, ``is_comm_precision_critical``,
``parameter_class``, ``optimizer_kind``). This base implements that contract from
two declarative marker lists so a new model only has to describe its parameters,
not add branching in Python.

Two orthogonal precision axes:

    compute_fp32_markers  -- parameters whose name matches compute in fp32;
                             everything else computes in bf16. There is no fp8
                             compute path (that needs Transformer Engine), so the
                             compute axis is fp32/bf16 only.
    comm_critical_markers -- parameters that must keep fp32 collectives (both
                             gradient reduce and parameter publish). Everything
                             else is downcast-eligible; the concrete bf16/fp8
                             choice is the global knob resolved in the registry,
                             never declared per parameter here.

``force_comm_critical_below_ndim`` is the cross-cutting rule: 1-D tensors (norms,
biases) always keep fp32 collectives -- precision sensitive and too small for the
bytes to matter.

The axes are deliberately independent: a parameter may compute in fp32 yet
downcast its collectives (an action-expert weight), or compute in bf16 yet keep
fp32 collectives (a 1-D norm). Folding them into one ordered "kind" would lose
exactly those combinations, so a model supplies one list per axis.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Protocol
import logging
from types import SimpleNamespace

POLICY_VERSION = 1
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16,
          "fp16": torch.float16, "fp8_e4m3_delta": torch.uint8}
# The single definition of the two public precision enums: the registry, the
# collective plan and the wire resolver all validate against these.
GRAD_REDUCE_MODES = ("fp32", "bf16", "mixed", "compute")
# Each parameter-publish precision decomposes into the three internal levers the
# collective paths consume: the fp32-master publish dtype, the bf16 error-feedback
# variant, and the delta quantization. Restricting the public surface to these five
# named combinations makes conflicting settings (bf16 and fp8 at once) unrepresentable.
PARAM_SYNC_DECOMPOSITION = {
    "fp32": ("compute", "none", "none"),
    "bf16": ("bf16", "none", "none"),
    "bf16_ef": ("bf16", "error_feedback", "none"),
    "bf16_ef_delta": ("bf16", "error_feedback_delta", "none"),
    "fp8_e4m3_delta": ("compute", "none", "fp8_e4m3_delta"),
}
PARAM_SYNC_PRECISIONS = tuple(PARAM_SYNC_DECOMPOSITION)


def dtype_label(dtype: torch.dtype) -> str:
    """Return the config-level name of a resolved wire dtype."""
    return next(key for key, value in DTYPES.items() if value == dtype)


@dataclass(frozen=True)
class ParameterCapability:
    """Resolved, serializable precision contract; never holds mutable tensors."""

    name: str
    shape: tuple[int, ...]
    numel: int
    compute_dtype: str
    grad_reduce_dtype: str
    parameter_sync_dtype: str
    optimizer_kind: str
    communication_critical: bool
    owner_policy: str = "greedy_whole_tensor"
    requires_grad: bool = True
    parameter_class: str = "default"


@dataclass(frozen=True)
class ParameterMetadata:
    name: str
    parameter: nn.Parameter


class ParameterPolicy(Protocol):
    def classify(self, metadata: ParameterMetadata) -> ParameterCapability: ...
    def validate(self, capabilities) -> None: ...


class WirePrecisionResolver:
    """Resolve one parameter's wire dtypes from the run's precision plan.

    Both knobs must be set explicitly: an unset value is not a precision. In
    particular, publish mode 'fp32' means compute precision for non-critical
    replicas, not unconditional FP32 transmission.
    """

    def __init__(
        self,
        *,
        grad_reduce_dtype,
        param_sync_precision,
        param_sync_fp8_include=(),
        param_sync_bf16_with_fp8=None,
    ):
        """Record the plan and reject an unsupported combination up front."""
        grad = str(grad_reduce_dtype).lower()
        precision = str(param_sync_precision).lower()
        if grad not in GRAD_REDUCE_MODES:
            raise ValueError(f"invalid grad_reduce_dtype={grad!r}")
        if precision not in PARAM_SYNC_DECOMPOSITION:
            raise ValueError(f"invalid param_sync_precision={precision!r}")
        include = param_sync_fp8_include or ()
        if isinstance(include, str) or any(not isinstance(s, str) or not s for s in include):
            raise ValueError("param_sync_fp8_include must contain non-empty markers")
        self.grad_reduce_mode = grad
        self.param_sync_precision = precision
        self.param_sync_mode, _, self.param_sync_quantization = PARAM_SYNC_DECOMPOSITION[precision]
        self.param_sync_fp8_include = tuple(include)
        self.param_sync_bf16_with_fp8 = (
            False if param_sync_bf16_with_fp8 is None else bool(param_sync_bf16_with_fp8)
        )

    def resolve_grad_wire_dtype(self, record):
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

    def resolve_param_wire_dtype(self, record):
        if record.is_comm_critical:
            return torch.float32
        compute_dtype = record.compute.dtype
        if (
            self.param_sync_quantization == "fp8_e4m3_delta"
            and self.fp8_in_scope(record.name)
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
        if self.param_sync_mode == "bf16":
            return torch.bfloat16
        return torch.float32

    def fp8_in_scope(self, name):
        return any(token in name for token in self.param_sync_fp8_include)


class CapabilityPolicyAdapter:
    """Resolve a model's marker policy into per-parameter capabilities.

    The model-side policy answers what a parameter *is* (compute dtype,
    comm-critical, optimizer kind, class label); this adapter adds the run's
    precision plan, which is what turns those answers into concrete wire dtypes.
    """

    def __init__(self, policy, runtime_config):
        self.markers = policy
        self.wire = WirePrecisionResolver(**runtime_config)

    def classify(self, metadata):
        """Return the capability of one parameter under the run's precision plan."""
        name, parameter = metadata.name, metadata.parameter
        # Frozen tensors are neither cast nor given a master by the registry, so
        # they keep whatever dtype they were built with.
        dtype = (
            self.markers.compute_dtype(name, parameter)
            if parameter.requires_grad
            else parameter.dtype
        )
        compute_dtype = dtype_label(dtype) if dtype in DTYPES.values() else None
        if compute_dtype not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"invalid compute dtype for {name}: {compute_dtype}")
        communication_critical = self.markers.is_comm_precision_critical(name, parameter)
        record = SimpleNamespace(
            name=name,
            compute=SimpleNamespace(dtype=DTYPES[compute_dtype]),
            is_comm_critical=communication_critical,
        )
        return ParameterCapability(
            name=name,
            shape=tuple(parameter.shape),
            numel=parameter.numel(),
            compute_dtype=compute_dtype,
            grad_reduce_dtype=dtype_label(self.wire.resolve_grad_wire_dtype(record)),
            parameter_sync_dtype=dtype_label(self.wire.resolve_param_wire_dtype(record)),
            optimizer_kind=self.markers.optimizer_kind(name, parameter),
            communication_critical=communication_critical,
            requires_grad=parameter.requires_grad,
            parameter_class=self.markers.parameter_class(name, parameter),
        )

    def validate(self, capabilities):
        if hasattr(self.markers, "validate_markers"):
            for group, marker in self.markers.validate_markers(c.name for c in capabilities):
                logging.getLogger(__name__).warning(
                    "%s marker %r matched no parameter name", group, marker
                )


def build_parameter_policy(runtime_config, *, policy):
    """Wrap a model's marker policy in the capability-resolving adapter.

    The model supplies its own marker policy (the ``compute_dtype`` /
    ``is_comm_precision_critical`` / ``optimizer_kind`` / ``parameter_class``
    contract); this module never imports model implementations. ``runtime_config``
    carries the resolved collective precision plan the adapter needs to compute
    wire dtypes.
    """
    return CapabilityPolicyAdapter(policy, runtime_config)


def resolve_parameter_capabilities(module, policy):
    """Resolve and validate every parameter before any tensor mutation."""
    capabilities = tuple(
        policy.classify(ParameterMetadata(name, parameter))
        for name, parameter in module.named_parameters()
    )
    for (name, parameter), cap in zip(module.named_parameters(), capabilities):
        if not isinstance(cap, ParameterCapability):
            raise TypeError("policy.classify must return ParameterCapability")
        if (cap.name, cap.shape, cap.numel, cap.requires_grad) != (
            name, tuple(parameter.shape), parameter.numel(), parameter.requires_grad
        ):
            raise ValueError(f"policy metadata mismatch for {name}")
        if cap.compute_dtype not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"unsupported compute dtype for {name}")
        if cap.grad_reduce_dtype not in ("fp32", "bf16"):
            raise ValueError(f"unsupported gradient dtype for {name}")
        if cap.parameter_sync_dtype not in DTYPES:
            raise ValueError(f"unsupported sync dtype for {name}")
        if cap.optimizer_kind not in ("adamw", "muon"):
            raise ValueError(f"unsupported optimizer kind for {name}")
        if cap.owner_policy != "greedy_whole_tensor":
            raise ValueError(f"unsupported owner policy for {name}")
        if type(cap.communication_critical) is not bool:
            raise ValueError(f"communication_critical must be bool for {name}")
        if cap.communication_critical and (
            cap.grad_reduce_dtype != "fp32" or cap.parameter_sync_dtype != "fp32"
        ):
            raise ValueError(f"critical parameter {name} requires FP32 communication")
    policy.validate(capabilities)
    return capabilities

# Labels used by classify() when a model does not supply its own naming. Keyed by
# (fp32_compute, comm_critical); shown in the registry's precision summary.
_DEFAULT_LABELS = {
    (True, False): "fp32_compute_downcast_comm",
    (True, True): "fp32_compute_fp32_comm",
    (False, True): "bf16_compute_fp32_comm",
    (False, False): "bf16_compute_downcast_comm",
}


class MarkerParameterPolicy:
    """Interpret two per-axis marker lists as the replicated-sharded policy contract."""

    def __init__(
        self,
        *,
        compute_fp32_markers: tuple[str, ...] = (),
        comm_critical_markers: tuple[str, ...] = (),
        labels: dict | None = None,
        force_comm_critical_below_ndim: int = 2,
        adamw_markers: tuple[str, ...] = (),
        muon_ndims: tuple[int, ...] = (2, 3),
    ):
        """Record the per-model markers that drive both precision axes."""
        def markers(values):
            if isinstance(values, str) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError("markers must be a sequence of non-empty strings")
            return tuple(dict.fromkeys(values))
        self._compute_fp32_markers = markers(compute_fp32_markers)
        self._comm_critical_markers = markers(comm_critical_markers)
        self._labels = dict(labels) if labels is not None else dict(_DEFAULT_LABELS)
        self._force_comm_critical_below_ndim = int(force_comm_critical_below_ndim)
        if self._force_comm_critical_below_ndim < 0:
            raise ValueError("force_comm_critical_below_ndim must be non-negative")
        self._adamw_markers = markers(adamw_markers)
        self._muon_ndims = tuple(muon_ndims)

    def _is_fp32_compute(self, name: str) -> bool:
        return any(marker in name for marker in self._compute_fp32_markers)

    def compute_dtype(self, name: str, parameter: nn.Parameter) -> torch.dtype:
        """Forward/backward dtype for one parameter (fp32 or bf16)."""
        return torch.float32 if self._is_fp32_compute(name) else torch.bfloat16

    def is_comm_precision_critical(self, name: str, parameter: nn.Parameter) -> bool:
        """True when the parameter must keep full-precision collectives."""
        if parameter.ndim < self._force_comm_critical_below_ndim:
            return True
        return any(marker in name for marker in self._comm_critical_markers)

    def parameter_class(self, name: str, parameter: nn.Parameter) -> str:
        """Return the class label for the (compute, comm) pair this parameter hits."""
        key = (
            self._is_fp32_compute(name),
            self.is_comm_precision_critical(name, parameter),
        )
        return self._labels[key]

    def optimizer_kind(self, name: str, parameter: nn.Parameter) -> str:
        """Route matmul tensors to muon unless an adamw marker matches."""
        lowered = name.lower()
        if parameter.ndim not in self._muon_ndims:
            return "adamw"
        if any(marker.lower() in lowered for marker in self._adamw_markers):
            return "adamw"
        return "muon"

    def validate_markers(self, names):
        """Return ``(list_name, marker)`` pairs whose substring matched no name.

        A configured marker that hits nothing usually means the model renamed a
        submodule, which would silently drop its parameters into the default class
        (bf16 compute / downcast collectives / Muon). Reporting the misses lets the
        caller surface that at startup instead of shipping a misclassification.
        """
        names = list(names)
        misses = []
        for list_name, markers in (
            ("compute_fp32_markers", self._compute_fp32_markers),
            ("comm_critical_markers", self._comm_critical_markers),
            ("adamw_markers", self._adamw_markers),
        ):
            misses.extend(
                (list_name, marker)
                for marker in markers
                if not any(
                    marker.lower() in name.lower() if list_name == "adamw_markers"
                    else marker in name for name in names
                )
            )
        return misses


__all__ = [
    "CapabilityPolicyAdapter",
    "GRAD_REDUCE_MODES",
    "MarkerParameterPolicy",
    "PARAM_SYNC_DECOMPOSITION",
    "PARAM_SYNC_PRECISIONS",
    "ParameterCapability",
    "ParameterMetadata",
    "ParameterPolicy",
    "WirePrecisionResolver",
    "build_parameter_policy",
    "dtype_label",
    "resolve_parameter_capabilities",
]
