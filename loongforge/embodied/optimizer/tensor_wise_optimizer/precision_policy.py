# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Per-parameter precision policy shared by tensor_wise model integrations.

The model declares what each parameter *is*; the strategy decides how it moves on
the wire. A model supplies marker lists by subclassing ``ParameterClassPolicy``,
which implements the duck-typed contract the parameter plan consumes (``compute_dtype``,
``is_comm_precision_critical``, ``parameter_class``, ``optimizer_kind``).

Two independent axes:

    compute_fp32_markers  -- matching parameters compute in fp32, the rest in bf16.
    comm_critical_markers -- matching parameters keep fp32 collectives (grad reduce
                             and param publish); the rest may be downcast, with the
                             dtype chosen by the global knob in the parameter plan.

``force_comm_critical_below_ndim`` keeps 1-D tensors (norms, biases) in fp32
collectives: they are precision sensitive and too small to matter for bandwidth.
The axes stay separate so combinations like fp32 compute with downcast comm (an
action-expert weight) remain expressible.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
import logging
from types import SimpleNamespace

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16,
          "fp16": torch.float16, "fp8_e4m3_delta": torch.uint8}
# Public precision enums, validated by the parameter plan, collective plan and resolver.
GRAD_REDUCE_MODES = ("fp32", "bf16", "mixed", "compute")
# Each param_sync_precision decomposes into (master publish dtype, bf16 error
# feedback variant, delta quantization). Only these named combinations are allowed,
# so conflicting settings (bf16 and fp8 at once) cannot be expressed.
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
    """One parameter's name plus its tensor, as handed to a policy."""

    name: str
    parameter: nn.Parameter


class WirePrecisionResolver:
    """Resolve one parameter's wire dtypes from the run's precision plan.

    Both knobs must be set explicitly. Publish mode 'fp32' means compute precision
    for non-critical replicas, not unconditional fp32 transmission.
    """    """Resolve one parameter's wire dtypes from the run's precision plan.

    Both knobs must be set explicitly. Publish mode 'fp32' means compute precision
    for non-critical replicas, not unconditional fp32 transmission.
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
        """Dtype used for this record's gradient collective."""
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
        """Dtype used for this record's parameter-publish collective."""
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
        """True when ``name`` matches one of the configured FP8 markers."""
        return any(token in name for token in self.param_sync_fp8_include)


class CapabilityPolicyAdapter:
    """Combine a model's marker policy with the run's precision plan.

    The policy says what a parameter *is*; the plan turns that into wire dtypes.
    """

    def __init__(self, policy, runtime_config):
        self.markers = policy
        self.wire = WirePrecisionResolver(**runtime_config)

    def classify(self, metadata):
        """Return the capability of one parameter under the run's precision plan."""
        name, parameter = metadata.name, metadata.parameter
        # Frozen tensors are neither cast nor given a master by the parameter plan, so
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
        """Warn about configured markers that matched no parameter name."""
        if hasattr(self.markers, "validate_markers"):
            for group, marker in self.markers.validate_markers(c.name for c in capabilities):
                logging.getLogger(__name__).warning(
                    "%s marker %r matched no parameter name", group, marker
                )


def build_parameter_policy(runtime_config, *, policy):
    """Wrap a model's marker policy in the capability-resolving adapter.

    ``runtime_config`` is the resolved collective precision plan. This module never
    imports model implementations.
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
# (fp32_compute, comm_critical); shown in the plan's precision summary.
_DEFAULT_LABELS = {
    (True, False): "fp32_compute_downcast_comm",
    (True, True): "fp32_compute_fp32_comm",
    (False, True): "bf16_compute_fp32_comm",
    (False, False): "bf16_compute_downcast_comm",
}


class ParameterClassPolicy:
    """Declare each parameter's precision class from per-axis name markers.

    A *marker* is a parameter-name substring: a parameter matches a list when any of
    its markers occurs in the name. ``compute_fp32_markers`` picks fp32 vs bf16
    compute, ``comm_critical_markers`` forces fp32 collectives, ``adamw_markers``
    keeps a tensor off the Muon route, and the (compute, comm) pair names the class
    reported by ``parameter_class``.
    """

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
        """Record the per-model marker lists that drive precision and routing."""

        def markers(values):
            """Validate a marker sequence and drop duplicates, keeping order."""
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
        """Return ``(list_name, marker)`` pairs that matched no parameter name.

        A miss usually means a renamed submodule that silently fell into the default
        class (bf16 compute / downcast comm / Muon).
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


class DefaultParameterPolicy:
    """Marker-free policy for models that declare no precision classes.

    Every parameter keeps the dtype it already has, 1-D tensors (norms, biases)
    keep fp32 collectives, and everything routes to the registry optimizer. A
    model that needs fp32 compute, fp32-critical collectives or Muon supplies a
    ``ParameterClassPolicy`` instead; ``ParameterClassPolicy()`` is not a
    neutral default because it casts to bf16 and routes matrices to Muon.
    """

    def __init__(self, force_comm_critical_below_ndim: int = 2):
        """Record the ndim below which collectives stay fp32."""
        if force_comm_critical_below_ndim < 0:
            raise ValueError("force_comm_critical_below_ndim must be non-negative")
        self._force_comm_critical_below_ndim = int(force_comm_critical_below_ndim)

    def compute_dtype(self, name: str, parameter: nn.Parameter) -> torch.dtype:
        """Keep the parameter's current dtype, so ``--dtype`` stays in charge."""
        return parameter.dtype

    def is_comm_precision_critical(self, name: str, parameter: nn.Parameter) -> bool:
        """True for tensors too small or too sensitive to downcast on the wire."""
        return parameter.ndim < self._force_comm_critical_below_ndim

    def parameter_class(self, name: str, parameter: nn.Parameter) -> str:
        """Return the single class label shown in the precision summary."""
        return "default"

    def optimizer_kind(self, name: str, parameter: nn.Parameter) -> str:
        """Route every parameter to the registry optimizer; Muon is opt-in."""
        return "adamw"


__all__ = [
    "CapabilityPolicyAdapter",
    "DefaultParameterPolicy",
    "GRAD_REDUCE_MODES",
    "PARAM_SYNC_DECOMPOSITION",
    "PARAM_SYNC_PRECISIONS",
    "ParameterCapability",
    "ParameterClassPolicy",
    "ParameterMetadata",
    "WirePrecisionResolver",
    "build_parameter_policy",
    "dtype_label",
    "resolve_parameter_capabilities",
]
