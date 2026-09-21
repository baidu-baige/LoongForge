# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Config-driven parameter precision policy shared by optimizer-state-shard model integrations.

The optimizer-state-shard registry is model-agnostic: it consumes a parameter policy through a
small duck-typed contract (``compute_dtype``, ``is_comm_precision_critical``,
``classify``, ``optimizer_kind``). This base implements that contract from two
declarative marker lists so a new model only has to describe its parameters, not
add branching in Python.

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
PARAM_SYNC_DECOMPOSITION = {
    "fp32": ("compute", "none", "none"),
    "bf16": ("bf16", "none", "none"),
    "bf16_ef": ("bf16", "error_feedback", "none"),
    "bf16_ef_delta": ("bf16", "error_feedback_delta", "none"),
    "fp8_e4m3_delta": ("compute", "none", "fp8_e4m3_delta"),
}


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
    """Single implementation of the historical registry wire-dtype algorithm.

    In particular, publish mode 'fp32' means compute precision for non-critical
    replicas, not unconditional FP32 transmission.
    """

    def __init__(self, config=None):
        config = config or {}
        get = lambda key, default: (
            config.get(key, default) if isinstance(config, dict)
            else getattr(config, key, default)
        )
        # Values equal to defaults (including False and an empty whitelist) may
        # still be explicit constraints. None alone means "not supplied".
        self.explicit_fields = frozenset(
            key for key in (
                "grad_reduce_dtype", "param_sync_precision",
                "param_sync_fp8_include", "param_sync_bf16_with_fp8",
            ) if get(key, None) is not None
        )
        grad = get("grad_reduce_dtype", None)
        sync = get("param_sync_precision", None)
        self.grad_reduce_mode = "fp32" if grad is None else str(grad).lower()
        precision = "fp32" if sync is None else str(sync).lower()
        self.param_sync_precision = precision
        if self.grad_reduce_mode not in ("fp32", "bf16", "mixed", "compute"):
            raise ValueError(f"invalid grad_reduce_dtype={self.grad_reduce_mode!r}")
        if precision not in PARAM_SYNC_DECOMPOSITION:
            raise ValueError(f"invalid param_sync_precision={precision!r}")
        self.param_sync_mode, _, self.param_sync_quantization = PARAM_SYNC_DECOMPOSITION[precision]
        include = get("param_sync_fp8_include", ()) or ()
        if isinstance(include, str) or any(not isinstance(s, str) or not s for s in include):
            raise ValueError("param_sync_fp8_include must contain non-empty markers")
        self.param_sync_fp8_include = tuple(include)
        shadow = get("param_sync_bf16_with_fp8", None)
        self.param_sync_bf16_with_fp8 = False if shadow is None else shadow

    def _resolve_grad_wire_dtype(self, record):
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
        if record.is_comm_critical:
            return torch.float32
        compute_dtype = record.compute.dtype
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
        if self.param_sync_mode == "bf16":
            return torch.bfloat16
        return torch.float32

    def _fp8_in_scope(self, name):
        return any(token in name for token in self.param_sync_fp8_include)


class LegacyParameterPolicyAdapter:
    """Adapt classify(name, tensor) labels without changing the legacy API."""

    def __init__(self, policy, runtime_config=None, explicit_rules=()):
        self.legacy = policy
        self.wire = WirePrecisionResolver(runtime_config)
        self.explicit_rules = tuple(explicit_rules)
        allowed = {"compute_dtype", "communication_critical", "optimizer_kind"}
        for rule in self.explicit_rules:
            if not isinstance(rule.get("marker"), str) or not rule["marker"]:
                raise ValueError("explicit rule requires a non-empty marker")
            if set(rule) - allowed - {"marker"}:
                raise ValueError(f"unknown explicit policy fields: {rule}")
            if "compute_dtype" in rule and rule["compute_dtype"] not in ("fp32", "bf16", "fp16"):
                raise ValueError(f"invalid explicit compute dtype: {rule}")
            if "optimizer_kind" in rule and rule["optimizer_kind"] not in ("adamw", "muon"):
                raise ValueError(f"invalid explicit optimizer kind: {rule}")
            if "communication_critical" in rule and type(rule["communication_critical"]) is not bool:
                raise ValueError(f"invalid explicit communication flag: {rule}")

    def classify(self, metadata):
        name, parameter = metadata.name, metadata.parameter
        dtype = self.legacy.compute_dtype(name, parameter)
        dtype_name = next((k for k, v in DTYPES.items() if v == dtype), None)
        values = {
            "compute_dtype": dtype_name,
            "communication_critical": self.legacy.is_comm_precision_critical(name, parameter),
            "optimizer_kind": self.legacy.optimizer_kind(name, parameter),
        }
        explicit = {}
        for rule in self.explicit_rules:
            if rule["marker"] in name:
                for key, value in rule.items():
                    if key == "marker":
                        continue
                    if key in explicit and explicit[key] != value:
                        raise ValueError(f"conflicting {key} rules for {name}")
                    explicit[key] = value
        values.update(explicit)
        # Frozen tensors are not cast or assigned masters by the registry.
        if not parameter.requires_grad:
            values["compute_dtype"] = next(
                (k for k, v in DTYPES.items() if v == parameter.dtype), None
            )
        if values["compute_dtype"] not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"invalid compute dtype for {name}: {values['compute_dtype']}")
        record = SimpleNamespace(
            name=name, compute=SimpleNamespace(dtype=DTYPES[values["compute_dtype"]]),
            is_comm_critical=values["communication_critical"],
        )
        label = lambda dtype: next(k for k, v in DTYPES.items() if v == dtype)
        return ParameterCapability(
            name=name, shape=tuple(parameter.shape), numel=parameter.numel(),
            grad_reduce_dtype=label(self.wire._resolve_grad_wire_dtype(record)),
            parameter_sync_dtype=label(self.wire._resolve_param_wire_dtype(record)),
            requires_grad=parameter.requires_grad,
            parameter_class=self.legacy.classify(name, parameter), **values,
        )

    def validate(self, capabilities):
        if hasattr(self.legacy, "validate_markers"):
            for group, marker in self.legacy.validate_markers(c.name for c in capabilities):
                logging.getLogger(__name__).warning(
                    "%s marker %r matched no parameter name", group, marker
                )


def build_parameter_policy(model_config=None, runtime_config=None, *, legacy_policy=None):
    """Build from model-neutral rules, or adapt an existing model policy.

    Model config accepts MarkerParameterPolicy constructor fields and
    ``explicit_rules`` (substring rules overriding defaults). Models provide
    their own legacy policy; this module never imports model implementations.
    """
    config = dict(model_config or {})
    rules = config.pop("explicit_rules", ())
    if legacy_policy is not None and config:
        raise ValueError("marker config and legacy_policy cannot both be supplied")
    return LegacyParameterPolicyAdapter(
        legacy_policy if legacy_policy is not None else MarkerParameterPolicy(**config),
        runtime_config, rules,
    )


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
    """Interpret two per-axis marker lists as the optimizer-state-shard policy contract."""

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

    def classify(self, name: str, parameter: nn.Parameter) -> str:
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
    "MarkerParameterPolicy", "ParameterCapability", "ParameterMetadata",
    "ParameterPolicy", "LegacyParameterPolicyAdapter", "build_parameter_policy",
    "resolve_parameter_capabilities", "WirePrecisionResolver",
]
