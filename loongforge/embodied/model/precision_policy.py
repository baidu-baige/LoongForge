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
        self._compute_fp32_markers = tuple(compute_fp32_markers)
        self._comm_critical_markers = tuple(comm_critical_markers)
        self._labels = dict(labels) if labels is not None else dict(_DEFAULT_LABELS)
        self._force_comm_critical_below_ndim = int(force_comm_critical_below_ndim)
        self._adamw_markers = tuple(adamw_markers)
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
                if not any(marker in name for name in names)
            )
        return misses


__all__ = ["MarkerParameterPolicy"]
