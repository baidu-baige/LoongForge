# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Muon plus AdamW fallback, exposed as one optimizer to the training loop.

Shared by both ``--optimizer muon`` paths: ``build_optimizer`` over the model
parameters, and the tensor_wise trainers over their fp32 masters.
"""

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from loongforge.embodied.optimizer.muon import DistributedMuon

# Name substrings of 2-D weights that stay on AdamW under ``--optimizer muon``:
# the token embedding and the output head, which Muon's recipe leaves out.
MUON_ADAMW_MARKERS = ("embed", "lm_head")

# Ranks routed to Muon unless a model opts more in. 3-D is left out: a generic
# 3-D tensor may be a Conv1d weight rather than an expert stack.
DEFAULT_MUON_NDIMS = (2,)


def muon_route(
    name: str,
    parameter: torch.Tensor,
    ndims: Sequence[int] = DEFAULT_MUON_NDIMS,
    adamw_markers: Sequence[str] = MUON_ADAMW_MARKERS,
) -> str:
    """``"muon"`` for a tensor of one of ``ndims`` whose name has no marker, else ``"adamw"``."""
    if parameter.ndim not in ndims:
        return "adamw"
    lowered = name.lower()
    if any(marker in lowered for marker in adamw_markers):
        return "adamw"
    return "muon"


class CombinedOptimizer(Optimizer):
    """Drive several inner optimizers as if they were a single one."""

    def __init__(self, optimizers: Sequence[Optimizer]):
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one inner optimizer.")
        self.optimizers: List[Optimizer] = list(optimizers)
        # Read by LR schedulers, e.g. ``min_lr / optimizer.defaults["lr"]``.
        self.defaults = dict(self.optimizers[0].defaults)

    @property
    def param_groups(self):
        """All inner param groups, in inner-optimizer order."""
        groups: List[Dict[str, Any]] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):
        """Ignore reassignment; inner optimizers own their group dicts."""
        # LR schedulers mutate the shared param-group dicts; reassignment is a no-op.
        pass

    @property
    def state(self):
        """All inner optimizer states, merged by parameter."""
        merged: Dict[Any, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def register_step_pre_hook(self, hook):
        """Register ``hook`` on the first inner optimizer."""
        return self.optimizers[0].register_step_pre_hook(hook)

    def step(self, closure=None):
        """Step every inner optimizer once."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero the gradients of every inner optimizer's parameters."""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """One standard ``{"state", "param_groups"}`` dict, so torch DCP accepts it.

        Each inner optimizer numbers its parameters from 0; they are offset here
        so the indices stay unique across optimizers.
        """
        state, param_groups, offset = {}, [], 0
        for opt in self.optimizers:
            inner = opt.state_dict()
            state.update({offset + index: value for index, value in inner["state"].items()})
            for group in inner["param_groups"]:
                param_groups.append({**group, "params": [offset + i for i in group["params"]]})
            offset += sum(len(group["params"]) for group in opt.param_groups)
        return {"state": state, "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        """Split ``state_dict`` back by group position.

        The keys are not assumed to be ints: torch DCP hands them back as
        parameter names. Keeps torch's ``state_dict`` argument name, which DCP
        passes by keyword.
        """
        expected = sum(len(opt.param_groups) for opt in self.optimizers)
        if len(state_dict["param_groups"]) != expected:
            raise ValueError(
                f"CombinedOptimizer expects {expected} param groups, "
                f"got {len(state_dict['param_groups'])}."
            )
        start = 0
        for opt in self.optimizers:
            groups = state_dict["param_groups"][start:start + len(opt.param_groups)]
            start += len(opt.param_groups)
            keys = {key for group in groups for key in group["params"]}
            opt.load_state_dict({
                "state": {k: v for k, v in state_dict["state"].items() if k in keys},
                "param_groups": groups,
            })


def _split_param_groups_by_scaled_lr(
    named_parameters: Sequence[Tuple[str, torch.Tensor]],
    base_lr: float,
    lr_scale_for=None,
) -> List[Dict[str, Any]]:
    """Bucket ``(name, parameter)`` pairs by their LR, scaled per parameter name.

    ``lr_scale_for(name)`` returns the multiplier for one parameter and defaults
    to no scaling. The model supplies it, so the rule for which parameters run at
    a different LR stays with the model rather than in this generic optimizer.
    """
    lr_to_params: Dict[float, List[torch.Tensor]] = {base_lr: []}
    for name, parameter in named_parameters:
        scale = 1.0 if lr_scale_for is None else float(lr_scale_for(name))
        lr_to_params.setdefault(base_lr * scale, []).append(parameter)
    return [{"params": ps, "lr": lr} for lr, ps in lr_to_params.items() if ps]


def build_muon_optimizer(
    muon_parameters: Sequence[Tuple[str, torch.Tensor]],
    *,
    lr: float,
    adamw_parameters: Sequence[Tuple[str, torch.Tensor]] = (),
    weight_decay: float = 0.0,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    adjust_lr_fn: str = "match_rms_adamw",
    lr_scale_for=None,
    adamw_betas: Tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
) -> "torch.optim.Optimizer":
    """Build DistributedMuon for matrix-like weights plus AdamW fallback groups.

    The caller passes the two ``(name, parameter)`` lists already split, so this
    builder never routes parameters itself. ``lr_scale_for`` optionally scales
    the LR of individual parameters (e.g. MoE experts), and is supplied by the
    model.
    """
    muon_groups = _split_param_groups_by_scaled_lr(muon_parameters, lr, lr_scale_for)
    adamw_groups = _split_param_groups_by_scaled_lr(
        adamw_parameters, lr, lr_scale_for
    )

    if not muon_groups:
        raise RuntimeError(
            "build_muon_optimizer: no parameters were routed to Muon. "
            "Use --optimizer AdamW instead."
        )

    muon_opt = DistributedMuon(
        muon_groups,
        lr=lr,
        weight_decay=weight_decay,
        momentum=float(momentum),
        nesterov=bool(nesterov),
        ns_steps=int(ns_steps),
        adjust_lr_fn=adjust_lr_fn,
    )

    inner_opts: List[Optimizer] = [muon_opt]
    if adamw_groups:
        adamw_opt = AdamW(
            adamw_groups,
            lr=lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
            fused=False,
            # foreach=True matches the upstream baseline, whose AdamW path ran
            # ``not is_torch_npu_available()`` -> True (no torch_npu anywhere).
            # The dead NPU gate is dropped; the value it produced is kept.
            foreach=True,
        )
        inner_opts.append(adamw_opt)

    return CombinedOptimizer(inner_opts)


def build_muon_optimizer_from_args(
    muon_parameters: Sequence[Tuple[str, torch.Tensor]],
    adamw_parameters: Sequence[Tuple[str, torch.Tensor]],
    training_args,
    *,
    lr_scale_for=None,
) -> "torch.optim.Optimizer":
    """``build_muon_optimizer`` with every hyperparameter read from ``training_args``."""
    return build_muon_optimizer(
        muon_parameters,
        lr=training_args.lr_base,
        adamw_parameters=adamw_parameters,
        weight_decay=training_args.weight_decay,
        momentum=training_args.muon_momentum,
        nesterov=training_args.muon_nesterov,
        ns_steps=training_args.muon_ns_steps,
        adjust_lr_fn=training_args.muon_adjust_lr_fn,
        lr_scale_for=lr_scale_for,
    )
