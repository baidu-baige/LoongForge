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

"""Muon plus AdamW fallback, exposed as one optimizer to the training loop."""

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from .muon import DistributedMuon, split_muon_adamw_params


class CombinedOptimizer(Optimizer):
    """Drive several inner optimizers as if they were a single one."""

    def __init__(self, optimizers: Sequence[Optimizer]):
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one inner optimizer.")
        self.optimizers: List[Optimizer] = list(optimizers)
        self.defaults = {}

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
        """Return the inner optimizers' state dicts, in order."""
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        """Restore the inner optimizers' state dicts, in order."""
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


def _split_param_groups_by_scaled_lr(
    params_and_names: Sequence[Tuple[torch.Tensor, str]],
    base_lr: float,
    lr_scale_for=None,
) -> List[Dict[str, Any]]:
    """Bucket (param, name) pairs by their LR, scaled per parameter name.

    ``lr_scale_for(name)`` returns the multiplier for one parameter and defaults
    to no scaling. The model supplies it, so the rule for which parameters run at
    a different LR stays with the model rather than in this generic optimizer.
    """
    lr_to_params: Dict[float, List[torch.Tensor]] = {base_lr: []}
    for p, name in params_and_names:
        scale = 1.0 if lr_scale_for is None else float(lr_scale_for(name))
        lr_to_params.setdefault(base_lr * scale, []).append(p)
    return [{"params": ps, "lr": lr} for lr, ps in lr_to_params.items() if ps]


def build_muon_optimizer(
    model: "nn.Module",
    *,
    lr: float,
    parameter_policy,
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

    ``parameter_policy`` is required: it is the only thing that decides which
    parameters Muon owns (``muon_exclude_name_patterns`` is already folded into
    its AdamW markers). ``lr_scale_for`` optionally scales the LR of individual
    parameters (e.g. MoE experts), and is supplied by the model.
    """
    muon_params, adamw_params, muon_names, adamw_names = split_muon_adamw_params(
        model,
        parameter_policy,
    )

    muon_groups = _split_param_groups_by_scaled_lr(
        list(zip(muon_params, muon_names)), lr, lr_scale_for
    )
    adamw_groups = _split_param_groups_by_scaled_lr(
        list(zip(adamw_params, adamw_names)), lr, lr_scale_for
    )

    if not muon_groups:
        raise RuntimeError(
            "build_muon_optimizer: no Muon-eligible (2D/3D) parameters were found. "
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
