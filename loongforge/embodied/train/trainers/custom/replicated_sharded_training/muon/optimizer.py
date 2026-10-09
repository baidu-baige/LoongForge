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


import re
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
        self._step_pre_hooks: List[Any] = []

    @property
    def param_groups(self):
        groups: List[Dict[str, Any]] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):
        # LR schedulers mutate the shared param-group dicts; reassignment is a no-op.
        pass

    @property
    def state(self):
        merged: Dict[Any, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def register_step_pre_hook(self, hook):
        return self.optimizers[0].register_step_pre_hook(hook)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


def _split_param_groups_by_scaled_lr(
    params_and_names: Sequence[Tuple[torch.Tensor, str]],
    base_lr: float,
    layer_to_scale: Dict[int, float],
    layer_re: "re.Pattern[str]",
) -> List[Dict[str, Any]]:
    """Bucket (param, name) pairs by their possibly MoE-scaled LR."""
    lr_to_params: Dict[float, List[torch.Tensor]] = {base_lr: []}
    for p, name in params_and_names:
        m = layer_re.search(name)
        lr_for_param = base_lr
        if m is not None:
            layer_idx = int(m.group(1))
            scale = layer_to_scale.get(layer_idx)
            if scale is not None:
                lr_for_param = base_lr * scale
        lr_to_params.setdefault(lr_for_param, []).append(p)
    return [{"params": ps, "lr": lr} for lr, ps in lr_to_params.items() if ps]


def build_muon_optimizer(
    model: "nn.Module",
    args_train,
    lr: float,
    weight_decay: float = 0.0,
    adamw_betas: Tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
    *,
    parameter_policy,
) -> "torch.optim.Optimizer":
    """Build DistributedMuon for matrix-like weights plus AdamW fallback groups.

    ``parameter_policy`` is required: it is the only thing that decides which
    parameters Muon owns (``muon_exclude_name_patterns`` is already folded into
    its AdamW markers).
    """
    muon_params, adamw_params, muon_names, adamw_names = split_muon_adamw_params(
        model,
        parameter_policy,
    )

    use_expert_lr = bool(getattr(args_train, "use_moe", False)) and bool(
        getattr(args_train, "use_moe_expert_lr", False)
    )
    layer_to_scale: Dict[int, float] = {}
    layer_re = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.")
    if use_expert_lr:
        token_moe_layers = set(getattr(args_train, "token_moe_layers", None) or [])
        if token_moe_layers:
            token_scale = (args_train.token_num_experts / args_train.token_top_k) ** 0.5
            for idx in token_moe_layers:
                layer_to_scale[idx] = token_scale

    muon_groups = _split_param_groups_by_scaled_lr(
        list(zip(muon_params, muon_names)), lr, layer_to_scale, layer_re
    )
    adamw_groups = _split_param_groups_by_scaled_lr(
        list(zip(adamw_params, adamw_names)), lr, layer_to_scale, layer_re
    )

    if not muon_groups:
        raise RuntimeError(
            "build_muon_optimizer: no Muon-eligible (2D/3D) parameters were found. "
            "Use build_optimizer(optimizer_type='adamw') instead."
        )

    muon_opt = DistributedMuon(
        muon_groups,
        lr=lr,
        weight_decay=weight_decay,
        momentum=float(getattr(args_train, "muon_momentum", 0.95)),
        nesterov=bool(getattr(args_train, "muon_nesterov", True)),
        ns_steps=int(getattr(args_train, "muon_ns_steps", 5)),
        adjust_lr_fn=getattr(args_train, "muon_adjust_lr_fn", "match_rms_adamw"),
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
