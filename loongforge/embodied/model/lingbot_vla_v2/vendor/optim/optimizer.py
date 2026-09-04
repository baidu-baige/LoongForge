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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ..utils.import_utils import is_torch_npu_available
from .muon import DistributedMuon, split_muon_adamw_params


# https://github.com/meta-llama/llama-recipes/blob/v0.0.4/src/llama_recipes/policies/anyprecision_optimizer.py
class AnyPrecisionAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        use_kahan_summation=True,
        momentum_dtype=torch.bfloat16,
        variance_dtype=torch.bfloat16,
        compensation_buffer_dtype=torch.bfloat16,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_kahan_summation": use_kahan_summation,
            "momentum_dtype": momentum_dtype,
            "variance_dtype": variance_dtype,
            "compensation_buffer_dtype": compensation_buffer_dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients.")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)

                # Main processing
                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                if weight_decay:  # weight decay, AdamW style
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # update momentum
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # update uncentered variance

                bias_correction1 = 1 - beta1**step  # adjust using bias1
                step_size = lr / bias_correction1

                denom_correction = (1 - beta2**step) ** 0.5  # adjust using bias2 and avoids math import
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)

                if use_kahan_summation:  # lr update to compensation
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))
                else:  # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)


def build_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
    post_training=False,
) -> "torch.optim.Optimizer":
    if param_groups is None:
        align_parameters = [
            name for name, _ in model.named_parameters() if "depth" in name
        ]

        if len(align_parameters) > 0:
            lr_gain = 10.0 if not post_training else 1.0
            param_groups = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (p.requires_grad and n not in align_parameters)
                    ],
                    "lr": lr,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (p.requires_grad and n in align_parameters)
                    ],
                    "lr": lr * lr_gain,
                }
            ]
        else:
            param_groups = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_type == "adamw":
        foreach = False if is_torch_npu_available() else (not fused)
        fused = False if is_torch_npu_available() else fused
        optim = AdamW(param_groups, lr, betas, eps, weight_decay, fused=fused, foreach=foreach)
    elif optimizer_type == "anyprecision_adamw":
        optim = AnyPrecisionAdamW(param_groups, lr, betas, eps, weight_decay)
    elif optimizer_type == "muon":
        raise ValueError(
            "optimizer_type='muon' must go through build_muon_optimizer(model, args_train, ...) "
            "so that 1D params can be routed to AdamW."
        )
    else:
        raise ValueError("Only adamw and anyprecision_adamw are supported as optimizers.")

    return optim


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
    parameter_policy=None,
) -> "torch.optim.Optimizer":
    """Build DistributedMuon for matrix-like weights plus AdamW fallback groups."""
    muon_params, adamw_params, muon_names, adamw_names = split_muon_adamw_params(
        model,
        no_decay_modules=None,
        no_decay_params=None,
        extra_adamw_name_patterns=getattr(args_train, "muon_exclude_name_patterns", None) or None,
        parameter_policy=parameter_policy,
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
        foreach = not is_torch_npu_available()
        adamw_opt = AdamW(
            adamw_groups,
            lr=lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
            fused=False,
            foreach=foreach,
        )
        inner_opts.append(adamw_opt)

    return CombinedOptimizer(inner_opts)
