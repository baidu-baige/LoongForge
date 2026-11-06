# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 optimizer path: capturable FusedAdam, port of
``cosmos_framework.utils.generator.fused_adam.FusedAdam``.

Dispatches to TransformerEngine's ``multi_tensor_adam_capturable`` kernel with the LR
and per-group step held as device tensors, exactly as cosmos-framework does for the
Cosmos3 recipes (``optimizer_type="FusedAdam"``). FP32 parameters only (they are the
FSDP fp32 masters); master-weight and fp16/bf16 branches are omitted.

Importing this module registers it in the framework optimizer registry as
``CapturableFusedAdamW`` (``--optimizer CapturableFusedAdamW``); ``modeling_cosmos3``
imports the ``cosmos3`` package, so the name is registered once the Cosmos3 model is loaded.
"""

import torch
from torch.distributed.tensor import DTensor

from loongforge.embodied.optimizer.optimizer import OPTIMIZER_REGISTRY


def _local(tensor):
    if isinstance(tensor, DTensor):
        local = tensor.to_local()
        return local.wait() if hasattr(local, "wait") else local
    return tensor


class CapturableFusedAdam(torch.optim.Optimizer):
    """AdamW (decoupled weight decay) via TE ``multi_tensor_adam_capturable``."""

    def __init__(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-8,
                 adam_w_mode=True, weight_decay=0.0):
        import transformer_engine_torch as tex

        lr = torch.tensor(lr, dtype=torch.float32)
        defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.capturable = True
        # Hold each group's LR as a device fp32 tensor from the start (as cosmos does): the LR
        # scheduler captures these as its base LRs, so schedule factors are applied in fp32.
        for group in self.param_groups:
            if not group["params"]:
                continue
            device = group["params"][0].device
            if not isinstance(group["lr"], torch.Tensor):
                group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)
            group["lr"] = group["lr"].to(device=device)
        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        self._kernel = tex.multi_tensor_adam_capturable

    def add_param_group(self, param_group):
        for p in param_group["params"]:
            if p.dtype != torch.float32:
                raise ValueError("CapturableFusedAdam requires fp32 parameters (use fp32 FSDP masters).")
        super().add_param_group(param_group)

    @torch.no_grad()
    def step(self, closure=None):
        from transformer_engine.pytorch.optimizers import multi_tensor_applier

        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if not group["params"]:
                continue
            device = group["params"][0].device
            bias_correction = 1 if group.get("bias_correction") else 0
            beta1, beta2 = group["betas"]
            if "step" in group:
                group["step"] = (
                    group["step"].to(device=device)
                    if isinstance(group["step"], torch.Tensor)
                    else torch.tensor(group["step"], dtype=torch.int32, device=device)
                )
                group["step"] += (self._dummy_overflow_buf != 1).to(torch.int)
            else:
                group["step"] = torch.tensor([1], dtype=torch.int, device=device)
            group["lr"] = (
                group["lr"].to(device=device)
                if isinstance(group["lr"], torch.Tensor)
                else torch.tensor(group["lr"], dtype=torch.float32, device=device)
            )

            g_32, p_32, m_32, v_32 = [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p).float()
                    state["exp_avg_sq"] = torch.zeros_like(p).float()
                if _local(p).numel() == 0:
                    continue
                g_32.append(_local(p.grad))
                p_32.append(_local(p))
                m_32.append(_local(state["exp_avg"]))
                v_32.append(_local(state["exp_avg_sq"]))

            self._dummy_overflow_buf.copy_(torch.zeros((1,), device=device))
            inv_scale = torch.ones((1,), device=device, dtype=torch.float32)
            if g_32:
                multi_tensor_applier(
                    self._kernel, self._dummy_overflow_buf, [g_32, p_32, m_32, v_32],
                    group["lr"], beta1, beta2, group["eps"], group["step"],
                    self.adam_w_mode, bias_correction, group["weight_decay"], inv_scale,
                )
        return loss


# cosmos-framework FusedAdam (TE multi_tensor_adam_capturable), fp32 params only.
OPTIMIZER_REGISTRY["CapturableFusedAdamW"] = CapturableFusedAdam
