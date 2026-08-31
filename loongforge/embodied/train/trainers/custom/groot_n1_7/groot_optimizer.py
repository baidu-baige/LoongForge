# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.7 precision-compatible optimizer path.

The precision-compatible update is owned by the GR00T trainer because it is
coupled to the full-iteration graph's static gradient buffers and scale.
When the graph is disabled, the trainer uses the framework optimizer registry
instead; the optional GR00T AOT package is only needed by the graph-owned
capturable path.
"""

from __future__ import annotations

import math

import torch

from loongforge.embodied.optimizer.lr_scheduler import build_param_groups
from loongforge.embodied.optimizer.optimizer import build_optimizer as build_generic_optimizer

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as _TEFusedAdam
except ImportError:  # pragma: no cover
    _TEFusedAdam = None

if _TEFusedAdam is not None:
    class GrootCapturableAdamW(_TEFusedAdam):
        """CUDA-graph capturable AdamW built on Transformer Engine's FusedAdam."""

        def __init__(self, params, **kwargs):
            self._alignment_max_steps = int(kwargs.pop("alignment_max_steps", 0))
            super().__init__(params, **kwargs)
            self._alignment_bias_tables = {}
            self._loongforge_grad_scale = None

        def set_grad_scale(self, grad_scale: torch.Tensor) -> None:
            """Bind the device-side gradient scale used by the fused update."""
            if not isinstance(grad_scale, torch.Tensor) or grad_scale.numel() != 1 \
                    or grad_scale.dtype != torch.float32 or not grad_scale.is_cuda:
                raise RuntimeError("GR00T fused grad clip requires a CUDA float32 scalar.")
            self._loongforge_grad_scale = grad_scale

        def _bias_tables(self, device: torch.device):
            cached = self._alignment_bias_tables.get(device)
            if cached is not None:
                return cached
            if self._alignment_max_steps <= 0:
                raise RuntimeError("GR00T capturable AdamW requires alignment_max_steps.")
            beta1, beta2 = self.param_groups[0]["betas"]
            correction1 = torch.tensor(
                [0.0] + [1.0 - beta1 ** step for step in range(1, self._alignment_max_steps + 1)],
                dtype=torch.float64, device=device,
            )
            correction2 = torch.tensor(
                [0.0] + [math.sqrt(1.0 - beta2 ** step) for step in range(1, self._alignment_max_steps + 1)],
                dtype=torch.float64, device=device,
            )
            self._alignment_bias_tables[device] = (correction1, correction2)
            return correction1, correction2

        @torch.no_grad()
        def step(self, closure=None, grad_scaler=None):
            """Run one fused AdamW update over all parameter groups."""
            if closure is not None or grad_scaler is not None:
                raise NotImplementedError("GR00T capturable AdamW does not support closures/scalers.")
            result = None
            for group in self.param_groups:
                params, grads, exp_avgs, exp_avg_sqs = [], [], [], []
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if gradient.is_sparse:
                        raise RuntimeError("GR00T capturable AdamW does not support sparse gradients.")
                    state = self.state[parameter]
                    if not state:
                        self.initialize_state(parameter, False)
                    params.append(parameter)
                    grads.append(gradient)
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                if not params:
                    continue
                beta1, beta2 = group["betas"]
                if self.capturable:
                    from loongforge.embodied.train.trainers.custom.groot_n1_7.groot_fused_adamw import (
                        capturable_grad_scaled_step,
                        capturable_step,
                    )

                    group.setdefault("step", torch.zeros((), dtype=torch.int64, device=params[0].device))
                    group["step"].add_(1)
                    correction1, correction2 = self._bias_tables(params[0].device)
                    if self._loongforge_grad_scale is None:
                        # Eager GR00T clips gradients in-place before calling
                        # the same capturable AdamW kernel used by Graph-on.
                        capturable_step(
                            params, grads, exp_avgs, exp_avg_sqs,
                            lr=group["lr"], step=group["step"],
                            bias_correction1=correction1, bias_correction2_sqrt=correction2,
                            beta2=beta2, first_moment_weight=1.0 - beta1,
                            second_moment_weight=1.0 - beta2, eps=group["eps"],
                            weight_decay=group["weight_decay"],
                        )
                    else:
                        capturable_grad_scaled_step(
                            params, grads, exp_avgs, exp_avg_sqs,
                            lr=group["lr"], step=group["step"],
                            bias_correction1=correction1, bias_correction2_sqrt=correction2,
                            grad_scale=self._loongforge_grad_scale, beta2=beta2,
                            first_moment_weight=1.0 - beta1, second_moment_weight=1.0 - beta2,
                            eps=group["eps"], weight_decay=group["weight_decay"],
                        )
                else:
                    from loongforge.embodied.train.trainers.custom.groot_n1_7.groot_fused_adamw import (
                        eager_step,
                    )

                    group["step"] = int(group.get("step", 0)) + 1
                    step = group["step"]
                    lr = float(group["lr"])
                    eager_step(
                        params, grads, exp_avgs, exp_avg_sqs,
                        decay_factor=1.0 - lr * group["weight_decay"],
                        beta2=beta2,
                        first_moment_weight=1.0 - beta1,
                        second_moment_weight=1.0 - beta2,
                        eps=group["eps"],
                        bias_correction1=1.0 - beta1 ** step,
                        bias_correction2_sqrt=math.sqrt(1.0 - beta2 ** step),
                        lr=lr,
                    )
            return result
else:  # pragma: no cover
    GrootCapturableAdamW = None


def build_groot_optimizer(model, training_args, *, capturable: bool = True):
    """Build the GR00T optimizer for the full-iteration graph path.

    The precision-compatible optimizer is reserved for the full-iteration
    graph path.  Eager training delegates to the generic registry so it does
    not require the optional GR00T AOT operators.
    """
    if training_args.optimizer != "TEFusedAdamW" or not capturable:
        return build_generic_optimizer(model, training_args)
    if GrootCapturableAdamW is None:
        raise ImportError("TransformerEngine FusedAdam is required for GR00T full-iteration Graph")
    groups = build_param_groups(model, training_args)
    if capturable:
        for group in groups:
            params = group.get("params", [])
            if params:
                group["lr"] = torch.tensor(
                    float(group["lr"]), dtype=torch.float64, device=params[0].device
                )
    optimizer = GrootCapturableAdamW(
        groups,
        lr=training_args.lr_base,
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_eps,
        adam_w_mode=True,
        # The graph path schedules the capturable kernel directly; its static
        # LR buffers preserve the precision required by graph replay.
        capturable=capturable,
        alignment_max_steps=training_args.train_iters + 2,
    )
    optimizer.capturable = capturable
    if capturable:
        # TE stores its capturable default LR as a CUDA float32 scalar.  The
        # Transformers cosine-with-min-LR scheduler divides min_lr by this
        # default and would therefore round every schedule factor to float32,
        # even though the per-group Graph LR buffers above are float64.  The
        # default is scheduler metadata only; optimizer updates read group LR.
        optimizer.defaults["lr"] = float(training_args.lr_base)
    return optimizer
