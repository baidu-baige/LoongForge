# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.6 optimization factory.

Extends the vanilla FinetuneTrainer behavior with an optional per-microbatch
CUDA graph runner. When the graph runner is disabled, returns a plain
TorchOptimization() (identical to FinetuneTrainer).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)


def build_optimization(training_args):
    """Return a TorchOptimization for GR00T-N1.6."""
    from loongforge.engines.torch.train_step import TorchOptimization, LossHealth

    # Lazy import: graph runner itself pulls torch, CUDA, model-specific types.
    from loongforge.engines.torch.optimizations.groot_n1_6.per_microbatch_cuda_graph import (
        GrootN1d6PerMicrobatchCudaGraphRunner,
    )

    if not GrootN1d6PerMicrobatchCudaGraphRunner.is_enabled(training_args):
        return TorchOptimization()

    state = SimpleNamespace(runner=None)

    def wrap_model(model, args, ctx):
        import torch
        import torch.distributed as dist
        from loongforge.engines.torch.distributed.parallel import wrap_model as _wrap
        from loongforge.engines.torch.initialize import resolve_dtype

        if (
            ctx.is_distributed
            and ctx.world_size > 1
            and args.cuda_graph_ddp_sync_in_graph
        ):
            model = _wrap(model, args, ctx)
        else:
            model = model.to(dtype=resolve_dtype(args.dtype), device=ctx.device)

        if ctx.is_distributed and ctx.world_size > 1 and not hasattr(model, "module"):
            with torch.no_grad():
                for tensor in list(model.parameters()) + list(model.buffers()):
                    if tensor is not None and tensor.device.type == "cuda":
                        dist.broadcast(tensor, src=0)

        state.runner = GrootN1d6PerMicrobatchCudaGraphRunner(model, args, ctx)
        logger.info(
            "Using model-managed train step runner: %s",
            state.runner.__class__.__name__,
        )
        return model

    def forward_backward(step_ctx):
        import torch
        with step_ctx.timers("forward-backward"):
            state.runner.zero_grad(step_ctx)
            output, _loss_val = state.runner.step(step_ctx)
        return {
            key: (value.detach().item() if isinstance(value, torch.Tensor) else float(value))
            for key, value in output.items()
        }, LossHealth()

    return TorchOptimization(
        wrap_model=wrap_model,
        forward_backward=forward_backward,
        clean_nan_gradients=lambda _m, _o: None,
    )
