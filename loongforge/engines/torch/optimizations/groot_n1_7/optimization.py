# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.7 optimization factory.

Extends the vanilla FinetuneTrainer behavior with full-iteration CUDA graph
support and GR00T-specific DDP bucket alignment, fused optimizer, and
post-setup RNG reset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace

logger = logging.getLogger(__name__)


# ── Bucket warmup functions (moved from groot_trainer.py) ──

def _arm_static_graph_bucket_warmup(model) -> None:
    """Record normal DDP's first-backward parameter ready order."""
    import torch
    import torch.nn as nn

    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return

    @dataclass
    class _StaticGraphBucketWarmup:
        parameters: list
        expect_sparse_gradients: list
        ready_order: list = field(default_factory=list)
        recorded_indices: set = field(default_factory=set)
        hook_handles: list = field(default_factory=list)

        def record_ready(self, parameter_index: int, _parameter) -> None:
            if parameter_index in self.recorded_indices:
                return
            self.recorded_indices.add(parameter_index)
            self.ready_order.append(parameter_index)

        def remove_hooks(self) -> None:
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles.clear()

    raw_model = model.module
    ignored_names = set(getattr(raw_model, "_ddp_params_and_buffers_to_ignore", ()))
    entries = [
        (module_name, module, parameter_name, parameter)
        for module_name, module in raw_model.named_modules()
        for parameter_name, parameter in module.named_parameters(recurse=False)
        if parameter.requires_grad
        and f"{module_name}.{parameter_name}" not in ignored_names
    ]
    parameters = []
    sparse = []
    seen = set()
    for _module_name, module, _parameter_name, parameter in entries:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        parameters.append(parameter)
        sparse.append(isinstance(module, (nn.Embedding, nn.EmbeddingBag)) and module.sparse)
    state = _StaticGraphBucketWarmup(parameters, sparse)
    state.hook_handles = [
        parameter.register_post_accumulate_grad_hook(
            partial(state.record_ready, index)
        )
        for index, parameter in enumerate(parameters)
    ]
    model._loong_static_graph_bucket_warmup = state


def _align_static_graph_buckets_after_warmup(model) -> bool:
    """Rebuild static DDP buckets in the observed first-backward order."""
    import torch
    import torch.distributed as dist

    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return False
    state = getattr(model, "_loong_static_graph_bucket_warmup", None)
    if state is None:
        return False
    try:
        ready_order = list(state.ready_order)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            payload = [ready_order if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            ready_order = payload[0]
        if (
            len(ready_order) != len(state.parameters)
            or len(set(ready_order)) != len(state.parameters)
            or min(ready_order, default=-1) != 0
            or max(ready_order, default=-1) != len(state.parameters) - 1
        ):
            raise RuntimeError(
                "Static DDP bucket warmup did not observe every trainable parameter exactly once: "
                f"observed={len(ready_order)} expected={len(state.parameters)}"
            )
        config = model._bucket_config
        limits = (
            list(config.per_bucket_bytes_caps)
            if config.per_bucket_bytes_caps
            else [config.first_bucket_bytes_cap, config.bucket_bytes_cap]
        )
        ordered = [state.parameters[index] for index in ready_order]
        sparse = [state.expect_sparse_gradients[index] for index in ready_order]
        bucket_indices, _ = dist._compute_bucket_assignment_by_size(
            ordered, limits, sparse, ready_order
        )
        from loongforge.engines.torch.optimizations.groot_n1_7.groot_ddp_reducer_bucket_control import (
            initialize_buckets,
        )

        initialize_buckets(model.reducer, bucket_indices)
        model._has_rebuilt_buckets = True
        logger.info("Initialized %d GR00T DDP buckets from first-backward ready order.", len(bucket_indices))
        return True
    finally:
        state.remove_hooks()
        delattr(model, "_loong_static_graph_bucket_warmup")


def _full_iteration_graph_stream_priority(backbone_pipeline: bool) -> int:
    """Return the validated split-Graph train stream priority."""
    import torch
    low_priority, high_priority = torch.cuda.Stream.priority_range()
    default = high_priority if backbone_pipeline else 0
    return max(high_priority, min(low_priority, -3 if backbone_pipeline else default))


def build_optimization(training_args):
    """Return a TorchOptimization for GR00T-N1.7."""
    import torch
    from loongforge.engines.torch.train_step import TorchOptimization
    from loongforge.engines.torch.optimizations.groot_n1_7.full_iteration_cuda_graph import (
        GrootN1d7FullIterationCudaGraphRunner,
    )
    from loongforge.engines.torch.optimizations.groot_n1_7.groot_optimizer import (
        build_groot_optimizer,
    )
    from loongforge.engines.torch.distributed.utils import unwrap_model

    # Determine graph runner type
    graph_mode = False
    if training_args.cuda_graph_impl == "local":
        if training_args.cuda_graph_scope == "full_iteration":
            if not GrootN1d7FullIterationCudaGraphRunner.is_enabled(training_args):
                raise RuntimeError(
                    "GR00T-N1.7 full-iteration CUDA graph was requested but CUDA is unavailable; "
                    "eager fallback is forbidden."
                )
            graph_mode = True
        else:
            raise RuntimeError(
                "GR00T-N1.7 only supports --cuda-graph-scope=full_iteration; "
                f"got {training_args.cuda_graph_scope!r}."
            )

    if not graph_mode:
        # Eager mode: only override optimizer and after_setup
        from loongforge.engines.torch.initialize import set_seed

        def _on_step_end_eager(metrics, completed_steps, model):
            for key, value in list(metrics.items()):
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    metrics[key] = value.detach().cpu().item()
                elif hasattr(value, "item") and value.__class__.__module__.split(".")[0] == "numpy":
                    metrics[key] = value.item()

        return TorchOptimization(
            build_optimizer=lambda model, args, ctx: build_groot_optimizer(
                model, args, capturable=False,
            ),
            after_setup=lambda args: set_seed(args.seed),
            on_step_end=_on_step_end_eager,
        )

    # Graph mode
    state = SimpleNamespace(runner=None, graph_stream=None)

    def wrap_model(model, args, ctx):
        from loongforge.engines.torch.distributed.parallel import wrap_model as _wrap

        default_stream = torch.cuda.current_stream(ctx.device)
        backbone_pipeline = True
        graph_priority = _full_iteration_graph_stream_priority(backbone_pipeline)
        graph_stream = torch.cuda.Stream(device=ctx.device, priority=graph_priority)
        graph_stream.wait_stream(default_stream)
        with torch.cuda.stream(graph_stream):
            model = _wrap(model, args, ctx)
        _arm_static_graph_bucket_warmup(model)
        default_stream.wait_stream(graph_stream)
        torch.cuda.synchronize(ctx.device)
        state.graph_stream = graph_stream
        state.runner = GrootN1d7FullIterationCudaGraphRunner(model, args, ctx, graph_stream)
        logger.info(
            "Using model-managed train step runner: %s (stream_priority=%d)",
            state.runner.__class__.__name__,
            graph_priority,
        )
        return model

    def build_optimizer_fn(model, args, ctx):
        default_stream = torch.cuda.current_stream(ctx.device)
        state.graph_stream.wait_stream(default_stream)
        with torch.cuda.stream(state.graph_stream):
            optimizer = build_groot_optimizer(model, args)
        default_stream.wait_stream(state.graph_stream)
        return optimizer

    def on_step_end(metrics, completed_steps, model):
        for key, value in list(metrics.items()):
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[key] = value.detach().cpu().item()
            elif hasattr(value, "item") and value.__class__.__module__.split(".")[0] == "numpy":
                metrics[key] = value.item()
        if completed_steps == 1:
            _align_static_graph_buckets_after_warmup(model)

    def run_step(step_ctx):
        return state.runner.step(step_ctx)

    def move_batch_to_device(batch, model):
        raw_model = unwrap_model(model)
        backbone = getattr(getattr(raw_model, "model", None), "backbone", None)
        prepare_host = getattr(backbone, "prepare_host_position_metadata", None)
        if callable(prepare_host):
            prepare_host(batch)
        device = next(model.parameters()).device
        if hasattr(batch, "to"):
            batch = batch.to(device)
        return batch

    def close():
        if state.runner is not None:
            state.runner.close()

    from loongforge.engines.torch.initialize import set_seed

    return TorchOptimization(
        wrap_model=wrap_model,
        build_optimizer=build_optimizer_fn,
        after_setup=lambda args: set_seed(args.seed),
        on_step_end=on_step_end,
        run_step=run_step,
        move_batch_to_device=move_batch_to_device,
        close=close,
    )
