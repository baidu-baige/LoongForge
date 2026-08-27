# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.7 finetune trainer with full-iteration CUDA graph support."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.distributed as dist
from torch import nn

from loongforge.embodied.distributed.parallel import wrap_model
from loongforge.embodied.train.trainers.custom.groot_n1_7.groot_optimizer import (
    build_groot_optimizer,
)
from loongforge.embodied.distributed.utils import unwrap_model
from loongforge.embodied.train.trainers.custom.groot_n1_7.full_iteration_cuda_graph import (
    GrootN1d7FullIterationCudaGraphRunner,
)

from loongforge.embodied.train.trainers.supervised.finetune_trainer import FinetuneTrainer
from loongforge.embodied.train.utils.utils import set_seed

logger = logging.getLogger(__name__)


@dataclass
class _StaticGraphBucketWarmup:
    """Capture the first backward's ready order for static DDP buckets."""

    parameters: list[nn.Parameter]
    expect_sparse_gradients: list[bool]
    ready_order: list[int] = field(default_factory=list)
    recorded_indices: set[int] = field(default_factory=set)
    hook_handles: list = field(default_factory=list)

    def record_ready(self, parameter_index: int, _parameter: nn.Parameter) -> None:
        """Append a parameter index the first time its gradient becomes ready.

        Gradient accumulation runs one backward per micro-batch, so the hook
        fires several times per optimizer step; only the first backward defines
        the bucket order.
        """
        if parameter_index in self.recorded_indices:
            return
        self.recorded_indices.add(parameter_index)
        self.ready_order.append(parameter_index)

    def remove_hooks(self) -> None:
        """Detach all registered gradient-ready hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


def _arm_static_graph_bucket_warmup(model) -> None:
    """Record normal DDP's first-backward parameter ready order."""
    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return
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
        from loongforge.embodied.train.trainers.custom.groot_n1_7.groot_ddp_reducer_bucket_control import (
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
    low_priority, high_priority = torch.cuda.Stream.priority_range()
    default = high_priority if backbone_pipeline else 0
    # The split Graph benchmark selected priority -3 on this hardware. Clamp
    # the model-owned value to the device's supported range.
    return max(high_priority, min(low_priority, -3 if backbone_pipeline else default))


def _select_cuda_graph_runner_type(trainer):
    training_args = trainer.training_args
    if training_args.cuda_graph_impl != "local":
        return None
    if training_args.cuda_graph_scope == "full_iteration":
        if not GrootN1d7FullIterationCudaGraphRunner.is_enabled(trainer):
            raise RuntimeError(
                "GR00T-N1.7 full-iteration CUDA graph was requested but CUDA is unavailable; "
                "eager fallback is forbidden."
            )
        return GrootN1d7FullIterationCudaGraphRunner
    raise RuntimeError(
        "GR00T-N1.7 only supports --cuda-graph-scope=full_iteration; "
        f"got {training_args.cuda_graph_scope!r}."
    )


class GrootN1d7Trainer(FinetuneTrainer):
    """GR00T-N1.7 finetune trainer with an optional full-iteration CUDA graph."""

    def __init__(self, training_args, model_cfg, data_cfg):
        super().__init__(training_args, model_cfg, data_cfg)
        self._train_step_runner: GrootN1d7FullIterationCudaGraphRunner | None = None
        self._full_iteration_graph_stream: torch.cuda.Stream | None = None

    def _setup(self) -> None:
        """Build training resources and match HF Trainer's post-build RNG reset."""
        super()._setup()
        set_seed(self.training_args.seed)

    def _on_train_begin(self) -> None:
        """Log the complete GR00T-N1.7 module tree once before training."""
        super()._on_train_begin()
        # The full module tree is intentionally omitted to keep startup logs compact.

    def _wrap_model_for_training(self) -> None:
        """Wrap the model, installing the CUDA graph runner when enabled."""
        runner_type = _select_cuda_graph_runner_type(self)
        if runner_type is GrootN1d7FullIterationCudaGraphRunner:
            default_stream = torch.cuda.current_stream(self.ctx.device)
            backbone_pipeline = True
            graph_priority = _full_iteration_graph_stream_priority(backbone_pipeline)
            graph_stream = torch.cuda.Stream(
                device=self.ctx.device,
                priority=graph_priority,
            )
            graph_stream.wait_stream(default_stream)
            with torch.cuda.stream(graph_stream):
                self.model = wrap_model(self.model, self.training_args, self.ctx)
            _arm_static_graph_bucket_warmup(self.model)
            default_stream.wait_stream(graph_stream)
            torch.cuda.synchronize(self.ctx.device)
            self._full_iteration_graph_stream = graph_stream
            self._train_step_runner = GrootN1d7FullIterationCudaGraphRunner(
                self,
                graph_stream,
            )
            logger.info(
                "Using model-managed train step runner: %s (stream_priority=%d)",
                self._train_step_runner.__class__.__name__,
                graph_priority,
            )
            return

        if runner_type is None:
            super()._wrap_model_for_training()
            return

    def _build_optimizer(self):
        """Keep eager on the reference optimizer; graph owns its capturable path."""
        if self._full_iteration_graph_stream is None:
            return build_groot_optimizer(
                self.model,
                self.training_args,
                capturable=False,
            )
        default_stream = torch.cuda.current_stream(self.ctx.device)
        self._full_iteration_graph_stream.wait_stream(default_stream)
        with torch.cuda.stream(self._full_iteration_graph_stream):
            optimizer = build_groot_optimizer(
                self.model,
                self.training_args,
            )
        default_stream.wait_stream(self._full_iteration_graph_stream)
        return optimizer

    def _on_step_end(self, metrics):
        # The capturable optimizer intentionally stores each LR as a CUDA
        # scalar tensor.  Normalize graph-owned scalar values before the
        # shared logger exports JSONL; public framework logging remains
        # unchanged for other trainers.
        for key, value in list(metrics.items()):
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[key] = value.detach().cpu().item()
            elif hasattr(value, "item") and value.__class__.__module__.split(".")[0] == "numpy":
                metrics[key] = value.item()
        if (
            isinstance(self._train_step_runner, GrootN1d7FullIterationCudaGraphRunner)
            and self.completed_steps == 1
        ):
            _align_static_graph_buckets_after_warmup(self.model)

    def _train_step(self):
        """Let the full-iteration runner own the complete optimizer step."""
        if isinstance(self._train_step_runner, GrootN1d7FullIterationCudaGraphRunner):
            return self._train_step_runner.step()
        return super()._train_step()

    def _move_batch_to_device(self, batch):
        """Move a batch while preparing Qwen metadata on the owning trainer."""
        full_iteration_graph = isinstance(
            self._train_step_runner,
            GrootN1d7FullIterationCudaGraphRunner,
        )
        raw_model = unwrap_model(self.model)
        backbone = getattr(getattr(raw_model, "model", None), "backbone", None)
        prepare_host = getattr(backbone, "prepare_host_position_metadata", None)
        if callable(prepare_host):
            prepare_host(batch)

        batch = super()._move_batch_to_device(batch)
        # Full-iteration capture prepares pointer-bound Vision metadata once on
        # its static batch. Rebuilding it on each transient GPU batch would
        # reintroduce scalar D2H synchronization in the Graph path.
        if full_iteration_graph:
            return batch

        # Eager uses pointer-bound metadata on each transient GPU batch.
        prepare = getattr(backbone, "prepare_cuda_graph_batch", None)
        if callable(prepare):
            prepare(batch)
        return batch

    def _clean_nan_gradients(self) -> None:
        """Skip host-side gradient cleanup while graph-owned grad buffers are active."""
        if self._train_step_runner is not None:
            return
        super()._clean_nan_gradients()

    def _finalize(self) -> None:
        """Release captured NCCL graph resources before process-group teardown."""
        if self._train_step_runner is not None:
            self._train_step_runner.close()
        super()._finalize()
