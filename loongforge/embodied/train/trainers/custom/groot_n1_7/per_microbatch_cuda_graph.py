# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.7 per-microbatch CUDA graph runner."""

from __future__ import annotations

import contextlib
from contextlib import nullcontext
import dataclasses
import logging
import os
import queue
import time
from threading import Thread
from typing import Any

import torch
import torch.distributed as dist
from transformers.feature_extraction_utils import BatchFeature

from loongforge.embodied.distributed.utils import unwrap_model
from loongforge.embodied.train.utils.utils import resolve_dtype

logger = logging.getLogger(__name__)


class _NoopDdpLogger:
    def set_runtime_stats_and_log(self) -> None:
        """No-op implementation to suppress DDP logger output during CUDA graph capture."""
        return


def _clone_static(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        kwargs = {f.name: _clone_static(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return value.__class__(**kwargs)
    if isinstance(value, list):
        return [_clone_static(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_static(v) for v in value)
    if isinstance(value, dict):
        return {k: _clone_static(v) for k, v in value.items()}
    return value


def _copy_into_static(dst: Any, src: Any) -> None:
    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
        if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
            raise RuntimeError(
                "Tensor shape/dtype/device changed: "
                f"dst={tuple(dst.shape)}/{dst.dtype}/{dst.device}, "
                f"src={tuple(src.shape)}/{src.dtype}/{src.device}"
            )
        dst.copy_(src)
        return
    if dataclasses.is_dataclass(dst) and dataclasses.is_dataclass(src):
        if dst.__class__ is not src.__class__:
            raise RuntimeError(
                f"PreparedBatch type changed: {dst.__class__.__name__} vs {src.__class__.__name__}"
            )
        for f in dataclasses.fields(dst):
            _copy_into_static(getattr(dst, f.name), getattr(src, f.name))
        return
    if isinstance(dst, list) and isinstance(src, list):
        if len(dst) != len(src):
            raise RuntimeError(f"List length changed: {len(dst)} vs {len(src)}")
        for d_item, s_item in zip(dst, src):
            _copy_into_static(d_item, s_item)
        return
    if isinstance(dst, tuple) and isinstance(src, tuple):
        if len(dst) != len(src):
            raise RuntimeError(f"Tuple length changed: {len(dst)} vs {len(src)}")
        for d_item, s_item in zip(dst, src):
            _copy_into_static(d_item, s_item)
        return
    if isinstance(dst, dict) and isinstance(src, dict):
        if dst.keys() != src.keys():
            raise RuntimeError(f"Dict keys changed: {sorted(dst.keys())} vs {sorted(src.keys())}")
        for key in dst:
            _copy_into_static(dst[key], src[key])
        return
    if type(dst) is not type(src):
        raise RuntimeError(f"Value type changed: {type(dst).__name__} vs {type(src).__name__}")


def _as_output_dict(result: Any) -> dict[str, torch.Tensor]:
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError("GR00T train forward returned an empty tuple")
        loss = result[0]
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError(f"GR00T train forward returned non-tensor loss: {type(loss).__name__}")
        return {"action_loss": loss}
    if isinstance(result, dict):
        return result
    raise RuntimeError(f"GR00T train forward returned unsupported type: {type(result).__name__}")


class _GrootPerMicrobatchCudaGraphRunnerBase:
    """Capture one CUDAGraph per microbatch and replay with eager RNG injection."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.training_args = trainer.training_args
        self.ctx = trainer.ctx
        self.grad_accum = trainer.training_args.gradient_accumulation_steps
        self.warmup_steps = max(int(trainer.training_args.cuda_graph_warmup_steps), 1)

        self._warmup_count = 0
        self._captured = False
        self._graphs: list[torch.cuda.CUDAGraph] = []
        self._static_batches: list[Any] = []
        self._graph_outputs: list[dict[str, torch.Tensor]] = []
        self._noise_bufs: list[torch.Tensor] = []
        self._time_bufs: list[torch.Tensor] = []
        self._action_head = None
        self._raw_model = unwrap_model(trainer.model)
        self._grad_buffers_ready = False
        self._shared_pool = None
        self._graph_stream: torch.cuda.Stream | None = None
        self._materialized_sync_grad_ids: set[int] = set()
        self._profile_enabled = os.environ.get("LOONGFORGE_PROFILE_TRAIN_STEP", "0") == "1"
        self._last_profile: dict[str, float] = {}
        self._last_grad_sync_ms: float | None = None
        bucket_mb = float(self.training_args.cuda_graph_grad_sync_bucket_mb)
        self._max_grad_sync_bucket_numel = max(int(bucket_mb * 1024 * 1024 / 4), 1)
        self._grad_sync_impl = self.training_args.cuda_graph_grad_sync_impl
        self._grad_sync_dtype = self.training_args.cuda_graph_grad_sync_dtype
        self._grad_sync_reduce_op = (
            dist.ReduceOp.AVG if hasattr(dist.ReduceOp, "AVG") else dist.ReduceOp.SUM
        )
        self._grad_sync_needs_div = self._grad_sync_reduce_op == dist.ReduceOp.SUM
        self._sync_params: list[torch.nn.Parameter] | None = None
        self._sync_params_all_present = False
        self._grad_sync_buckets: list[list[torch.Tensor]] | None = None
        self._grad_sync_comm_buckets: list[list[torch.Tensor]] | None = None
        self._pending_graph_wait_ms: float | None = None
        self._prefetch_queue: queue.Queue | None = None
        self._prefetch_thread: Thread | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._prefetched_gpu_batches: list[Any] | None = None
        self._saved_ddp_logger = None
        self._noop_ddp_logger = _NoopDdpLogger()
        self._ddp_sync_in_graph = (
            self.ctx.is_distributed
            and self.ctx.world_size > 1
            and hasattr(trainer.model, "reducer")
            and self.training_args.cuda_graph_ddp_sync_in_graph
        )
        if self.ctx.is_main:
            logger.info(
                "CUDA graph gradient sync: %s",
                (
                    "ddp_in_graph"
                    if self._ddp_sync_in_graph
                    else f"manual_allreduce/{self._grad_sync_impl}/{self._grad_sync_dtype}"
                ),
            )

    @classmethod
    def is_enabled(cls, trainer) -> bool:
        """Return whether per-microbatch CUDA graph execution is enabled for this trainer.

        Args:
            trainer: Trainer instance whose ``args`` are inspected.

        Returns:
            ``True`` when CUDA is available, ``cuda_graph_impl`` is ``"local"``,
            and ``cuda_graph_scope`` is ``"per_microbatch"``; ``False`` otherwise.
        """
        training_args = trainer.training_args
        return (
            torch.cuda.is_available()
            and training_args.cuda_graph_impl == "local"
            and training_args.cuda_graph_scope == "per_microbatch"
        )

    def step(self) -> tuple[dict[str, torch.Tensor], float]:
        """Run one training step, dispatching between warmup, capture, and graph replay.

        During warmup the microbatches are executed eagerly. Once warmup is
        complete the CUDA graphs are captured on the first call, then replayed
        on every subsequent call. If a shape change is detected at replay time
        the graphs are invalidated and re-captured automatically.

        Returns:
            Tuple of ``(output_dict, loss)`` where *output_dict* contains at
            least ``"action_loss"`` and *loss* is the all-reduced scalar loss.
        """
        profile: dict[str, float] | None = {} if self._profile_enabled else None
        fetch_start = time.perf_counter()
        batches = self._fetch_prefetched_batches()
        if profile is not None:
            torch.cuda.synchronize()
            profile["fetch_ms"] = (time.perf_counter() - fetch_start) * 1000.0

        if self._warmup_count < self.warmup_steps:
            run_start = time.perf_counter()
            output, accum_loss = self._run_eager_batches(batches)
            if profile is not None:
                torch.cuda.synchronize()
                profile["runner_ms"] = (time.perf_counter() - run_start) * 1000.0
                if self._last_grad_sync_ms is not None:
                    profile["grad_sync_ms"] = self._last_grad_sync_ms
            self._warmup_count += 1
            self._grad_buffers_ready = self._has_any_grad_buffer()
            if self._warmup_count == self.warmup_steps:
                logger.info(
                    "CUDA graph warmup complete: scope=per_microbatch steps=%d",
                    self.warmup_steps,
                )
            result = self._sync_step_output(output)
            if profile is not None:
                self._last_profile = profile
            return result

        if not self._captured:
            run_start = time.perf_counter()
            output, _accum_loss = self._capture_and_replay(batches)
            if profile is not None:
                torch.cuda.synchronize()
                profile["runner_ms"] = (time.perf_counter() - run_start) * 1000.0
            result = self._sync_step_output(output)
            if profile is not None:
                self._last_profile = profile
            return result

        try:
            copy_start = time.perf_counter()
            self._load_static_batches(batches)
            if profile is not None:
                torch.cuda.synchronize()
                profile["static_copy_ms"] = (time.perf_counter() - copy_start) * 1000.0
        except RuntimeError as exc:
            logger.info("CUDA graph invalidated, re-capturing: %s", exc)
            self._invalidate()
            run_start = time.perf_counter()
            output, _accum_loss = self._capture_and_replay(batches)
            if profile is not None:
                torch.cuda.synchronize()
                profile["runner_ms"] = (time.perf_counter() - run_start) * 1000.0
            result = self._sync_step_output(output)
            if profile is not None:
                self._last_profile = profile
            return result

        run_start = time.perf_counter()
        output, _accum_loss = self._replay()
        if profile is not None:
            torch.cuda.synchronize()
            profile["runner_ms"] = (time.perf_counter() - run_start) * 1000.0
            if self._pending_graph_wait_ms is not None:
                profile["graph_wait_ms"] = self._pending_graph_wait_ms
                self._pending_graph_wait_ms = None
            if self._last_grad_sync_ms is not None:
                profile["grad_sync_ms"] = self._last_grad_sync_ms
        sync_start = time.perf_counter()
        result = self._sync_step_output(output)
        if profile is not None:
            torch.cuda.synchronize()
            profile["loss_sync_ms"] = (time.perf_counter() - sync_start) * 1000.0
            self._last_profile = profile
            if self.ctx.is_main:
                logger.info(
                    "CUDA graph runner profile: %s",
                    ", ".join(f"{key}={value:.2f}ms" for key, value in sorted(profile.items())),
                )
        return result

    def metrics_batch_size(self, local_batch_size: int) -> int:
        """Return the effective batch size used for metrics reporting.

        Args:
            local_batch_size: Per-rank batch size for the current step.

        Returns:
            The unmodified *local_batch_size*.
        """
        return local_batch_size

    def _sync_step_output(self, output: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
        if output is None or "action_loss" not in output:
            raise RuntimeError("GR00T train step did not produce an action_loss output")

        return output, float(output["action_loss"].detach())

    def _run_eager_batches(self, batches: list[Any]) -> tuple[dict[str, torch.Tensor], float]:
        output = None
        accum_loss = 0.0
        self._set_graph_warmup_flag(True)
        self._set_shape_recording(True)
        try:
            with self._graph_stream_ctx():
                for micro_idx, batch in enumerate(batches):
                    output = _as_output_dict(self.trainer._train_forward(batch))
                    loss = output["action_loss"] / self.grad_accum
                    self._backward(loss, micro_idx)
                    accum_loss += float(output["action_loss"].detach())
        finally:
            self._set_shape_recording(False)
            self._set_graph_warmup_flag(False)
            self._wait_default_on_graph_stream()
        sync_start = time.perf_counter()
        self._manual_sync_grads_if_needed()
        if self._profile_enabled:
            torch.cuda.synchronize()
            self._last_grad_sync_ms = (time.perf_counter() - sync_start) * 1000.0
        return output, accum_loss

    def _capture_and_replay(
        self,
        batches: list[Any],
    ) -> tuple[dict[str, torch.Tensor], float]:
        self._static_batches = [_clone_static(batch) for batch in batches]
        self._graphs = []
        self._graph_outputs = []
        self._allocate_rng_buffers()
        self._static_capture_warmup()

        self._zero_grad_for_capture(set_to_none=True)
        self._set_graph_warmup_flag(False)
        with self._preserve_rng_state():
            self._shared_pool = torch.cuda.graph_pool_handle()
            for micro_idx, static_batch in enumerate(self._static_batches):
                self._eager_rng_single(micro_idx)
                self._capture_one_microbatch(micro_idx, static_batch)

        self._clear_rng_bufs_on_model()
        self._grad_buffers_ready = True
        self._captured = True
        self._zero_grad_for_capture(set_to_none=False)
        logger.info(
            "Captured %d GR00T per-microbatch CUDA graph(s)",
            len(self._graphs),
        )
        return self._replay()

    def _replay(self) -> tuple[dict[str, torch.Tensor], float]:
        self._zero_grad_for_capture(set_to_none=False)
        accum_loss = 0.0
        with self._graph_stream_ctx():
            for micro_idx, graph in enumerate(self._graphs):
                self._eager_rng_single(micro_idx)
                self._set_rng_buf_on_model(micro_idx)
                graph.replay()
                output = self._graph_outputs[micro_idx]

        wait_start = time.perf_counter()
        self._wait_default_on_graph_stream()
        if self._profile_enabled:
            torch.cuda.synchronize()
            self._pending_graph_wait_ms = (time.perf_counter() - wait_start) * 1000.0
        self._clear_rng_bufs_on_model()
        sync_start = time.perf_counter()
        self._manual_sync_grads_if_needed()
        if self._profile_enabled:
            torch.cuda.synchronize()
            self._last_grad_sync_ms = (time.perf_counter() - sync_start) * 1000.0
        return self._graph_outputs[-1], accum_loss

    def _capture_one_microbatch(
        self,
        micro_idx: int,
        static_batch: Any,
    ) -> None:
        self._set_rng_buf_on_model(micro_idx)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        ddp_should_sync = self._prepare_ddp_capture(micro_idx)
        try:
            with self._graph_stream_ctx():
                with torch.cuda.graph(graph, pool=self._shared_pool, stream=self._get_graph_stream()):
                    output = _as_output_dict(self.trainer._train_forward(static_batch))
                    loss = output["action_loss"] / self.grad_accum
                    loss.backward()
        finally:
            self._restore_ddp_capture(ddp_should_sync)
        self._graphs.append(graph)
        self._graph_outputs.append(output)

    def _prepare_ddp_capture(self, micro_idx: int) -> bool | None:
        if not self._ddp_sync_in_graph:
            return None
        ddp_model = self.trainer.model
        should_sync = micro_idx == self.grad_accum - 1
        ddp_model.require_backward_grad_sync = should_sync
        ddp_model.require_forward_param_sync = should_sync
        if getattr(ddp_model, "find_unused_parameters", False) and not getattr(ddp_model, "static_graph", False):
            raise RuntimeError(
                "CUDA graph DDP sync capture requires --no-ddp-find-unused-parameters."
            )
        if self._saved_ddp_logger is None:
            self._saved_ddp_logger = ddp_model.logger
            ddp_model.logger = self._noop_ddp_logger
        return should_sync

    def _restore_ddp_capture(self, should_sync: bool | None) -> None:
        if should_sync is None:
            return
        ddp_model = self.trainer.model
        ddp_model.require_backward_grad_sync = True
        ddp_model.require_forward_param_sync = True
        if self._saved_ddp_logger is not None:
            ddp_model.logger = self._saved_ddp_logger
            self._saved_ddp_logger = None

    def _backward(self, loss: torch.Tensor, micro_idx: int) -> None:
        if self.ctx.is_distributed and micro_idx < self.grad_accum - 1 and hasattr(self.trainer.model, "no_sync"):
            with self.trainer.model.no_sync():
                loss.backward()
            return
        loss.backward()

    def _load_static_batches(self, batches: list[Any]) -> None:
        if len(batches) != len(self._static_batches):
            raise RuntimeError(
                f"Microbatch count changed: {len(self._static_batches)} -> {len(batches)}"
            )
        for static_batch, batch in zip(self._static_batches, batches):
            _copy_into_static(static_batch, batch)

    def _find_action_head(self):
        if self._action_head is not None:
            return self._action_head
        raw_model = unwrap_model(self.trainer.model)
        for _name, mod in raw_model.named_modules():
            if hasattr(mod, "action_encoder") and hasattr(mod, "sample_time"):
                self._action_head = mod
                return mod
        raise RuntimeError("Could not find GR00T action_head for CUDA graph RNG plumbing")

    def _set_shape_recording(self, enabled: bool) -> None:
        action_head = self._find_action_head()
        action_head._split_record_shape = enabled

    def _allocate_rng_buffers(self) -> None:
        action_head = self._find_action_head()
        actions_shape = action_head._split_actions_shape
        if actions_shape is None:
            raise RuntimeError(
                "GR00T action shape was not recorded during CUDA graph warmup. "
                "Increase --cuda-graph-warmup-steps if needed."
            )

        device = action_head._split_actions_device
        dtype = action_head._split_actions_dtype
        batch_size = actions_shape[0]

        self._noise_bufs = []
        self._time_bufs = []
        for _ in range(self.grad_accum):
            noise_buf = torch.empty(actions_shape, device=device, dtype=dtype)
            time_buf = torch.empty((batch_size,), device=device, dtype=dtype)
            self._noise_bufs.append(noise_buf)
            self._time_bufs.append(time_buf)

    def _eager_rng_single(self, micro_idx: int) -> None:
        action_head = self._find_action_head()
        noise_buf = self._noise_bufs[micro_idx]
        time_buf = self._time_bufs[micro_idx]

        noise_buf.copy_(torch.randn_like(noise_buf))
        sample = action_head.sample_time(
            time_buf.shape[0],
            device=time_buf.device,
            dtype=time_buf.dtype,
        )
        time_buf.copy_(sample)

    def _set_rng_buf_on_model(self, micro_idx: int) -> None:
        action_head = self._find_action_head()
        action_head._split_noise_buf = self._noise_bufs[micro_idx]
        action_head._split_time_buf = self._time_bufs[micro_idx]

    def _clear_rng_bufs_on_model(self) -> None:
        action_head = self._find_action_head()
        action_head._split_noise_buf = None
        action_head._split_time_buf = None

    def _set_graph_warmup_flag(self, value: bool) -> None:
        raw_model = unwrap_model(self.trainer.model)
        for module in raw_model.modules():
            module._in_graph_warmup = value

    def _static_capture_warmup(self) -> None:
        self._set_graph_warmup_flag(True)
        try:
            with self._preserve_rng_state():
                with self._graph_stream_ctx():
                    for _ in range(self.warmup_steps):
                        self._zero_grad_for_capture(set_to_none=True)
                        for micro_idx, static_batch in enumerate(self._static_batches):
                            self._eager_rng_single(micro_idx)
                            self._set_rng_buf_on_model(micro_idx)
                            output = _as_output_dict(self.trainer._train_forward(static_batch))
                            loss = output["action_loss"] / self.grad_accum
                            loss.backward()
                self._grad_buffers_ready = self._has_any_grad_buffer()
                self._zero_grad_for_capture(set_to_none=True)
        finally:
            self._clear_rng_bufs_on_model()
            self._set_graph_warmup_flag(False)
            self._wait_default_on_graph_stream()

    def _zero_grad_for_capture(self, *, set_to_none: bool = False) -> None:
        if set_to_none:
            self._materialized_sync_grad_ids.clear()
            self._grad_sync_buckets = None
            try:
                self.trainer.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                for param in self._raw_model.parameters():
                    param.grad = None
            return

        if self._grad_buffers_ready:
            for param in self._raw_model.parameters():
                if id(param) in self._materialized_sync_grad_ids:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.zero_()
            self._materialized_sync_grad_ids.clear()
            return
        try:
            self.trainer.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            self.trainer.optimizer.zero_grad()

    def zero_grad(self) -> None:
        """Set all model parameters' gradients to zero."""
        self._zero_grad_for_capture(set_to_none=not self._captured)

    def _has_any_grad_buffer(self) -> bool:
        return any(param.grad is not None for param in self._raw_model.parameters())

    def _manual_sync_grads_if_needed(self) -> None:
        if self._ddp_sync_in_graph:
            return
        if not self.ctx.is_distributed or self.ctx.world_size <= 1:
            return

        if self._sync_params is None:
            trainable_params = [param for param in self._raw_model.parameters() if param.requires_grad]
            self._sync_params_all_present = all(param.grad is not None for param in trainable_params)
            self._sync_params = (
                trainable_params
                if not self._sync_params_all_present
                else [param for param in trainable_params if param.grad is not None]
            )
        params = self._sync_params
        if not params:
            return

        if not self._sync_params_all_present:
            missing_flags = torch.tensor(
                [1 if param.grad is None else 0 for param in params],
                device=self.ctx.device,
                dtype=torch.int32,
            )
            dist.all_reduce(missing_flags, op=dist.ReduceOp.SUM)
        else:
            missing_flags = None

        if self._captured and self._sync_params_all_present and self._grad_sync_buckets is not None:
            self._sync_cached_grad_buckets()
            return

        bucket: list[torch.Tensor] = []
        bucket_numel = 0
        bucket_dtype = None
        bucket_device = None
        max_bucket_numel = self._max_grad_sync_bucket_numel
        pending_coalesced: list[tuple[Any, list[torch.Tensor], list[torch.Tensor] | None]] = []
        built_buckets: list[list[torch.Tensor]] = []
        built_comm_buckets: list[list[torch.Tensor] | None] = []

        def flush_bucket() -> None:
            nonlocal bucket, bucket_numel
            if not bucket:
                return
            if self._grad_sync_impl == "coalesced":
                bucket_to_reduce = self._make_comm_bucket(bucket)
                if bucket_to_reduce is not bucket:
                    self._copy_bucket(bucket_to_reduce, bucket)
                work = dist.all_reduce_coalesced(
                    bucket_to_reduce,
                    op=self._grad_sync_reduce_op,
                    async_op=True,
                )
                pending_coalesced.append((
                    work,
                    bucket_to_reduce,
                    None if bucket_to_reduce is bucket else bucket,
                ))
            else:
                bucket_to_reduce = self._make_comm_bucket(bucket)
                if bucket_to_reduce is not bucket:
                    self._copy_bucket(bucket_to_reduce, bucket)
                flat = torch._utils._flatten_dense_tensors(bucket_to_reduce)
                dist.all_reduce(flat, op=self._grad_sync_reduce_op)
                if self._grad_sync_needs_div:
                    flat.div_(self.ctx.world_size)
                for synced, grad in zip(torch._utils._unflatten_dense_tensors(flat, bucket_to_reduce), bucket):
                    grad.copy_(synced)
            if self._captured and self._sync_params_all_present:
                built_buckets.append(bucket)
                built_comm_buckets.append(bucket_to_reduce if bucket_to_reduce is not bucket else None)
            bucket = []
            bucket_numel = 0

        for idx, param in enumerate(params):
            if param.grad is None:
                if missing_flags is not None and int(missing_flags[idx].item()) == self.ctx.world_size:
                    continue
                param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
                self._materialized_sync_grad_ids.add(id(param))

            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                flush_bucket()
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad.div_(self.ctx.world_size)
                continue

            if bucket and (grad.dtype != bucket_dtype or grad.device != bucket_device):
                flush_bucket()
            bucket.append(grad)
            bucket_dtype = grad.dtype
            bucket_device = grad.device
            bucket_numel += grad.numel()
            if bucket_numel >= max_bucket_numel:
                flush_bucket()

        flush_bucket()
        for work, reduced_bucket, target_bucket in pending_coalesced:
            work.wait()
            if self._grad_sync_needs_div:
                torch._foreach_div_(reduced_bucket, self.ctx.world_size)
            if target_bucket is not None:
                self._copy_bucket(target_bucket, reduced_bucket)
        if self._captured and self._sync_params_all_present:
            self._grad_sync_buckets = built_buckets
            self._grad_sync_comm_buckets = built_comm_buckets if any(
                comm_bucket is not None for comm_bucket in built_comm_buckets
            ) else None

    def _sync_cached_grad_buckets(self) -> None:
        pending_coalesced: list[tuple[Any, list[torch.Tensor], list[torch.Tensor] | None]] = []
        comm_buckets = self._grad_sync_comm_buckets or []
        for bucket_idx, bucket in enumerate(self._grad_sync_buckets or []):
            if not bucket:
                continue
            if self._grad_sync_impl == "coalesced":
                bucket_to_reduce = bucket
                target_bucket = None
                if comm_buckets and comm_buckets[bucket_idx] is not None:
                    bucket_to_reduce = comm_buckets[bucket_idx]
                    target_bucket = bucket
                    self._copy_bucket(bucket_to_reduce, bucket)
                work = dist.all_reduce_coalesced(
                    bucket_to_reduce,
                    op=self._grad_sync_reduce_op,
                    async_op=True,
                )
                pending_coalesced.append((work, bucket_to_reduce, target_bucket))
            else:
                bucket_to_reduce = bucket
                if comm_buckets and comm_buckets[bucket_idx] is not None:
                    bucket_to_reduce = comm_buckets[bucket_idx]
                    self._copy_bucket(bucket_to_reduce, bucket)
                flat = torch._utils._flatten_dense_tensors(bucket_to_reduce)
                dist.all_reduce(flat, op=self._grad_sync_reduce_op)
                if self._grad_sync_needs_div:
                    flat.div_(self.ctx.world_size)
                for synced, grad in zip(torch._utils._unflatten_dense_tensors(flat, bucket_to_reduce), bucket):
                    grad.copy_(synced)
        for work, bucket, target_bucket in pending_coalesced:
            work.wait()
            if self._grad_sync_needs_div:
                torch._foreach_div_(bucket, self.ctx.world_size)
            if target_bucket is not None:
                self._copy_bucket(target_bucket, bucket)

    def _make_comm_bucket(self, bucket: list[torch.Tensor]) -> list[torch.Tensor]:
        if self._grad_sync_dtype != "bf16":
            return bucket
        if all(grad.dtype == torch.bfloat16 for grad in bucket):
            return bucket
        return [
            torch.empty_like(grad, dtype=torch.bfloat16, memory_format=torch.preserve_format)
            for grad in bucket
        ]

    @staticmethod
    def _copy_bucket(dst_bucket: list[torch.Tensor], src_bucket: list[torch.Tensor]) -> None:
        for dst, src in zip(dst_bucket, src_bucket):
            dst.copy_(src)

    def _ensure_prefetch_thread(self) -> None:
        if self._prefetch_queue is not None:
            return
        prefetch_count = max(int(os.environ.get("LOONGFORGE_BATCH_PREFETCH_COUNT", "2")), 0)
        if prefetch_count <= 0:
            return
        self._prefetch_queue = queue.Queue(maxsize=prefetch_count)

        def _worker() -> None:
            while True:
                try:
                    batches = [
                        self.trainer._fetch_batch_cpu("vla")
                        for _ in range(self.grad_accum)
                    ]
                    self._prefetch_queue.put(batches)
                except BaseException as exc:  # noqa: BLE001
                    self._prefetch_queue.put(exc)
                    return

        self._prefetch_thread = Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()

    def _fetch_prefetched_batches(self) -> list[Any]:
        if self._prefetched_gpu_batches is None:
            self._start_gpu_prefetch()
        if self._prefetched_gpu_batches is None:
            return [
                self.trainer._fetch_batch("vla")
                for _ in range(self.grad_accum)
            ]

        if self._copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self._copy_stream)
        batches = self._prefetched_gpu_batches
        self._prefetched_gpu_batches = None
        self._start_gpu_prefetch()
        return batches

    def _fetch_prefetched_cpu_batches(self) -> list[Any]:
        self._ensure_prefetch_thread()
        if self._prefetch_queue is None:
            return [
                self.trainer._fetch_batch_cpu("vla")
                for _ in range(self.grad_accum)
            ]
        item = self._prefetch_queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def _start_gpu_prefetch(self) -> None:
        if self._prefetched_gpu_batches is not None:
            return
        cpu_batches = self._fetch_prefetched_cpu_batches()
        stream = self._get_copy_stream()
        with torch.cuda.stream(stream):
            self._prefetched_gpu_batches = [
                self.trainer._move_batch_to_device(batch)
                for batch in cpu_batches
            ]

    def _get_copy_stream(self) -> torch.cuda.Stream:
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        return self._copy_stream

    def _clear_warmup_sentinels(self) -> None:
        raw_model = unwrap_model(self.trainer.model)
        for module in raw_model.modules():
            if hasattr(module, "_capture_has_invalid_images"):
                module._capture_has_invalid_images = False

    @contextlib.contextmanager
    def _preserve_rng_state(self):
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()
        try:
            yield
        finally:
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_states)

    def _get_graph_stream(self) -> torch.cuda.Stream:
        if self._graph_stream is None:
            self._graph_stream = torch.cuda.Stream()
        return self._graph_stream

    @contextlib.contextmanager
    def _graph_stream_ctx(self):
        stream = self._get_graph_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            yield

    def _wait_default_on_graph_stream(self) -> None:
        if self._graph_stream is not None:
            torch.cuda.current_stream().wait_stream(self._graph_stream)

    def _invalidate(self) -> None:
        self._warmup_count = 0
        self._captured = False
        self._graphs = []
        self._static_batches = []
        self._graph_outputs = []
        self._noise_bufs = []
        self._time_bufs = []
        self._shared_pool = None
        self._clear_rng_bufs_on_model()
        self._clear_warmup_sentinels()


@dataclasses.dataclass
class GrootN1d7ActionGraphBatch:
    """Static CUDA graph inputs after the frozen Qwen backbone has run eagerly."""

    backbone_output: BatchFeature
    action_input: BatchFeature


def _clone_static_n1d7(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, BatchFeature):
        return BatchFeature(data={k: _clone_static_n1d7(v) for k, v in value.items()})
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        kwargs = {
            f.name: _clone_static_n1d7(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
        return value.__class__(**kwargs)
    if isinstance(value, list):
        return [_clone_static_n1d7(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_static_n1d7(v) for v in value)
    if isinstance(value, dict):
        return {k: _clone_static_n1d7(v) for k, v in value.items()}
    return value


def _copy_into_static_n1d7(dst: Any, src: Any) -> None:
    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
        if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
            raise RuntimeError(
                "Tensor shape/dtype/device changed: "
                f"dst={tuple(dst.shape)}/{dst.dtype}/{dst.device}, "
                f"src={tuple(src.shape)}/{src.dtype}/{src.device}"
            )
        dst.copy_(src)
        return
    if isinstance(dst, BatchFeature) and isinstance(src, BatchFeature):
        if dst.keys() != src.keys():
            raise RuntimeError(f"BatchFeature keys changed: {sorted(dst.keys())} vs {sorted(src.keys())}")
        for key in dst:
            _copy_into_static_n1d7(dst[key], src[key])
        return
    if dataclasses.is_dataclass(dst) and dataclasses.is_dataclass(src):
        if dst.__class__ is not src.__class__:
            raise RuntimeError(
                f"PreparedBatch type changed: {dst.__class__.__name__} vs {src.__class__.__name__}"
            )
        for f in dataclasses.fields(dst):
            _copy_into_static_n1d7(getattr(dst, f.name), getattr(src, f.name))
        return
    if isinstance(dst, list) and isinstance(src, list):
        if len(dst) != len(src):
            raise RuntimeError(f"List length changed: {len(dst)} vs {len(src)}")
        for d_item, s_item in zip(dst, src):
            _copy_into_static_n1d7(d_item, s_item)
        return
    if isinstance(dst, tuple) and isinstance(src, tuple):
        if len(dst) != len(src):
            raise RuntimeError(f"Tuple length changed: {len(dst)} vs {len(src)}")
        for d_item, s_item in zip(dst, src):
            _copy_into_static_n1d7(d_item, s_item)
        return
    if isinstance(dst, dict) and isinstance(src, dict):
        if dst.keys() != src.keys():
            raise RuntimeError(f"Dict keys changed: {sorted(dst.keys())} vs {sorted(src.keys())}")
        for key in dst:
            _copy_into_static_n1d7(dst[key], src[key])
        return
    if type(dst) is not type(src):
        raise RuntimeError(f"Value type changed: {type(dst).__name__} vs {type(src).__name__}")


class GrootN1d7PerMicrobatchCudaGraphRunner(_GrootPerMicrobatchCudaGraphRunnerBase):
    """Capture GR00T-N1.7 action-head train graph while running Qwen eagerly."""

    def __init__(self, trainer):
        super().__init__(trainer)
        self._state_dropout_bufs: list[torch.Tensor] = []

    def step(self) -> tuple[dict[str, torch.Tensor], float]:
        self._broadcast_buffers_for_ddp_parity()
        return super().step()

    def _raw_policy(self):
        return unwrap_model(self.trainer.model)

    def _raw_groot_model(self):
        raw_policy = self._raw_policy()
        model = getattr(raw_policy, "model", None)
        if model is None:
            raise RuntimeError("GrootN1d7Policy.model was not found for CUDA graph runner")
        return model

    def _prepare_action_graph_batch(self, batch: Any) -> GrootN1d7ActionGraphBatch:
        model = self._raw_groot_model()
        backbone = model.backbone
        if any(param.requires_grad for param in backbone.parameters()):
            raise RuntimeError(
                "GR00T-N1.7 action-head CUDA graph requires a frozen Qwen backbone."
            )

        batch_inputs = batch.to_model_inputs()
        backbone_inputs, action_inputs = model.prepare_input(batch_inputs)
        with torch.no_grad(), self._train_autocast_context():
            backbone_output = backbone(backbone_inputs)
        return GrootN1d7ActionGraphBatch(
            backbone_output=backbone_output,
            action_input=action_inputs,
        )

    def _broadcast_buffers_for_ddp_parity(self) -> None:
        if not self.ctx.is_distributed or self.ctx.world_size <= 1:
            return
        if not bool(getattr(self.training_args, "ddp_broadcast_buffers", True)):
            return
        for buffer in self._raw_model.buffers():
            if buffer is not None and buffer.device.type == "cuda":
                dist.broadcast(buffer, src=0)

    def _capture_and_replay(self, batches: list[Any]) -> tuple[dict[str, torch.Tensor], float]:
        action_batches = [self._prepare_action_graph_batch(batch) for batch in batches]
        self._static_batches = [_clone_static_n1d7(batch) for batch in action_batches]
        self._graphs = []
        self._graph_outputs = []
        self._allocate_rng_buffers()
        self._debug_split_eager_loss_once()
        debug_eager_grad_norm = self._debug_eager_backward_grad_norm_once()
        self._static_capture_warmup()

        self._zero_grad_for_capture(set_to_none=True)
        self._set_graph_warmup_flag(False)
        with self._preserve_rng_state():
            self._shared_pool = torch.cuda.graph_pool_handle()
            for micro_idx, static_batch in enumerate(self._static_batches):
                self._eager_rng_single(micro_idx)
                self._capture_one_microbatch(micro_idx, static_batch)

        self._clear_rng_bufs_on_model()
        self._grad_buffers_ready = True
        self._captured = True
        self._zero_grad_for_capture(set_to_none=False)
        logger.info(
            "Captured %d GR00T-N1.7 action-head per-microbatch CUDA graph(s)",
            len(self._graphs),
        )
        result = self._replay()
        self._debug_graph_backward_grad_norm_once(debug_eager_grad_norm)
        return result

    def _load_static_batches(self, batches: list[Any]) -> None:
        action_batches = [self._prepare_action_graph_batch(batch) for batch in batches]
        if len(action_batches) != len(self._static_batches):
            raise RuntimeError(
                f"Microbatch count changed: {len(self._static_batches)} -> {len(action_batches)}"
            )
        for static_batch, batch in zip(self._static_batches, action_batches):
            _copy_into_static_n1d7(static_batch, batch)

    def _debug_split_eager_loss_once(self) -> None:
        if os.environ.get("GROOT_CUDA_GRAPH_DEBUG_SPLIT_EAGER_LOSS", "0") != "1":
            return
        direct_losses: list[float] = []
        buffered_losses: list[float] = []

        def _mean_loss(output: dict[str, torch.Tensor]) -> float:
            loss = output["action_loss"].detach().float()
            if self.ctx.is_distributed and self.ctx.world_size > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss.div_(self.ctx.world_size)
            return float(loss)

        try:
            with self._preserve_rng_state():
                for static_batch in self._static_batches:
                    self._clear_rng_bufs_on_model()
                    with torch.no_grad():
                        output = self._train_forward_output(static_batch)
                    direct_losses.append(_mean_loss(output))

            with self._preserve_rng_state():
                for micro_idx, static_batch in enumerate(self._static_batches):
                    self._eager_rng_single(micro_idx)
                    self._set_rng_buf_on_model(micro_idx)
                    with torch.no_grad():
                        output = self._train_forward_output(static_batch)
                    buffered_losses.append(_mean_loss(output))
        finally:
            self._clear_rng_bufs_on_model()
        if self.ctx.is_main:
            logger.info(
                "CUDA graph debug split eager action losses before capture: direct=%s buffered=%s",
                direct_losses,
                buffered_losses,
            )

    def _debug_enabled(self, name: str) -> bool:
        return os.environ.get(name, "0") == "1"

    def _current_grad_norm(self) -> float:
        device = self.ctx.device
        total = torch.zeros((), device=device, dtype=torch.float32)
        for param in self._raw_model.parameters():
            grad = param.grad
            if grad is None:
                continue
            total += grad.detach().float().pow(2).sum()
        return float(total.sqrt())

    def _debug_eager_backward_grad_norm_once(self) -> float | None:
        if not self._debug_enabled("GROOT_CUDA_GRAPH_DEBUG_BACKWARD_GRAD_NORM"):
            return None

        try:
            self._zero_grad_for_capture(set_to_none=True)
            self._set_graph_warmup_flag(False)
            with self._preserve_rng_state():
                with self._graph_stream_ctx():
                    for micro_idx, static_batch in enumerate(self._static_batches):
                        self._eager_rng_single(micro_idx)
                        self._set_rng_buf_on_model(micro_idx)
                        output = self._train_forward_output(static_batch)
                        loss = output["action_loss"] / self.grad_accum
                        with self._manual_ddp_no_sync_context():
                            loss.backward()
            self._wait_default_on_graph_stream()
            pre_sync_norm = self._current_grad_norm()
            self._manual_sync_grads_if_needed()
            post_sync_norm = self._current_grad_norm()
            if self.ctx.is_main:
                logger.info(
                    "CUDA graph debug eager backward grad norm before capture: pre_sync=%.9f post_sync=%.9f",
                    pre_sync_norm,
                    post_sync_norm,
                )
            return post_sync_norm
        finally:
            self._clear_rng_bufs_on_model()
            self._zero_grad_for_capture(set_to_none=True)
            self._wait_default_on_graph_stream()

    def _debug_graph_backward_grad_norm_once(self, eager_grad_norm: float | None) -> None:
        if eager_grad_norm is None:
            return
        graph_grad_norm = self._current_grad_norm()
        if self.ctx.is_main:
            logger.info(
                "CUDA graph debug replay backward grad norm after sync: eager=%.9f graph=%.9f diff=%.9f",
                eager_grad_norm,
                graph_grad_norm,
                abs(eager_grad_norm - graph_grad_norm),
            )

    def _advance_cuda_rng_before_split_buffers(self, static_batch: Any | None) -> None:
        if not isinstance(static_batch, GrootN1d7ActionGraphBatch):
            return
        action_head = self._raw_groot_model().action_head
        action_head.set_frozen_modules_to_eval_mode()
        with torch.no_grad(), self._train_autocast_context():
            action_head.process_backbone_output(static_batch.backbone_output)

    def _train_autocast_context(self):
        if self.trainer._cfg_bool("disable_train_autocast", False):
            return nullcontext()
        dtype = getattr(self.trainer, "_compute_dtype", None)
        if dtype is None:
            dtype = resolve_dtype(self.training_args.dtype)
            self.trainer._compute_dtype = dtype
        return torch.autocast("cuda", dtype=dtype)

    def _manual_ddp_no_sync_context(self):
        if self._ddp_sync_in_graph:
            return nullcontext()
        no_sync = getattr(self.trainer.model, "no_sync", None)
        if not callable(no_sync):
            return nullcontext()
        return no_sync()

    def _eager_rng_single(self, micro_idx: int) -> None:
        action_head = self._find_action_head()
        noise_buf = self._noise_bufs[micro_idx]
        time_buf = self._time_bufs[micro_idx]
        static_batch = (
            self._static_batches[micro_idx]
            if micro_idx < len(self._static_batches)
            else None
        )

        cuda_state = None
        cuda_device = noise_buf.device if noise_buf.is_cuda else None
        if cuda_device is not None:
            cuda_state = torch.cuda.get_rng_state(cuda_device)
        try:
            self._advance_cuda_rng_before_split_buffers(static_batch)
            if self._state_dropout_bufs:
                dropout_buf = self._state_dropout_bufs[micro_idx]
                dropout = (
                    torch.rand(dropout_buf.shape[0], device=dropout_buf.device)
                    < float(action_head.state_dropout_prob)
                )
                dropout_buf.copy_(dropout)
            noise_buf.copy_(torch.randn_like(noise_buf))
        finally:
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, cuda_device)

        sample = action_head.sample_time(
            time_buf.shape[0],
            device=time_buf.device,
            dtype=time_buf.dtype,
        )
        time_buf.copy_(sample)

    def _allocate_rng_buffers(self) -> None:
        action_head = self._find_action_head()
        actions_shape = action_head._split_actions_shape
        if actions_shape is None:
            raise RuntimeError(
                "GR00T-N1.7 action shape was not recorded during CUDA graph warmup. "
                "Increase --cuda-graph-warmup-steps if needed."
            )

        device = action_head._split_actions_device
        dtype = action_head._split_actions_dtype
        batch_size = actions_shape[0]

        self._noise_bufs = []
        self._time_bufs = []
        self._state_dropout_bufs = []
        needs_state_dropout_buf = (
            bool(getattr(action_head, "_split_supports_state_dropout_buf", False))
            and bool(getattr(action_head, "training", False))
            and float(getattr(action_head, "state_dropout_prob", 0.0)) > 0.0
        )
        for _ in range(self.grad_accum):
            self._noise_bufs.append(torch.empty(actions_shape, device=device, dtype=dtype))
            self._time_bufs.append(torch.empty((batch_size,), device=device, dtype=dtype))
            if needs_state_dropout_buf:
                self._state_dropout_bufs.append(
                    torch.empty((batch_size,), device=device, dtype=torch.bool)
                )

    def _set_rng_buf_on_model(self, micro_idx: int) -> None:
        action_head = self._find_action_head()
        if self._state_dropout_bufs:
            action_head._split_state_dropout_buf = self._state_dropout_bufs[micro_idx]
        action_head._split_noise_buf = self._noise_bufs[micro_idx]
        action_head._split_time_buf = self._time_bufs[micro_idx]

    def _clear_rng_bufs_on_model(self) -> None:
        action_head = self._find_action_head()
        if hasattr(action_head, "_split_state_dropout_buf"):
            action_head._split_state_dropout_buf = None
        action_head._split_noise_buf = None
        action_head._split_time_buf = None

    def _static_capture_warmup(self) -> None:
        self._set_graph_warmup_flag(True)
        try:
            with self._preserve_rng_state():
                with self._graph_stream_ctx():
                    for _ in range(self.warmup_steps):
                        self._zero_grad_for_capture(set_to_none=True)
                        for micro_idx, static_batch in enumerate(self._static_batches):
                            self._eager_rng_single(micro_idx)
                            self._set_rng_buf_on_model(micro_idx)
                            output = self._train_forward_output(static_batch)
                            loss = output["action_loss"] / self.grad_accum
                            loss.backward()
                self._grad_buffers_ready = self._has_any_grad_buffer()
                self._zero_grad_for_capture(set_to_none=True)
        finally:
            self._clear_rng_bufs_on_model()
            self._set_graph_warmup_flag(False)
            self._wait_default_on_graph_stream()

    def _capture_one_microbatch(
        self,
        micro_idx: int,
        static_batch: GrootN1d7ActionGraphBatch,
    ) -> None:
        self._set_rng_buf_on_model(micro_idx)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        ddp_should_sync = self._prepare_ddp_capture(micro_idx)
        try:
            with self._graph_stream_ctx():
                with torch.cuda.graph(graph, pool=self._shared_pool, stream=self._get_graph_stream()):
                    output = self._train_forward_output(static_batch)
                    loss = output["action_loss"] / self.grad_accum
                    with self._manual_ddp_no_sync_context():
                        loss.backward()
        finally:
            self._restore_ddp_capture(ddp_should_sync)
        self._graphs.append(graph)
        self._graph_outputs.append(output)

    def _train_forward_output(self, batch: Any) -> dict[str, torch.Tensor]:
        if not isinstance(batch, GrootN1d7ActionGraphBatch):
            return _as_output_dict(self.trainer._train_forward(batch))

        action_head = self._raw_groot_model().action_head
        with self._train_autocast_context():
            output = action_head(batch.backbone_output, batch.action_input)
        loss = output.get("loss", None)
        if loss is None:
            action_loss = output["action_loss"]
            action_mask = output.get("action_mask", torch.ones_like(action_loss))
            loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        return {"action_loss": loss}

    def _invalidate(self) -> None:
        super()._invalidate()
        self._state_dropout_bufs = []
