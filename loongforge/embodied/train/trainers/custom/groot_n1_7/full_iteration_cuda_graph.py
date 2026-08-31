# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Full-iteration CUDA graph runner for GR00T-N1.7 training."""

from __future__ import annotations

import dataclasses
import ctypes
import logging
from contextlib import nullcontext
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from loongforge.embodied.distributed.utils import unwrap_model
from loongforge.embodied.train.utils.utils import resolve_dtype
logger = logging.getLogger(__name__)


_CUDA_EVENT_RECORD_EXTERNAL = 1
_cuda_event_record_with_flags = None


def _record_external_cuda_event(
    event: torch.cuda.Event,
    stream: torch.cuda.Stream,
) -> None:
    """Record a CUDA-graph event as an externally visible event."""
    global _cuda_event_record_with_flags
    if _cuda_event_record_with_flags is None:
        cudart = ctypes.CDLL("libcudart.so")
        record = cudart.cudaEventRecordWithFlags
        record.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        record.restype = ctypes.c_int
        _cuda_event_record_with_flags = record
    error = _cuda_event_record_with_flags(
        ctypes.c_void_p(event.cuda_event),
        ctypes.c_void_p(stream.cuda_stream),
        _CUDA_EVENT_RECORD_EXTERNAL,
    )
    if error:
        raise RuntimeError(
            "cudaEventRecordWithFlags(cudaEventRecordExternal) failed: "
            f"error={error}"
        )


@torch.no_grad()
def _compute_grad_norm_and_clip_scale(
    gradients: list[torch.Tensor],
    max_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.nn.utils.clip_grad import _get_total_norm

    total_norm = _get_total_norm(
        gradients,
        norm_type=2.0,
        error_if_nonfinite=False,
        foreach=None,
    )
    clip_coef = float(max_norm) / (total_norm + 1e-6)
    return total_norm, torch.clamp(clip_coef, max=1.0)


def _should_zero_grad_before_iteration(
    *,
    set_to_none: bool,
    direct_grad_write: bool,
) -> bool:
    """Return whether this Python execution should emit a gradient reset."""
    return set_to_none or not direct_grad_write


class _NoopDdpLogger:
    def set_runtime_stats_and_log(self) -> None:
        """Swallow DDP runtime-stat logging while the graph owns the iteration."""
        return


@dataclasses.dataclass
class _GraphOutputs:
    action_loss: torch.Tensor
    grad_norm: torch.Tensor
    nan_flag: torch.Tensor
    spike_flag: torch.Tensor


@dataclasses.dataclass
class _GraphValidationBatch:
    image_grid_thw: torch.Tensor | None
    input_ids: torch.Tensor | None


@dataclasses.dataclass
class _ActionGraphBatch:
    """Static inputs for the trainable action-head graph."""

    backbone_output: BatchFeature
    action_input: BatchFeature

    def to_action_head_inputs(self) -> tuple[BatchFeature, BatchFeature]:
        """Return the static backbone output and action input pair."""
        return self.backbone_output, self.action_input


def _clone_validation_batch(batch: Any) -> _GraphValidationBatch:
    def clone_cpu(name: str) -> torch.Tensor | None:
        value = getattr(batch, name, None)
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"Full-iteration CUDA graph validation field {name} must be a tensor."
            )
        return value.detach().cpu().clone()

    return _GraphValidationBatch(
        image_grid_thw=clone_cpu("image_grid_thw"),
        input_ids=clone_cpu("input_ids"),
    )


def _storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), storage._cdata


def _clone_static(
    value: Any,
    tensor_memo: dict[int, torch.Tensor] | None = None,
    storage_memo: dict[tuple[str, int], torch.UntypedStorage] | None = None,
) -> Any:
    if tensor_memo is None:
        tensor_memo = {}
    if storage_memo is None:
        storage_memo = {}
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise RuntimeError(
                f"Full-iteration CUDA graph static inputs require strided tensors, got {value.layout}."
            )
        existing = tensor_memo.get(id(value))
        if existing is not None:
            return existing
        source_storage = value.untyped_storage()
        storage_key = _storage_key(value)
        static_storage = storage_memo.get(storage_key)
        if static_storage is None:
            storage_owner = torch.empty(
                source_storage.nbytes(),
                dtype=torch.uint8,
                device=value.device,
            )
            static_storage = storage_owner.untyped_storage()
            storage_memo[storage_key] = static_storage
        cloned = torch.empty(0, dtype=value.dtype, device=value.device)
        cloned.set_(
            static_storage,
            value.storage_offset(),
            value.shape,
            value.stride(),
        )
        cloned.copy_(value)
        tensor_memo[id(value)] = cloned
        return cloned
    if isinstance(value, BatchFeature):
        return BatchFeature(
            data={
                key: _clone_static(item, tensor_memo, storage_memo)
                for key, item in value.items()
            }
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        kwargs = {
            field.name: _clone_static(
                getattr(value, field.name),
                tensor_memo,
                storage_memo,
            )
            for field in dataclasses.fields(value)
        }
        return value.__class__(**kwargs)
    if isinstance(value, list):
        return [_clone_static(item, tensor_memo, storage_memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_static(item, tensor_memo, storage_memo) for item in value)
    if isinstance(value, dict):
        return {
            key: _clone_static(item, tensor_memo, storage_memo)
            for key, item in value.items()
        }
    return value


@dataclasses.dataclass
class _CopyAliasState:
    src_to_dst_objects: dict[int, int] = dataclasses.field(default_factory=dict)
    dst_to_src_objects: dict[int, int] = dataclasses.field(default_factory=dict)
    src_to_dst_storages: dict[tuple[str, int], tuple[str, int]] = dataclasses.field(
        default_factory=dict
    )
    dst_to_src_storages: dict[tuple[str, int], tuple[str, int]] = dataclasses.field(
        default_factory=dict
    )

    @staticmethod
    def _bind(mapping: dict[Any, Any], key: Any, value: Any, path: str, kind: str) -> None:
        existing = mapping.setdefault(key, value)
        if existing != value:
            raise RuntimeError(
                f"Full-iteration CUDA graph {kind} alias contract changed at {path}."
            )

    def validate(self, dst: torch.Tensor, src: torch.Tensor, path: str) -> None:
        """Assert the source/destination alias contract is stable across steps."""
        self._bind(self.src_to_dst_objects, id(src), id(dst), path, "tensor-object")
        self._bind(self.dst_to_src_objects, id(dst), id(src), path, "tensor-object")
        src_storage = _storage_key(src)
        dst_storage = _storage_key(dst)
        self._bind(
            self.src_to_dst_storages,
            src_storage,
            dst_storage,
            path,
            "storage",
        )
        self._bind(
            self.dst_to_src_storages,
            dst_storage,
            src_storage,
            path,
            "storage",
        )


def _copy_static(
    dst: Any,
    src: Any,
    path: str = "batch",
    alias_state: _CopyAliasState | None = None,
) -> None:
    if alias_state is None:
        alias_state = _CopyAliasState()
    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
        expected = (
            dst.shape,
            dst.stride(),
            dst.storage_offset(),
            dst.dtype,
        )
        actual = (
            src.shape,
            src.stride(),
            src.storage_offset(),
            src.dtype,
        )
        if expected != actual:
            raise RuntimeError(
                f"Full-iteration CUDA graph tensor signature changed at {path}: "
                f"expected shape={tuple(dst.shape)} stride={dst.stride()} "
                f"storage_offset={dst.storage_offset()} "
                f"dtype={dst.dtype}; got shape={tuple(src.shape)} "
                f"stride={src.stride()} storage_offset={src.storage_offset()} "
                f"dtype={src.dtype} device={src.device}"
            )
        alias_state.validate(dst, src, path)
        dst.copy_(src, non_blocking=src.device.type == "cpu")
        return
    if isinstance(dst, BatchFeature) and isinstance(src, BatchFeature):
        if dst.keys() != src.keys():
            raise RuntimeError(
                f"Full-iteration CUDA graph keys changed at {path}: "
                f"{sorted(dst.keys())} != {sorted(src.keys())}"
            )
        for key in dst:
            _copy_static(dst[key], src[key], f"{path}.{key}", alias_state)
        return
    if dataclasses.is_dataclass(dst) and dataclasses.is_dataclass(src):
        if type(dst) is not type(src):
            raise RuntimeError(
                f"Full-iteration CUDA graph type changed at {path}: "
                f"{type(dst).__name__} != {type(src).__name__}"
            )
        for field in dataclasses.fields(dst):
            _copy_static(
                getattr(dst, field.name),
                getattr(src, field.name),
                f"{path}.{field.name}",
                alias_state,
            )
        return
    if isinstance(dst, (list, tuple)) and isinstance(src, type(dst)):
        if len(dst) != len(src):
            raise RuntimeError(
                f"Full-iteration CUDA graph length changed at {path}: {len(dst)} != {len(src)}"
            )
        for index, (dst_item, src_item) in enumerate(zip(dst, src)):
            _copy_static(dst_item, src_item, f"{path}[{index}]", alias_state)
        return
    if isinstance(dst, dict) and isinstance(src, dict):
        if dst.keys() != src.keys():
            raise RuntimeError(
                f"Full-iteration CUDA graph keys changed at {path}: "
                f"{sorted(dst.keys())} != {sorted(src.keys())}"
            )
        for key in dst:
            _copy_static(dst[key], src[key], f"{path}.{key}", alias_state)
        return
    if type(dst) is not type(src) or dst != src:
        raise RuntimeError(
            f"Full-iteration CUDA graph metadata changed at {path}: {dst!r} != {src!r}"
        )


class GrootN1d7FullIterationCudaGraphRunner:
    """Capture one complete GR00T-N1.7 optimizer iteration and replay it."""

    def __init__(
        self,
        trainer,
        graph_stream: torch.cuda.Stream,
    ) -> None:
        self.trainer = trainer
        self.training_args = trainer.training_args
        self.ctx = trainer.ctx
        self.graph_stream = graph_stream
        self.raw_model = unwrap_model(trainer.model)
        self.warmup_steps = int(self.training_args.cuda_graph_warmup_steps)
        self.warmup_count = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_batch: Any = None
        self.validation_batch: _GraphValidationBatch | None = None
        self.outputs: _GraphOutputs | None = None
        self.time_buffer: torch.Tensor | None = None
        self.lr_buffers: list[torch.Tensor] | None = None
        self.replay_count = 0
        self._optimizer_validated = False
        self._noop_ddp_logger = _NoopDdpLogger()
        self._copy_stream: torch.cuda.Stream | None = None
        self._copy_event: torch.cuda.Event | None = None
        # Beta timestep sampling is intentionally kept on the CPU so that the
        # graph replay consumes the same RNG stream as eager.  The old path
        # copied that tensor synchronously from inside Forward, which exposed
        # the preceding graph work to the host.  A dedicated stream/event lets
        # the tiny H2D copy proceed independently of input staging.
        self._time_stream: torch.cuda.Stream | None = None
        self._time_event: torch.cuda.Event | None = None
        self._time_host_buffer: torch.Tensor | None = None
        self._time_prefetch_buffer: torch.Tensor | None = None
        self._time_prefetch_host_buffers: list[torch.Tensor] = []
        self._time_prefetch_host_index = 0
        self._time_prefetch_pending = False
        # These paths were validated for GR00T-N1.7 and are part of its
        # production Graph contract. Keep them model-owned and deterministic;
        # campaign shell variables cannot silently change graph semantics.
        self._input_prefetch_enabled = True
        self._time_prefetch_enabled = False
        self._fused_optimizer_grad_clip = True
        self._direct_grad_write = True
        self._backbone_pipeline_enabled = True
        self._prefetched_cpu_batch: Any = None
        self._prefetched_gpu_batch: Any = None
        self._prefetched_action_input: BatchFeature | None = None
        self._backbone_graph: torch.cuda.CUDAGraph | None = None
        self._backbone_stream: torch.cuda.Stream | None = None
        self._backbone_input_event: torch.cuda.Event | None = None
        self._backbone_ready_event: torch.cuda.Event | None = None
        self._backbone_progress_event: torch.cuda.Event | None = None
        self._backbone_progress_event_native_external = False
        self._buffer_sync_event: torch.cuda.Event | None = None
        self._backbone_static_input: BatchFeature | None = None
        self._backbone_output: BatchFeature | None = None
        self._backbone_pending = False
        self._saved_ddp_broadcast_buffers: bool | None = None
        self._backbone_progress_layer = 8
        self._validate_configuration()
        if self._input_prefetch_enabled and self.ctx.is_main:
            logger.info(
                "Full-iteration input prefetch enabled: overlap next CPU/H2D batch "
                "with the current graph replay."
            )
        if self._time_prefetch_enabled and self.ctx.is_main:
            logger.info(
                "Full-iteration timestep prefetch enabled: preserve CPU RNG order while "
                "overlapping next Beta sample staging with the current graph replay."
            )
        if self._fused_optimizer_grad_clip and self.ctx.is_main:
            logger.info(
                "Full-iteration fused optimizer gradient clipping enabled."
            )
        if self._direct_grad_write and self.ctx.is_main:
            logger.info(
                "Full-iteration direct gradient write enabled: capture leaf gradients "
                "as first-write static buffers without per-replay zero/add passes."
            )
        if self._backbone_pipeline_enabled and self.ctx.is_main:
            logger.info(
                "Full-iteration frozen-backbone pipeline enabled: overlap next Qwen "
                "graph with the current action-head train graph; progress_layer=%d.",
                self._backbone_progress_layer,
            )

    @classmethod
    def is_enabled(cls, trainer) -> bool:
        """Return whether the trainer requests the full-iteration CUDA graph."""
        args = trainer.training_args
        return (
            torch.cuda.is_available()
            and args.cuda_graph_impl == "local"
            and args.cuda_graph_scope == "full_iteration"
        )

    def _validate_configuration(self) -> None:
        if self.training_args.gradient_accumulation_steps != 1:
            raise RuntimeError(
                "GR00T-N1.7 full-iteration CUDA graph currently requires "
                "--gradient-accumulation-steps=1."
            )
        if self.ctx.is_distributed and self.ctx.world_size > 1:
            if not hasattr(self.trainer.model, "reducer"):
                raise RuntimeError("Full-iteration CUDA graph requires the standard DDP wrapper.")
            if not self.training_args.cuda_graph_ddp_sync_in_graph:
                raise RuntimeError(
                    "Full-iteration CUDA graph requires --cuda-graph-ddp-sync-in-graph; "
                    "manual post-backward all-reduce is forbidden."
                )
        if self._backbone_pipeline_enabled:
            if not self._input_prefetch_enabled:
                raise RuntimeError(
                    "Frozen-backbone graph pipeline requires input prefetch."
                )
            if any(parameter.requires_grad for parameter in self._backbone().parameters()):
                raise RuntimeError(
                    "Frozen-backbone graph pipeline requires every Qwen backbone "
                    "parameter to have requires_grad=False."
                )
        if self._fused_optimizer_grad_clip:
            if float(self.training_args.clip_grad) <= 0:
                raise RuntimeError(
                    "Fused optimizer gradient clipping requires --clip-grad > 0."
                )

    def _validate_optimizer(self) -> None:
        if self._optimizer_validated:
            return
        if not getattr(self.trainer.optimizer, "capturable", False):
            raise RuntimeError(
                f"Optimizer {type(self.trainer.optimizer).__name__} is not capturable. "
                "Use TEFusedAdamW or TorchFusedAdamW with full-iteration graph mode."
            )
        if (
            self._fused_optimizer_grad_clip
            and not hasattr(self.trainer.optimizer, "set_grad_scale")
        ):
            raise RuntimeError(
                "Fused optimizer gradient clipping requires precision-compatible "
                "TEFusedAdamW."
            )
        self._optimizer_validated = True

    def _action_head(self):
        model = getattr(self.raw_model, "model", None)
        action_head = getattr(model, "action_head", None)
        if action_head is None:
            raise RuntimeError("GR00T-N1.7 action head was not found for graph RNG inputs.")
        return action_head

    def _backbone(self):
        model = getattr(self.raw_model, "model", None)
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            raise RuntimeError("GR00T-N1.7 backbone was not found for full-iteration capture.")
        return backbone

    def _groot_model(self):
        model = getattr(self.raw_model, "model", None)
        if model is None:
            raise RuntimeError("GR00T-N1.7 core model was not found for full-iteration capture.")
        return model

    @staticmethod
    def _module_parameter_dtype(module: torch.nn.Module) -> torch.dtype | None:
        for parameter in module.parameters():
            if torch.is_floating_point(parameter):
                return parameter.dtype
        return None

    def _convert_pipeline_value(
        self,
        value: Any,
        dtype: torch.dtype | None,
    ) -> Any:
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value) and dtype is not None:
                return value.to(device=self.ctx.device, dtype=dtype)
            return value.to(device=self.ctx.device)
        if isinstance(value, dict):
            return {
                key: self._convert_pipeline_value(item, dtype)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._convert_pipeline_value(item, dtype) for item in value]
        if isinstance(value, tuple):
            return tuple(self._convert_pipeline_value(item, dtype) for item in value)
        return value

    def _prepare_pipeline_inputs(
        self,
        batch: Any,
    ) -> tuple[BatchFeature, BatchFeature]:
        try:
            inputs = batch.to_model_inputs()
        except AttributeError as exc:
            raise TypeError(
                "Frozen-backbone graph pipeline expects a batch with to_model_inputs(), "
                f"got {type(batch).__name__}."
            ) from exc

        model = self._groot_model()
        backbone_dtype = self._module_parameter_dtype(model.backbone)
        action_dtype = self._module_parameter_dtype(model.action_head)
        backbone_keys = (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "position_ids",
            "mm_token_type_ids",
        )
        action_keys = ("state", "action", "action_mask", "embodiment_id")
        missing_backbone = [key for key in backbone_keys[:4] if key not in inputs]
        missing_action = [key for key in action_keys if key not in inputs]
        if missing_backbone or missing_action:
            raise RuntimeError(
                "Frozen-backbone graph pipeline input keys are incomplete: "
                f"backbone_missing={missing_backbone}, action_missing={missing_action}."
            )
        backbone_input = BatchFeature(
            data={
                key: self._convert_pipeline_value(inputs[key], backbone_dtype)
                for key in backbone_keys
                if key in inputs and inputs[key] is not None
            }
        )
        action_input = BatchFeature(
            data={
                key: self._convert_pipeline_value(inputs[key], action_dtype)
                for key in action_keys
            }
        )
        return backbone_input, action_input

    def _prepare_pipeline_graph_batch(self) -> None:
        if self._backbone_static_input is None:
            raise RuntimeError("Frozen-backbone graph static input is missing.")
        backbone = self._backbone()
        prepare = getattr(backbone, "prepare_cuda_graph_batch", None)
        if callable(prepare):
            prepare(self._backbone_static_input)
        prepare_action_head = getattr(self._action_head(), "prepare_cuda_graph_batch", None)
        if callable(prepare_action_head):
            prepare_action_head(
                self._backbone_static_input,
                backbone.model.config.image_token_id,
            )

    def _train_autocast_context(self):
        if self.trainer._cfg_bool("disable_train_autocast", False):
            return nullcontext()
        dtype = getattr(self.trainer, "_compute_dtype", None)
        if dtype is None:
            dtype = resolve_dtype(self.training_args.dtype)
            self.trainer._compute_dtype = dtype
        return torch.autocast("cuda", dtype=dtype)

    def _ensure_backbone_pipeline_resources(self) -> None:
        if self._backbone_stream is None:
            self._backbone_stream = torch.cuda.Stream(device=self.ctx.device)
        if self._backbone_input_event is None:
            self._backbone_input_event = torch.cuda.Event()
        if self._backbone_ready_event is None:
            self._backbone_ready_event = torch.cuda.Event()
        if (
            self._backbone_progress_layer >= 0
            and self._backbone_progress_event is None
        ):
            try:
                self._backbone_progress_event = torch.cuda.Event(external=True)
                self._backbone_progress_event_native_external = True
            except TypeError:
                # Older PyTorch builds expose no external= argument. The
                # CUDA runtime flag is applied when the event is recorded.
                self._backbone_progress_event = torch.cuda.Event()
                self._backbone_progress_event_native_external = False
            self._backbone_progress_event.record(self._backbone_stream)
        if self._buffer_sync_event is None:
            self._buffer_sync_event = torch.cuda.Event()

    def _register_backbone_progress_hook(self):
        if self._backbone_progress_layer < 0:
            return None
        self._ensure_backbone_pipeline_resources()
        layers = getattr(self._backbone().language_model, "layers", None)
        if layers is None:
            raise RuntimeError("Qwen language layers were not found for pipeline progress event.")
        if self._backbone_progress_layer >= len(layers):
            raise RuntimeError(
                "Frozen-backbone progress layer is out of range: "
                f"{self._backbone_progress_layer} >= {len(layers)}."
            )
        assert self._backbone_progress_event is not None

        def record_progress(_module, _inputs, output):
            assert self._backbone_stream is not None
            if self._backbone_progress_event_native_external:
                self._backbone_progress_event.record(self._backbone_stream)
            else:
                _record_external_cuda_event(
                    self._backbone_progress_event,
                    self._backbone_stream,
                )
            return output

        return layers[self._backbone_progress_layer].register_forward_hook(record_progress)

    def _wait_pipeline_progress_before_finish(self) -> None:
        if not self._backbone_pending or self._backbone_progress_event is None:
            return
        if self._backbone_stream is None:
            return
        torch.cuda.current_stream(self.ctx.device).wait_event(self._backbone_progress_event)

    def _activate_pipeline_ddp_buffer_sync(self) -> None:
        if self._saved_ddp_broadcast_buffers is not None:
            return
        ddp_model = self.trainer.model
        broadcast_buffers = bool(getattr(ddp_model, "broadcast_buffers", False))
        self._saved_ddp_broadcast_buffers = broadcast_buffers
        if broadcast_buffers:
            # The broadcast remains once per step and ahead of both graphs. It
            # cannot stay inside the train graph because the next frozen
            # backbone graph reads these buffers concurrently.
            ddp_model.broadcast_buffers = False

    def _sync_pipeline_ddp_buffers(self) -> None:
        self._ensure_backbone_pipeline_resources()
        self._activate_pipeline_ddp_buffer_sync()
        default_stream = torch.cuda.current_stream(self.ctx.device)
        self.graph_stream.wait_stream(default_stream)
        with torch.cuda.stream(self.graph_stream):
            if self._saved_ddp_broadcast_buffers:
                sync_buffers = getattr(self.trainer.model, "_sync_buffers", None)
                if not callable(sync_buffers):
                    raise RuntimeError(
                        "Frozen-backbone graph pipeline could not find DDP._sync_buffers()."
                    )
                sync_buffers()
            assert self._buffer_sync_event is not None
            self._buffer_sync_event.record(self.graph_stream)

    def _static_actions(self) -> torch.Tensor | None:
        actions = getattr(self.static_batch, "actions", None)
        if actions is not None:
            return actions
        if isinstance(self.static_batch, _ActionGraphBatch):
            return self.static_batch.action_input.get("action")
        return None

    def _prepare_graph_batch(self, batch: Any) -> None:
        backbone = self._backbone()
        prepare = getattr(backbone, "prepare_cuda_graph_batch", None)
        if callable(prepare):
            prepare(batch)
        prepare_action_head = getattr(self._action_head(), "prepare_cuda_graph_batch", None)
        if callable(prepare_action_head):
            prepare_action_head(batch, backbone.model.config.image_token_id)

    def _validate_graph_batch(self, batch: Any) -> None:
        validate = getattr(self._backbone(), "validate_cuda_graph_batch", None)
        if callable(validate):
            if self.validation_batch is None:
                raise RuntimeError("Full-iteration CUDA graph validation batch is missing.")
            validate(self.validation_batch, batch)
        validate_action = getattr(
            self._action_head(),
            "validate_cuda_graph_action_batch",
            None,
        )
        if callable(validate_action):
            validate_action(batch)

    def _ensure_copy_resources(self) -> None:
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.ctx.device)
        if self._copy_event is None:
            self._copy_event = torch.cuda.Event()

    def _ensure_time_resources(self) -> None:
        if self._time_stream is None:
            self._time_stream = torch.cuda.Stream(device=self.ctx.device)
        if self._time_event is None:
            self._time_event = torch.cuda.Event()

    def _fetch_cpu_batch(self) -> Any:
        cpu_batch = self.trainer._fetch_batch_cpu("vla")
        prepare_host = getattr(self._backbone(), "prepare_host_position_metadata", None)
        if callable(prepare_host):
            prepare_host(cpu_batch)
        return cpu_batch

    def _prefetch_next_batch(self) -> None:
        if not self._input_prefetch_enabled or self.graph is None:
            return
        completed_steps = self.warmup_count + self.replay_count
        if completed_steps >= int(self.training_args.train_iters):
            return
        if self._prefetched_gpu_batch is not None:
            raise RuntimeError("Full-iteration input prefetch staging batch was not consumed.")

        with self.trainer._stage_timers("batch-generator"):
            cpu_batch = self._fetch_cpu_batch()
            self._validate_graph_batch(cpu_batch)
            self._ensure_copy_resources()
            assert self._copy_stream is not None
            with torch.cuda.stream(self._copy_stream):
                gpu_batch = self.trainer._move_batch_to_device(cpu_batch)

        self._prefetched_cpu_batch = cpu_batch
        self._prefetched_gpu_batch = gpu_batch

    def _consume_prefetched_batch(self) -> Any:
        if self._backbone_pipeline_enabled:
            return self._consume_prefetched_pipeline_batch()
        if self.static_batch is None:
            raise RuntimeError("Full-iteration CUDA graph static batch is missing.")
        if self._prefetched_gpu_batch is None:
            raise RuntimeError("Full-iteration input prefetch did not stage the next batch.")
        self._ensure_copy_resources()
        assert self._copy_stream is not None
        assert self._copy_event is not None
        default_stream = torch.cuda.current_stream(self.ctx.device)
        with torch.cuda.stream(self._copy_stream):
            _copy_static(self.static_batch, self._prefetched_gpu_batch)
            self._copy_event.record(self._copy_stream)
        default_stream.wait_event(self._copy_event)
        self._prefetched_gpu_batch = None
        self._prefetched_cpu_batch = None
        return self.static_batch

    def _launch_prefetched_backbone(self) -> None:
        if not self._backbone_pipeline_enabled or self._prefetched_gpu_batch is None:
            return
        if self._backbone_graph is None or self._backbone_static_input is None:
            raise RuntimeError("Frozen-backbone graph was not captured before prefetch launch.")
        if self._backbone_pending:
            raise RuntimeError("Frozen-backbone graph output was not consumed before replay.")

        self._ensure_copy_resources()
        self._ensure_backbone_pipeline_resources()
        assert self._copy_stream is not None
        assert self._backbone_stream is not None
        assert self._backbone_input_event is not None
        assert self._backbone_ready_event is not None
        assert self._buffer_sync_event is not None
        with torch.cuda.stream(self._copy_stream):
            backbone_input, action_input = self._prepare_pipeline_inputs(
                self._prefetched_gpu_batch
            )
            _copy_static(self._backbone_static_input, backbone_input)
            self._backbone_input_event.record(self._copy_stream)
        self._prefetched_action_input = action_input

        self._backbone_stream.wait_event(self._backbone_input_event)
        self._backbone_stream.wait_event(self._buffer_sync_event)
        with torch.cuda.stream(self._backbone_stream):
            self._backbone_graph.replay()
            self._backbone_ready_event.record(self._backbone_stream)
        self._backbone_pending = True

    def _consume_prefetched_pipeline_batch(self) -> _ActionGraphBatch:
        if not isinstance(self.static_batch, _ActionGraphBatch):
            raise RuntimeError("Frozen-backbone train graph static batch is missing.")
        if not self._backbone_pending or self._prefetched_action_input is None:
            raise RuntimeError("Frozen-backbone pipeline did not stage the next batch.")
        if self._backbone_output is None:
            raise RuntimeError("Frozen-backbone pipeline output buffer is missing.")
        self._ensure_copy_resources()
        self._ensure_backbone_pipeline_resources()
        assert self._copy_stream is not None
        assert self._copy_event is not None
        assert self._backbone_ready_event is not None
        default_stream = torch.cuda.current_stream(self.ctx.device)
        self._copy_stream.wait_stream(default_stream)
        self._copy_stream.wait_event(self._backbone_ready_event)
        with torch.cuda.stream(self._copy_stream):
            _copy_static(self.static_batch.backbone_output, self._backbone_output)
            _copy_static(self.static_batch.action_input, self._prefetched_action_input)
            self._copy_event.record(self._copy_stream)
        default_stream.wait_event(self._copy_event)

        self._prefetched_action_input = None
        self._prefetched_gpu_batch = None
        self._prefetched_cpu_batch = None
        self._backbone_pending = False
        return self.static_batch

    def _fetch_batch(self) -> Any:
        with self.trainer._stage_timers("batch-generator"):
            if self.graph is not None and self._input_prefetch_enabled:
                batch = self._consume_prefetched_batch()
            else:
                cpu_batch = self._fetch_cpu_batch()
                default_stream = torch.cuda.current_stream(self.ctx.device)
                self._ensure_copy_resources()
                assert self._copy_stream is not None
                assert self._copy_event is not None

                if self.graph is not None:
                    if self.static_batch is None:
                        raise RuntimeError("Full-iteration CUDA graph static batch is missing.")
                    self._validate_graph_batch(cpu_batch)
                    self._copy_stream.wait_stream(default_stream)
                    with torch.cuda.stream(self._copy_stream):
                        _copy_static(self.static_batch, cpu_batch)
                        self._copy_event.record(self._copy_stream)
                    default_stream.wait_event(self._copy_event)
                    batch = self.static_batch
                else:
                    if self.warmup_count >= self.warmup_steps:
                        self.validation_batch = _clone_validation_batch(cpu_batch)
                    self._copy_stream.wait_stream(default_stream)
                    with torch.cuda.stream(self._copy_stream):
                        batch = self.trainer._move_batch_to_device(cpu_batch)
                        self._copy_event.record(self._copy_stream)
                    default_stream.wait_event(self._copy_event)
        self.trainer._on_after_train_batch_fetch(batch, 0)
        self.trainer._prepare_model_for_train_step()
        return batch

    def _ensure_lr_buffers(self) -> None:
        if self.lr_buffers is not None:
            return
        buffers = []
        for index, group in enumerate(self.trainer.optimizer.param_groups):
            value = group["lr"]
            if not isinstance(value, torch.Tensor) or not value.is_cuda:
                raise RuntimeError(
                    f"Capturable optimizer group {index} LR must be a CUDA tensor, got {value!r}."
                )
            buffers.append(value)
        self.lr_buffers = buffers

    @torch.no_grad()
    def _advance_scheduler(self) -> None:
        self._ensure_lr_buffers()
        assert self.lr_buffers is not None
        for group, buffer in zip(self.trainer.optimizer.param_groups, self.lr_buffers):
            group["lr"] = buffer
        self.trainer.lr_scheduler.step()
        for group, buffer in zip(self.trainer.optimizer.param_groups, self.lr_buffers):
            updated = group["lr"]
            if updated is not buffer:
                buffer.copy_(updated if isinstance(updated, torch.Tensor) else float(updated))
                group["lr"] = buffer

    def _fill_time_buffer(self) -> None:
        action_head = self._action_head()
        if self.static_batch is None:
            return
        actions = self._static_actions()
        if actions is None:
            raise RuntimeError("Static GR00T-N1.7 batch has no actions tensor.")
        if self.time_buffer is None:
            self.time_buffer = torch.empty(
                (actions.shape[0],),
                device=actions.device,
                dtype=actions.dtype,
            )
        self._ensure_time_resources()
        assert self._time_stream is not None
        assert self._time_event is not None
        if (
            self._time_host_buffer is None
            or self._time_host_buffer.shape != (actions.shape[0],)
            or self._time_host_buffer.dtype != actions.dtype
        ):
            self._time_host_buffer = torch.empty(
                (actions.shape[0],),
                dtype=actions.dtype,
                pin_memory=True,
            )

        # Generate the identical CPU Beta samples as the reference path, but
        # enqueue only the pinned-host -> device copy.
        sample_cpu = action_head.sample_time(
            actions.shape[0],
            device=torch.device("cpu"),
            dtype=actions.dtype,
        )
        self._time_host_buffer.copy_(sample_cpu)
        with torch.cuda.stream(self._time_stream):
            self.time_buffer.copy_(self._time_host_buffer, non_blocking=True)
            self._time_event.record(self._time_stream)
        action_head._split_time_buf = self.time_buffer

    def _prefetch_next_time_buffer(self) -> None:
        if not self._time_prefetch_enabled or self.graph is None:
            return
        completed_steps = self.warmup_count + self.replay_count
        if completed_steps >= int(self.training_args.train_iters):
            return
        if self._time_prefetch_pending:
            raise RuntimeError("Full-iteration timestep prefetch was not consumed.")
        if self.static_batch is None:
            raise RuntimeError("Full-iteration CUDA graph static batch is missing.")
        actions = self._static_actions()
        if actions is None:
            raise RuntimeError("Static GR00T-N1.7 batch has no actions tensor.")

        if self._time_prefetch_buffer is None:
            self._time_prefetch_buffer = torch.empty(
                (actions.shape[0],),
                device=actions.device,
                dtype=actions.dtype,
            )
        expected = (actions.shape[0],)
        if not self._time_prefetch_host_buffers:
            self._time_prefetch_host_buffers = [
                torch.empty(expected, dtype=actions.dtype, pin_memory=True)
                for _ in range(2)
            ]
        host_buffer = self._time_prefetch_host_buffers[self._time_prefetch_host_index]
        sample_cpu = self._action_head().sample_time(
            actions.shape[0],
            device=torch.device("cpu"),
            dtype=actions.dtype,
        )
        host_buffer.copy_(sample_cpu)
        self._ensure_time_resources()
        assert self._time_stream is not None
        with torch.cuda.stream(self._time_stream):
            self._time_prefetch_buffer.copy_(host_buffer, non_blocking=True)
        self._time_prefetch_host_index = (self._time_prefetch_host_index + 1) % len(
            self._time_prefetch_host_buffers
        )
        self._time_prefetch_pending = True

    def _consume_prefetched_time_buffer(self) -> None:
        if not self._time_prefetch_pending:
            raise RuntimeError("Full-iteration timestep prefetch did not stage the next sample.")
        assert self.time_buffer is not None
        assert self._time_prefetch_buffer is not None
        self._ensure_time_resources()
        assert self._time_stream is not None
        assert self._time_event is not None
        with torch.cuda.stream(self._time_stream):
            self.time_buffer.copy_(self._time_prefetch_buffer)
            self._time_event.record(self._time_stream)
        self._action_head()._split_time_buf = self.time_buffer
        self._time_prefetch_pending = False

    def _wait_time_buffer(self) -> None:
        if self._time_event is not None:
            self.graph_stream.wait_event(self._time_event)

    def _clear_time_buffer(self) -> None:
        self._action_head()._split_time_buf = None

    def _zero_grad(self, *, set_to_none: bool) -> None:
        try:
            self.trainer.optimizer.zero_grad(set_to_none=set_to_none)
        except TypeError:
            self.trainer.optimizer.zero_grad()

    def _clean_nan_gradients(self) -> None:
        if not self.training_args.check_for_nan_in_loss_and_grad:
            return
        for parameter in self.raw_model.parameters():
            if parameter.grad is not None:
                torch.nan_to_num(
                    parameter.grad,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                    out=parameter.grad,
                )

    def _iteration_body(self, *, set_to_none: bool) -> _GraphOutputs:
        if _should_zero_grad_before_iteration(
            set_to_none=set_to_none,
            direct_grad_write=self._direct_grad_write,
        ):
            self._zero_grad(set_to_none=set_to_none)
        loss, _log_losses = self.trainer._train_forward(self.static_batch)
        scaled_loss = loss / self.training_args.gradient_accumulation_steps
        finite = torch.isfinite(scaled_loss)
        below_threshold = scaled_loss <= float(self.training_args.loss_spike_threshold)
        valid = finite & below_threshold
        safe_loss = torch.where(valid, scaled_loss, torch.zeros_like(scaled_loss))
        safe_loss.backward()
        self._clean_nan_gradients()

        params = [parameter for parameter in self.raw_model.parameters() if parameter.grad is not None]
        if not params:
            raise RuntimeError("Full-iteration CUDA graph found no gradients after backward.")
        max_norm = float(self.training_args.clip_grad)
        if self._fused_optimizer_grad_clip:
            grad_norm, grad_scale = _compute_grad_norm_and_clip_scale(
                [parameter.grad for parameter in params],
                max_norm,
            )
            self.trainer.optimizer.set_grad_scale(grad_scale)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params,
                max_norm if max_norm > 0 else float("inf"),
                error_if_nonfinite=False,
            )
        self.trainer.optimizer.step()
        return _GraphOutputs(
            action_loss=loss.detach(),
            grad_norm=grad_norm.detach(),
            nan_flag=(~finite).to(dtype=torch.int32),
            spike_flag=(~valid).to(dtype=torch.int32),
        )

    def _wait_default_stream(self) -> None:
        torch.cuda.current_stream(self.ctx.device).wait_stream(self.graph_stream)

    def _run_eager_warmup(self, batch: Any) -> tuple[dict[str, torch.Tensor], float]:
        self.static_batch = batch
        default_stream = torch.cuda.current_stream(self.ctx.device)
        self.graph_stream.wait_stream(default_stream)
        with torch.cuda.stream(self.graph_stream):
            self._clear_time_buffer()
            outputs = self._iteration_body(set_to_none=True)
        self._wait_default_stream()
        nan_flag = int(outputs.nan_flag.item())
        spike_flag = int(outputs.spike_flag.item())
        self.trainer.nan_iterations += nan_flag
        self.trainer.skipped_iterations += spike_flag
        self._advance_scheduler()
        self.warmup_count += 1
        if self.ctx.is_main and self.warmup_count == self.warmup_steps:
            logger.info(
                "Full-iteration CUDA graph warmup complete: steps=%d",
                self.warmup_steps,
            )
        return {"action_loss": outputs.action_loss}, float(outputs.grad_norm)

    def _capture_backbone_pipeline(
        self,
        batch: Any,
    ) -> tuple[dict[str, torch.Tensor], float]:
        backbone_input, action_input = self._prepare_pipeline_inputs(batch)
        self._backbone_static_input = _clone_static(backbone_input)
        self._prepare_pipeline_graph_batch()
        self._ensure_backbone_pipeline_resources()
        self._sync_pipeline_ddp_buffers()
        assert self._backbone_stream is not None
        assert self._buffer_sync_event is not None
        self._backbone_stream.wait_event(self._buffer_sync_event)

        # Capture the frozen Qwen graph first. Its output remains in a private
        # staging allocation, so later replays can write it while the train
        # graph reads a separate static copy.
        torch.cuda.synchronize(self.ctx.device)
        self.ctx.barrier()
        backbone_rng_state = torch.cuda.get_rng_state(self.ctx.device)
        backbone_graph = torch.cuda.CUDAGraph()
        progress_hook = self._register_backbone_progress_hook()
        try:
            with torch.cuda.graph(
                backbone_graph,
                stream=self._backbone_stream,
                capture_error_mode="thread_local",
            ):
                with torch.no_grad(), self._train_autocast_context():
                    backbone_output = self._backbone()(self._backbone_static_input)
        finally:
            if progress_hook is not None:
                progress_hook.remove()
        self._backbone_graph = backbone_graph
        self._backbone_output = backbone_output

        # The capture executes on the dedicated backbone stream. The action
        # graph owns a separate copy of these outputs, so establish the
        # producer/consumer dependency before cloning on the current stream.
        torch.cuda.current_stream(self.ctx.device).wait_stream(self._backbone_stream)
        torch.cuda.synchronize(self.ctx.device)
        torch.cuda.set_rng_state(backbone_rng_state, self.ctx.device)

        self.static_batch = _ActionGraphBatch(
            backbone_output=_clone_static(backbone_output),
            action_input=_clone_static(action_input),
        )
        self._fill_time_buffer()
        if not any(parameter.grad is not None for parameter in self.raw_model.parameters()):
            raise RuntimeError(
                "Full-iteration capture requires materialized stable gradient buffers after warmup."
            )
        if self._direct_grad_write:
            self._zero_grad(set_to_none=True)

        torch.cuda.synchronize(self.ctx.device)
        self.ctx.barrier()
        graph = torch.cuda.CUDAGraph()
        saved_logger = None
        ddp_model = self.trainer.model
        if hasattr(ddp_model, "reducer"):
            saved_logger = ddp_model.logger
            ddp_model.logger = self._noop_ddp_logger
        try:
            self.graph_stream.wait_stream(torch.cuda.current_stream(self.ctx.device))
            with torch.cuda.graph(
                graph,
                stream=self.graph_stream,
                capture_error_mode="thread_local",
            ):
                outputs = self._iteration_body(set_to_none=False)
        finally:
            if saved_logger is not None:
                ddp_model.logger = saved_logger

        torch.cuda.synchronize(self.ctx.device)
        self.ctx.barrier()
        self.graph = graph
        self.outputs = outputs
        if self._direct_grad_write:
            missing_gradients = [
                name
                for name, parameter in self.raw_model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            if missing_gradients:
                raise RuntimeError(
                    "Direct gradient write did not materialize every trainable gradient "
                    f"during capture; first missing parameters: {missing_gradients[:5]}."
                )

        # CUDA capture executes the train body once. Match the ordinary full
        # graph path's immediate replay. The ordinary graph also reruns Qwen
        # for that replay, so refresh the train-static backbone output before
        # the second optimizer update instead of reusing capture-time output.
        self._sync_pipeline_ddp_buffers()
        assert self._backbone_graph is not None
        assert self._backbone_stream is not None
        assert self._backbone_ready_event is not None
        assert self._buffer_sync_event is not None
        self._backbone_stream.wait_event(self._buffer_sync_event)
        with torch.cuda.stream(self._backbone_stream):
            self._backbone_graph.replay()
            self._backbone_ready_event.record(self._backbone_stream)

        self._ensure_copy_resources()
        assert self._copy_stream is not None
        assert self._copy_event is not None
        self._copy_stream.wait_event(self._backbone_ready_event)
        with torch.cuda.stream(self._copy_stream):
            assert self._backbone_output is not None
            _copy_static(self.static_batch.backbone_output, self._backbone_output)
            self._copy_event.record(self._copy_stream)
        self.graph_stream.wait_event(self._copy_event)
        with torch.cuda.stream(self.graph_stream):
            self.graph.replay()
        self.replay_count += 1
        self._wait_default_stream()
        torch.cuda.synchronize(self.ctx.device)

        self._prefetch_next_batch()
        self._launch_prefetched_backbone()
        self._prefetch_next_time_buffer()
        self._wait_pipeline_progress_before_finish()
        return self._finish_step()

    def _capture(self, batch: Any) -> tuple[dict[str, torch.Tensor], float]:
        if self._backbone_pipeline_enabled:
            return self._capture_backbone_pipeline(batch)
        self.static_batch = _clone_static(batch)
        self._prepare_graph_batch(self.static_batch)
        self._fill_time_buffer()
        if not any(parameter.grad is not None for parameter in self.raw_model.parameters()):
            raise RuntimeError(
                "Full-iteration capture requires materialized stable gradient buffers after warmup."
            )
        if self._direct_grad_write:
            self._zero_grad(set_to_none=True)

        # StatefulDataLoader may allocate pinned host buffers on its prefetch
        # thread while this thread captures.  Synchronize ranks for NCCL, then
        # restrict capture errors to operations issued by the capture thread.
        torch.cuda.synchronize(self.ctx.device)
        self.ctx.barrier()

        graph = torch.cuda.CUDAGraph()
        saved_logger = None
        ddp_model = self.trainer.model
        if hasattr(ddp_model, "reducer"):
            saved_logger = ddp_model.logger
            ddp_model.logger = self._noop_ddp_logger
        try:
            self.graph_stream.wait_stream(torch.cuda.current_stream(self.ctx.device))
            with torch.cuda.graph(
                graph,
                stream=self.graph_stream,
                capture_error_mode="thread_local",
            ):
                outputs = self._iteration_body(set_to_none=False)
        finally:
            if saved_logger is not None:
                ddp_model.logger = saved_logger

        torch.cuda.synchronize(self.ctx.device)
        self.ctx.barrier()

        self.graph = graph
        self.outputs = outputs
        if self._direct_grad_write:
            missing_gradients = [
                name
                for name, parameter in self.raw_model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            if missing_gradients:
                raise RuntimeError(
                    "Direct gradient write did not materialize every trainable gradient "
                    f"during capture; first missing parameters: {missing_gradients[:5]}."
                )
        with torch.cuda.stream(self.graph_stream):
            self.graph.replay()
        self.replay_count += 1
        self._wait_default_stream()
        torch.cuda.synchronize(self.ctx.device)
        self._prefetch_next_batch()
        self._prefetch_next_time_buffer()
        return self._finish_step()

    def _replay(self, batch: Any) -> tuple[dict[str, torch.Tensor], float]:
        if batch is not self.static_batch:
            raise RuntimeError("Full-iteration CUDA graph replay did not receive its static batch.")
        if self._time_prefetch_enabled:
            self._consume_prefetched_time_buffer()
        else:
            self._fill_time_buffer()
        assert self.graph is not None
        default_stream = torch.cuda.current_stream(self.ctx.device)
        if self._backbone_pipeline_enabled:
            self._sync_pipeline_ddp_buffers()
            self._launch_prefetched_backbone()
        self.graph_stream.wait_stream(default_stream)
        self._wait_time_buffer()
        with torch.cuda.stream(self.graph_stream):
            self.graph.replay()
        self.replay_count += 1
        self._prefetch_next_batch()
        self._launch_prefetched_backbone()
        self._prefetch_next_time_buffer()
        self._wait_pipeline_progress_before_finish()
        self._wait_default_stream()
        return self._finish_step()

    def _finish_step(self) -> tuple[dict[str, torch.Tensor], float]:
        assert self.outputs is not None
        nan_flag = int(self.outputs.nan_flag.item())
        spike_flag = int(self.outputs.spike_flag.item())
        self.trainer.nan_iterations += nan_flag
        self.trainer.skipped_iterations += spike_flag
        self._advance_scheduler()
        return {"action_loss": self.outputs.action_loss}, float(self.outputs.grad_norm)

    def step(self) -> tuple[dict[str, torch.Tensor], float]:
        """Run one training iteration via eager warmup, capture, or replay."""
        self._validate_optimizer()
        batch = self._fetch_batch()
        if self.warmup_count < self.warmup_steps:
            return self._run_eager_warmup(batch)
        if self.graph is None:
            return self._capture(batch)
        return self._replay(batch)

    def close(self) -> None:
        """Release captured graphs, static buffers, and restore DDP settings."""
        if self.graph is None and self._backbone_graph is None:
            if self._saved_ddp_broadcast_buffers is not None:
                self.trainer.model.broadcast_buffers = self._saved_ddp_broadcast_buffers
                self._saved_ddp_broadcast_buffers = None
            return
        self.ctx.barrier()
        torch.cuda.synchronize(self.ctx.device)
        if self.graph is not None:
            self.graph.reset()
        if self._backbone_graph is not None:
            self._backbone_graph.reset()
        self.graph = None
        self._backbone_graph = None
        self.outputs = None
        self.static_batch = None
        self.validation_batch = None
        self._backbone_static_input = None
        self._backbone_output = None
        self._prefetched_action_input = None
        self._backbone_pending = False
        self.time_buffer = None
        self._copy_event = None
        self._copy_stream = None
        self._time_event = None
        self._time_stream = None
        self._time_host_buffer = None
        self._time_prefetch_buffer = None
        self._time_prefetch_host_buffers = []
        self._time_prefetch_host_index = 0
        self._time_prefetch_pending = False
        self._prefetched_cpu_batch = None
        self._prefetched_gpu_batch = None
        self._backbone_input_event = None
        self._backbone_ready_event = None
        self._backbone_progress_event = None
        self._buffer_sync_event = None
        self._backbone_stream = None
        if self._saved_ddp_broadcast_buffers is not None:
            self.trainer.model.broadcast_buffers = self._saved_ddp_broadcast_buffers
            self._saved_ddp_broadcast_buffers = None
        self._clear_time_buffer()
        self.ctx.barrier()

    @property
    def captured(self) -> bool:
        """Return whether the training graph has been captured."""
        return self.graph is not None
