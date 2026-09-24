# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Full-iteration CUDA graph runner for GR00T-N1.7 training."""

from __future__ import annotations

import dataclasses
import ctypes
import logging
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.distributed as dist
from transformers.feature_extraction_utils import BatchFeature

from loongforge.embodied.distributed.utils import unwrap_model
from loongforge.embodied.model.groot_n1_7.modules.cuda_graph_flash_attention import (
    maybe_install_graph_safe_fa2_patches,
    prime_graph_safe_fa2_buffers,
)
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
    attention_mask: torch.Tensor | None


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
        attention_mask=clone_cpu("attention_mask"),
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
            field.name: (
                None
                if field.name.startswith("_loongforge_host_")
                else _clone_static(
                    getattr(value, field.name),
                    tensor_memo,
                    storage_memo,
                )
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
            if field.name.startswith("_loongforge_host_"):
                # Host-only metadata is refreshed by the model helper before
                # each replay and is intentionally not part of graph inputs.
                continue
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
        self.capture_count = 0
        self.fallback_count = 0
        self._eager_fallback = False
        self._fallback_cpu_batch: Any = None
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
        # Use the same in-place gradient clipping as eager. The optimizer's
        # capturable step consumes those clipped buffers directly, which keeps
        # Graph-on and Graph-off TEFusedAdamW numerically identical.
        self._fused_optimizer_grad_clip = False
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
        self._fp8_enabled = bool(getattr(self.training_args, "fp8", False))
        self._fp8_recipe = None
        self._fp8_group = None
        self._fp8_tensor_pointers: dict[str, int] | None = None
        self._fp8_skip_weight_update_ptr: int | None = None
        # CUDA stream capture records work without executing it.  The first
        # graph replay must therefore consume the timestep sampled immediately
        # before capture instead of sampling a second value.
        self._capture_replay_pending = False
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._configure_graph_attention_backend()
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

    def _configure_graph_attention_backend(self) -> None:
        """Keep eager and graph text attention on the same graph-safe FA2 path.

        The graph-safe padded FA2 adapter owns the fixed-shape unpadding buffers
        and computes ``cu_seqlens`` on device.  Falling back to SDPA here would
        make G0 numerically incomparable with the user-specified eager/B0
        FlashAttention path, so only genuinely non-FlashAttention models retain
        their configured backend.
        """
        try:
            language_model = self._backbone().language_model
            config = language_model.config
        except AttributeError:
            return
        if getattr(config, "_attn_implementation", None) in {
            "flash_attention_2",
            "flash_attention_3",
        }:
            logger.info(
                "CUDA graph Qwen text attention uses graph-safe FlashAttention; "
                "vision attention remains FlashAttention."
            )

    def _prime_graph_safe_fa2(self, static_input: Any) -> None:
        """Install the padded-FA2 adapter and allocate its fixed-shape buffers.

        The stock varlen path cannot be replayed: ``_get_unpad_data`` returns an
        ``indices`` tensor whose length is the *valid*-token count, and capture
        bakes that length into the recorded kernels.  Text length varies per
        batch, so the first replay with a different valid-token count is
        rejected (see ``update_cuda_graph_batch_metadata``).  The padded adapter
        instead packs to the fixed padded length with dummy tail sequences and
        recomputes ``cu_seqlens`` on device every replay, which is what makes a
        full-iteration graph possible at all.  It is mathematically equivalent
        to varlen FA2 but not bit-identical, and it also disables the
        ``attention_mask=None`` dense shortcut in ``Qwen3Backbone``.
        """
        input_ids = getattr(static_input, "input_ids", None)
        if input_ids is None:
            return
        backbone = self._backbone()
        language_model = backbone.language_model
        layers = getattr(language_model, "layers", None)
        if not layers:
            return
        config = language_model.config
        num_q_heads = int(getattr(config, "num_attention_heads"))
        num_kv_heads = int(
            getattr(config, "num_key_value_heads", num_q_heads)
        )
        head_dim = int(
            getattr(
                config,
                "head_dim",
                int(getattr(config, "hidden_size")) // num_q_heads,
            )
        )
        # Qwen attention projections may be stored in FP32 after the
        # compatibility patch, while the forward Q/K/V tensors are produced in
        # the trainer compute dtype.  The latter is the key used by FA2's
        # runtime buffer lookup.
        dtype = getattr(self.trainer, "_compute_dtype", None)
        if dtype is None:
            dtype = resolve_dtype(self.training_args.dtype)
        from loongforge.embodied.model.groot_n1_7.modules.cuda_graph_flash_attention import (
            _get_fa2_buffers,
        )

        _get_fa2_buffers(
            int(input_ids.shape[0]),
            int(input_ids.shape[-1]),
            num_q_heads,
            num_kv_heads,
            head_dim,
            dtype,
            self.ctx.device,
        )
        if maybe_install_graph_safe_fa2_patches(force=True):
            logger.info("Installed graph-safe padded FlashAttention before capture.")

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
        if self._fp8_enabled:
            if getattr(self.training_args, "fp8_backend", None) != "te":
                raise RuntimeError(
                    "GR00T-N1.7 CUDA graph FP8 support requires the TE backend."
                )
            if getattr(self.training_args, "fp8_te_recipe", None) != "blockwise":
                raise RuntimeError(
                    "GR00T-N1.7 CUDA graph FP8 support requires the blockwise TE recipe."
                )
            if getattr(self.training_args, "cuda_graph_scope", None) != "full_iteration":
                raise RuntimeError(
                    "GR00T-N1.7 CUDA graph FP8 support requires full_iteration scope."
                )
            from loongforge.embodied.distributed.fp8_utils.te_fp8.recipe import (
                build_fp8_recipe,
            )

            self._fp8_recipe = getattr(self.trainer, "_fp8_recipe", None)
            if self._fp8_recipe is None:
                self._fp8_recipe = build_fp8_recipe(
                    self.training_args.fp8_te_recipe,
                    recipe_args=self.training_args,
                )
                self.trainer._fp8_recipe = self._fp8_recipe
            if self._fp8_recipe.__class__.__name__ != "Float8BlockScaling":
                raise RuntimeError(
                    "Expected TransformerEngine Float8BlockScaling for the GR00T graph path."
                )

            try:
                from transformer_engine.pytorch import Linear as TELinear
            except ImportError as exc:
                raise RuntimeError("FP8 graph mode requires TransformerEngine PyTorch.") from exc
            fp8_linear_count = sum(isinstance(m, TELinear) for m in self._backbone().modules())
            if fp8_linear_count <= 0:
                raise RuntimeError(
                    "GR00T-N1.7 FP8 graph path requires at least one TE Linear "
                    "module in the frozen backbone."
                )
            logger.info(
                "GR00T-N1.7 FP8 graph coverage: te_linear=%d bf16_skip_modules=%s",
                fp8_linear_count,
                getattr(self.training_args, "fp8_skip_modules", None) or [],
            )

    def _iter_cuda_generators(self):
        """Return graph-safe default and parallel RNG generators without duplicates."""
        generators = []
        seen = set()

        def add(generator):
            if generator is None or id(generator) in seen:
                return
            if not hasattr(generator, "graphsafe_get_state"):
                raise RuntimeError(
                    "CUDA graph FP8 path requires graph-safe torch.Generator state APIs."
                )
            seen.add(id(generator))
            generators.append(generator)

        add(torch.cuda.default_generators[torch.cuda.current_device()])
        try:
            from megatron.core.tensor_parallel.random import get_all_rng_states

            try:
                for generator in get_all_rng_states().values():
                    add(generator)
            except AssertionError:
                # The embodied path may not initialize the optional parallel RNG
                # tracker on a single-GPU run; the default generator remains enough.
                pass
        except ImportError:
            pass
        return tuple(generators)

    def _register_graph_generators(self, graph: torch.cuda.CUDAGraph) -> None:
        """Register all RNG streams consumed by a captured graph."""
        for generator in self._iter_cuda_generators():
            graph.register_generator_state(generator)

    def _snapshot_cuda_generators(self):
        """Snapshot graph-safe RNG states before warmup/capture mutates them."""
        return tuple(
            (generator, generator.graphsafe_get_state())
            for generator in self._iter_cuda_generators()
        )

    @staticmethod
    def _restore_cuda_generators(states) -> None:
        for generator, state in states:
            generator.graphsafe_set_state(state)

    @staticmethod
    def _walk_tensor_pointers(value: Any, path: str, output: dict[str, int], seen: set[int]) -> None:
        if isinstance(value, torch.Tensor):
            if id(value) in seen:
                return
            seen.add(id(value))
            output[path] = value.data_ptr()
            return
        if isinstance(value, dict):
            for key, item in value.items():
                GrootN1d7FullIterationCudaGraphRunner._walk_tensor_pointers(
                    item, f"{path}.{key}", output, seen
                )
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                GrootN1d7FullIterationCudaGraphRunner._walk_tensor_pointers(
                    item, f"{path}[{index}]", output, seen
                )

    def _fp8_pointer_signature(self, module: torch.nn.Module) -> dict[str, int]:
        pointers: dict[str, int] = {}
        seen: set[int] = set()
        for name, buffer in module.named_buffers():
            pointers[f"buffer:{name}"] = buffer.data_ptr()
        for module_name, child in module.named_modules():
            meta = getattr(child, "fp8_meta", None)
            if meta is not None:
                self._walk_tensor_pointers(meta, f"meta:{module_name}", pointers, seen)
        return pointers

    def _remember_fp8_pointers(self, module: torch.nn.Module) -> None:
        self._fp8_tensor_pointers = self._fp8_pointer_signature(module)

    def _assert_fp8_pointers_stable(self, module: torch.nn.Module) -> None:
        if self._fp8_tensor_pointers is None:
            return
        current = self._fp8_pointer_signature(module)
        if current != self._fp8_tensor_pointers:
            changed = sorted(
                set(current) | set(self._fp8_tensor_pointers),
                key=str,
            )
            changed = [
                key
                for key in changed
                if current.get(key) != self._fp8_tensor_pointers.get(key)
            ]
            raise RuntimeError(
                "FP8 tensor/scale storage changed after CUDA graph capture; "
                f"first changed entries: {changed[:5]}"
            )

    def _prepare_fp8_graph_state(self, module: torch.nn.Module) -> None:
        if not self._fp8_enabled:
            return
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(False)
        skip_tensor = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        if skip_tensor is None or not skip_tensor.is_cuda:
            raise RuntimeError("TE did not create a CUDA FP8 weight-cache scalar.")
        pointer = skip_tensor.data_ptr()
        if self._fp8_skip_weight_update_ptr is None:
            self._fp8_skip_weight_update_ptr = pointer
        elif pointer != self._fp8_skip_weight_update_ptr:
            raise RuntimeError("TE FP8 weight-cache scalar was reallocated.")

        # Blockwise TE does not use delayed global amax reduction, but assigning
        # the cached recipe/group makes the module metadata explicit and mirrors
        # Megatron's graph wrapper for future recipe extensions.
        for child in module.modules():
            fp8_meta = getattr(child, "fp8_meta", None)
            if isinstance(fp8_meta, dict):
                fp8_meta["recipe"] = self._fp8_recipe
                if self._fp8_group is not None:
                    fp8_meta["fp8_group"] = self._fp8_group

    @contextmanager
    def _fp8_capture_context(self, module: torch.nn.Module):
        """Protect TE global state while recording a custom graph."""
        if not self._fp8_enabled:
            yield
            return

        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
        from transformer_engine.pytorch.graph import (
            restore_fp8_tensors,
            save_fp8_tensors,
            set_capture_end,
            set_capture_start,
        )

        self._prepare_fp8_graph_state(module)
        saved_state = FP8GlobalStateManager.get_autocast_state()
        saved_fp8_tensors = save_fp8_tensors([module], self._fp8_recipe)
        set_capture_start()
        try:
            with self.trainer._fp8_graph_capture_ctx():
                yield
        finally:
            try:
                set_capture_end()
            finally:
                restore_fp8_tensors([module], saved_fp8_tensors)
                FP8GlobalStateManager.set_autocast_state(saved_state)

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
                key: (
                    self._convert_pipeline_value(inputs[key], backbone_dtype).to(dtype=torch.bool)
                    if key == "attention_mask"
                    and isinstance(inputs[key], torch.Tensor)
                    and inputs[key].dtype != torch.bool
                    else self._convert_pipeline_value(inputs[key], backbone_dtype)
                )
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

    def _graph_batch_incompatibility(self, batch: Any) -> str | None:
        try:
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
        except RuntimeError as error:
            return str(error)
        return None

    def _synchronize_fallback(self, local_reason: str | None) -> bool:
        flag = torch.tensor(
            int(local_reason is not None), dtype=torch.int32, device=self.ctx.device
        )
        if self.ctx.is_distributed:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        fallback = bool(flag.item())
        if fallback and self.ctx.is_main:
            logger.warning(
                "Full-iteration CUDA graph input became incompatible%s; "
                "all ranks are switching permanently to eager execution.",
                f": {local_reason}" if local_reason else " on another rank",
            )
        return fallback

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
        if (
            not self._input_prefetch_enabled
            or self.graph is None
            or self._eager_fallback
        ):
            return
        completed_steps = self.warmup_count + self.replay_count
        if completed_steps >= int(self.training_args.train_iters):
            return
        if self._prefetched_gpu_batch is not None:
            raise RuntimeError("Full-iteration input prefetch staging batch was not consumed.")

        with self.trainer._stage_timers("batch-generator"):
            cpu_batch = self._fetch_cpu_batch()
            local_reason = self._graph_batch_incompatibility(cpu_batch)
            if self._synchronize_fallback(local_reason):
                self._fallback_cpu_batch = cpu_batch
                self._eager_fallback = True
                self._wait_default_stream()
                return
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
            update_metadata = getattr(
                self._backbone(), "update_cuda_graph_batch_metadata", None
            )
            if callable(update_metadata):
                update_metadata(self._backbone_static_input, self._prefetched_cpu_batch)
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
            if self._eager_fallback:
                cpu_batch = self._fallback_cpu_batch
                self._fallback_cpu_batch = None
                if cpu_batch is None:
                    cpu_batch = self._fetch_cpu_batch()
                batch = self.trainer._move_batch_to_device(cpu_batch)
            elif self.graph is not None and self._input_prefetch_enabled:
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
                    local_reason = self._graph_batch_incompatibility(cpu_batch)
                    if self._synchronize_fallback(local_reason):
                        self._eager_fallback = True
                        batch = self.trainer._move_batch_to_device(cpu_batch)
                        cpu_batch = None
                    if cpu_batch is None:
                        pass
                    else:
                        self._copy_stream.wait_stream(default_stream)
                        with torch.cuda.stream(self._copy_stream):
                            _copy_static(self.static_batch, cpu_batch)
                            update_metadata = getattr(
                                self._backbone(), "update_cuda_graph_batch_metadata", None
                            )
                            if callable(update_metadata):
                                update_metadata(self.static_batch, cpu_batch)
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

        # Preserve the exact reference ordering: Beta.sample returns a CPU
        # fp32 tensor and the eager action head then converts it to the action
        # dtype/device.  Sampling directly as bf16 changes the Beta values.
        sample_cpu = action_head.beta_dist.sample([actions.shape[0]])
        sample_cpu = ((1 - sample_cpu) * action_head.config.noise_s).to(
            dtype=actions.dtype
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
        action_head = self._action_head()
        sample_cpu = action_head.beta_dist.sample([actions.shape[0]])
        sample_cpu = ((1 - sample_cpu) * action_head.config.noise_s).to(
            dtype=actions.dtype
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

    def _clear_noise_buffer(self) -> None:
        self._action_head()._split_noise_buf = None

    def _clear_state_dropout_buffer(self) -> None:
        self._action_head()._split_state_dropout_buf = None

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
            # Keep explicit graph-owned scalar outputs.  A bare detach can
            # alias a temporary reduction allocation whose storage is reused
            # before the capture call returns.
            action_loss=loss.detach().clone(),
            grad_norm=grad_norm.detach().clone(),
            nan_flag=(~finite).to(dtype=torch.int32).clone(),
            spike_flag=(~valid).to(dtype=torch.int32).clone(),
        )

    def _wait_default_stream(self) -> None:
        torch.cuda.current_stream(self.ctx.device).wait_stream(self.graph_stream)

    def _run_eager_warmup(self, batch: Any) -> tuple[dict[str, torch.Tensor], float]:
        self.static_batch = batch
        default_stream = torch.cuda.current_stream(self.ctx.device)
        self.graph_stream.wait_stream(default_stream)
        with torch.cuda.stream(self.graph_stream):
            self._clear_time_buffer()
            self._clear_noise_buffer()
            self._clear_state_dropout_buffer()
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
        self._prime_graph_safe_fa2(self._backbone_static_input)
        input_ids = getattr(self._backbone_static_input, "input_ids", None)
        if input_ids is not None:
            prime_graph_safe_fa2_buffers((int(input_ids.shape[-1]),), self.ctx.device)
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
        backbone_graph = torch.cuda.CUDAGraph()
        self._register_graph_generators(backbone_graph)
        backbone_rng_states = self._snapshot_cuda_generators()
        progress_hook = self._register_backbone_progress_hook()
        try:
            with self._fp8_capture_context(self._backbone()):
                with torch.cuda.graph(
                    backbone_graph,
                    stream=self._backbone_stream,
                    pool=self._graph_pool,
                    capture_error_mode="thread_local",
                ):
                    with torch.no_grad(), self._train_autocast_context():
                        backbone_output = self._backbone()(self._backbone_static_input)
        finally:
            if progress_hook is not None:
                progress_hook.remove()
        self._backbone_graph = backbone_graph
        self._backbone_output = backbone_output
        self._remember_fp8_pointers(self._backbone())

        # The capture executes on the dedicated backbone stream. The action
        # graph owns a separate copy of these outputs, so establish the
        # producer/consumer dependency before cloning on the current stream.
        torch.cuda.current_stream(self.ctx.device).wait_stream(self._backbone_stream)
        torch.cuda.synchronize(self.ctx.device)
        self._restore_cuda_generators(backbone_rng_states)

        self.static_batch = _ActionGraphBatch(
            backbone_output=_clone_static(backbone_output),
            action_input=_clone_static(action_input),
        )
        # Eager draws the CUDA Philox stream in the order state-dropout
        # (torch.rand) -> action noise (torch.randn) inside the action head
        # forward.  Both draws stay inside the captured graph so that replay
        # reproduces that order against the graph-registered generators; only
        # the CPU Beta timestep, which consumes no device Philox, is prefilled.
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
        self._register_graph_generators(graph)
        graph_rng_states = self._snapshot_cuda_generators()
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
                pool=self._graph_pool,
                capture_error_mode="thread_local",
            ):
                outputs = self._iteration_body(set_to_none=False)
        finally:
            if saved_logger is not None:
                ddp_model.logger = saved_logger
            self._restore_cuda_generators(graph_rng_states)

        torch.cuda.synchronize(self.ctx.device)
        self.ctx.barrier()
        self.graph = graph
        self.outputs = outputs
        self.capture_count += 1
        if self.ctx.is_main:
            logger.info(
                "Full-iteration CUDA graph capture complete: session=%d, graphs=2, "
                "fp8_backbone=%s, eager_fallbacks=%d",
                self.capture_count,
                self._fp8_enabled,
                self.fallback_count,
            )
        if self._fp8_enabled:
            self._remember_fp8_pointers(self._backbone())
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

        # CUDA stream capture records work without executing it.  Replay the
        # backbone for the captured input first, then replay the action graph;
        # this is the first real optimizer iteration and the only scheduler
        # advancement associated with this outer training step.
        self._replay_captured_backbone()
        self._capture_replay_pending = True
        return self._replay(self.static_batch)

    def _capture(self, batch: Any) -> tuple[dict[str, torch.Tensor], float]:
        if self._backbone_pipeline_enabled:
            return self._capture_backbone_pipeline(batch)
        self.static_batch = _clone_static(batch)
        self._prepare_graph_batch(self.static_batch)
        self._prime_graph_safe_fa2(self.static_batch)
        input_ids = getattr(self.static_batch, "input_ids", None)
        if input_ids is not None:
            prime_graph_safe_fa2_buffers(
                (int(input_ids.shape[-1]),),
                self.ctx.device,
            )
        self._prepare_fp8_graph_state(self.raw_model)
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
        self._register_graph_generators(graph)
        graph_rng_states = self._snapshot_cuda_generators()
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
                pool=self._graph_pool,
                capture_error_mode="thread_local",
            ):
                with self._fp8_capture_context(self.raw_model):
                    outputs = self._iteration_body(set_to_none=False)
        finally:
            if saved_logger is not None:
                ddp_model.logger = saved_logger
            self._restore_cuda_generators(graph_rng_states)

        torch.cuda.synchronize(self.ctx.device)
        self.ctx.barrier()

        self.graph = graph
        self.outputs = outputs
        self.capture_count += 1
        if self.ctx.is_main:
            logger.info(
                "Full-iteration CUDA graph capture complete: session=%d, graphs=1, "
                "fp8=%s, eager_fallbacks=%d",
                self.capture_count,
                self._fp8_enabled,
                self.fallback_count,
            )
        if self._fp8_enabled:
            self._remember_fp8_pointers(self._backbone())
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
        # Capture records kernels but does not execute them.  The first replay
        # below is therefore the first real optimizer iteration.
        self._capture_replay_pending = True
        return self._replay(self.static_batch)

    def _replay_captured_backbone(self) -> None:
        """Materialize the pipeline backbone output for the capture batch."""
        if not self._backbone_pipeline_enabled:
            return
        if (
            self._backbone_graph is None
            or self._backbone_static_input is None
            or not isinstance(self.static_batch, _ActionGraphBatch)
            or self._backbone_output is None
        ):
            raise RuntimeError("Captured frozen-backbone graph is not ready for its first replay.")
        self._ensure_copy_resources()
        self._ensure_backbone_pipeline_resources()
        assert self._backbone_stream is not None
        assert self._backbone_ready_event is not None
        assert self._copy_stream is not None
        assert self._copy_event is not None
        default_stream = torch.cuda.current_stream(self.ctx.device)
        self._backbone_stream.wait_stream(default_stream)
        with torch.cuda.stream(self._backbone_stream):
            self._backbone_graph.replay()
            self._backbone_ready_event.record(self._backbone_stream)
        self._copy_stream.wait_event(self._backbone_ready_event)
        with torch.cuda.stream(self._copy_stream):
            _copy_static(self.static_batch.backbone_output, self._backbone_output)
            self._copy_event.record(self._copy_stream)
        default_stream.wait_event(self._copy_event)

    def _replay(self, batch: Any) -> tuple[dict[str, torch.Tensor], float]:
        if batch is not self.static_batch:
            raise RuntimeError("Full-iteration CUDA graph replay did not receive its static batch.")
        if self._capture_replay_pending:
            # _fill_time_buffer() ran immediately before capture.  Reusing
            # that value preserves the eager RNG sequence for the first real
            # graph iteration.
            self._capture_replay_pending = False
        elif self._time_prefetch_enabled:
            self._consume_prefetched_time_buffer()
        else:
            self._fill_time_buffer()
        assert self.graph is not None
        if self._fp8_enabled:
            self._assert_fp8_pointers_stable(self._backbone())
            skip_ptr = self._fp8_skip_weight_update_ptr
            if skip_ptr is not None:
                from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

                skip_tensor = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
                if skip_tensor is None or skip_tensor.data_ptr() != skip_ptr:
                    raise RuntimeError("TE FP8 weight-cache scalar changed before replay.")
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
        # Clone the scalar before another graph from the shared pool can reuse
        # the capture output allocation (notably on the first capture step).
        action_loss = self.outputs.action_loss.detach().clone()
        nan_flag = int(self.outputs.nan_flag.item())
        spike_flag = int(self.outputs.spike_flag.item())
        self.trainer.nan_iterations += nan_flag
        self.trainer.skipped_iterations += spike_flag
        self._advance_scheduler()
        return {"action_loss": action_loss}, float(self.outputs.grad_norm)

    def _run_eager_fallback_step(self, batch: Any) -> tuple[dict[str, torch.Tensor], float]:
        """Continue with native inputs while preserving optimizer and scheduler state."""
        if self.fallback_count == 0:
            self.fallback_count = 1
            # No captured work is pending when fallback is selected: the
            # preceding replay has completed and the incompatible batch was
            # only inspected on CPU. Release the private graph pools before
            # running eager, otherwise the action-head forward cannot allocate
            # its normal activations on a 96-GiB device.
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
            self._prefetched_gpu_batch = None
            self._prefetched_cpu_batch = None
            self._backbone_pending = False
            self._clear_time_buffer()
            self._clear_noise_buffer()
            self._clear_state_dropout_buffer()
            self.time_buffer = None
            self._time_host_buffer = None
            self._time_prefetch_buffer = None
            self._time_prefetch_host_buffers = []
            # Remove pointer-keyed Qwen metadata belonging to the destroyed
            # static graph tensors. Native eager batches must rebuild their
            # own visual/attention metadata from their current pointers.
            backbone = self._backbone()
            qwen_model = getattr(backbone.model, "model", backbone.model)
            from loongforge.embodied.model.groot_n1_7.modules import qwen3_backbone

            qwen3_backbone._CUDA_GRAPH_ATTENTION_MASK_METADATA.clear()
            for owner in (backbone, qwen_model, getattr(backbone, "language_model", None)):
                if owner is None:
                    continue
                for name in tuple(owner.__dict__):
                    if name.startswith("_loongforge_cuda_graph_"):
                        owner.__dict__.pop(name, None)
            torch.cuda.empty_cache()
            # DDP's reducer/logger was captured with graph-owned CUDA events.
            # Returning through DDP.forward after replay can make its eager
            # runtime-stat logger touch those captured events. The model has no
            # mutable buffers requiring broadcast, so keep the graph-era DDP
            # buffer policy and bypass only DDP's forward wrapper. Gradients
            # are reduced explicitly below in a fixed parameter order.
            if self.ctx.is_main:
                logger.warning(
                    "Full-iteration CUDA graph disabled after %d replays; "
                    "continuing with eager steps and the existing optimizer state.",
                    self.replay_count,
                )
        ddp_model = self.trainer.model
        saved_logger = getattr(ddp_model, "logger", None)
        if saved_logger is not None:
            ddp_model.logger = self._noop_ddp_logger
        if self._saved_ddp_broadcast_buffers is not None:
            ddp_model.broadcast_buffers = self._saved_ddp_broadcast_buffers
        try:
            prepare = getattr(self._backbone(), "prepare_cuda_graph_batch", None)
            if callable(prepare):
                prepare(batch)
            self._zero_grad(set_to_none=True)
            loss, _log_losses = self.trainer._train_forward(batch)
        finally:
            if saved_logger is not None:
                ddp_model.logger = saved_logger
        scaled_loss = loss / self.training_args.gradient_accumulation_steps
        finite = torch.isfinite(scaled_loss)
        below_threshold = scaled_loss <= float(self.training_args.loss_spike_threshold)
        valid = finite & below_threshold
        torch.where(valid, scaled_loss, torch.zeros_like(scaled_loss)).backward()
        self._clean_nan_gradients()
        params = [parameter for parameter in self.raw_model.parameters() if parameter.grad is not None]
        max_norm = float(self.training_args.clip_grad)
        if self._fused_optimizer_grad_clip:
            grad_norm, grad_scale = _compute_grad_norm_and_clip_scale(
                [parameter.grad for parameter in params], max_norm
            )
            self.trainer.optimizer.set_grad_scale(grad_scale)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, max_norm if max_norm > 0 else float("inf"), error_if_nonfinite=False
            )
        self.trainer.optimizer.step()
        outputs = _GraphOutputs(
            action_loss=loss.detach().clone(),
            grad_norm=grad_norm.detach().clone(),
            nan_flag=(~finite).to(dtype=torch.int32).clone(),
            spike_flag=(~valid).to(dtype=torch.int32).clone(),
        )
        nan_flag = int(outputs.nan_flag.item())
        spike_flag = int(outputs.spike_flag.item())
        self.trainer.nan_iterations += nan_flag
        self.trainer.skipped_iterations += spike_flag
        self._advance_scheduler()
        return {"action_loss": outputs.action_loss}, float(outputs.grad_norm)

    def step(self) -> tuple[dict[str, torch.Tensor], float]:
        """Run one training iteration via eager warmup, capture, or replay."""
        self._validate_optimizer()
        batch = self._fetch_batch()
        if self._eager_fallback:
            return self._run_eager_fallback_step(batch)
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
        if self.ctx.is_main:
            logger.info(
                "Full-iteration CUDA graph summary: warmup_steps=%d, "
                "capture_sessions=%d, replays=%d, eager_fallbacks=%d",
                self.warmup_count,
                self.capture_count,
                self.replay_count,
                self.fallback_count,
            )
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
        self._clear_noise_buffer()
        self._clear_state_dropout_buffer()
        self.ctx.barrier()

    @property
    def captured(self) -> bool:
        """Return whether the training graph has been captured."""
        return self.graph is not None
