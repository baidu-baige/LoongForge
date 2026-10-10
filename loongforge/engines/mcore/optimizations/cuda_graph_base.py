# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Common scaffolding for CUDA graph wrappers around forward_backward_func.

Model-agnostic. Any model family can build a CUDA graph wrapper by subclassing
``BaseCudaGraphWrapper`` and implementing the three phase hooks.

Contains:
- Tensor-struct utilities (deep copy / in-place clone / shape extract / shape compare)
- ``StaticBufferLoader``: per-microbatch static buffer manager with stable addresses
- ``BaseCudaGraphWrapper``: template-method ``torch.nn.Module`` orchestrating the
  warmup / capture / replay phase machine. Subclasses override the three
  phase implementations (``_run_capture``, ``_run_replay``, ``_invalidate_graph_state``)
  while sharing data ingestion, warmup, shape tracking, loss allreduce, and
  iteration accounting.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext

import torch
from megatron.training.utils import average_losses_across_data_parallel_group

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor-struct utilities
# ---------------------------------------------------------------------------
def copy_tensors_in_struct(src):
    """Deep-copy a nested data structure, cloning all tensors to CUDA."""
    if isinstance(src, tuple):
        return tuple(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, list):
        return list(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, dict):
        return {k: copy_tensors_in_struct(v) for k, v in src.items()}
    elif isinstance(src, torch.Tensor):
        return src.clone().detach().cuda()
    else:
        return src


def clone_tensors_in_struct(tgt, src):
    """Clone tensors from src into pre-allocated tgt structure via copy_().

    Raises KeyError with a descriptive message if src is missing a key that
    exists in tgt (batch structure mismatch). The caller (StaticBufferLoader)
    catches this and triggers graph re-capture.
    """
    if isinstance(tgt, dict) and isinstance(src, dict):
        for k in tgt:
            if k not in src:
                raise KeyError(
                    f"Batch structure mismatch: key '{k}' present in static "
                    "buffer but absent in new batch. This typically means the "
                    "data pipeline yields batches with inconsistent fields."
                )
            clone_tensors_in_struct(tgt[k], src[k])
    elif isinstance(tgt, (tuple, list)) and isinstance(src, (tuple, list)):
        for t, s in zip(tgt, src):
            clone_tensors_in_struct(t, s)
    elif isinstance(tgt, torch.Tensor) and isinstance(src, torch.Tensor):
        tgt.copy_(src)


def extract_tensor_shapes(data):
    """Extract shape signatures for all tensors in a nested structure."""
    if isinstance(data, tuple):
        return tuple(extract_tensor_shapes(i) for i in data)
    elif isinstance(data, list):
        return list(extract_tensor_shapes(i) for i in data)
    elif isinstance(data, dict):
        return {k: extract_tensor_shapes(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.size()
    else:
        return type(data).__name__


def shapes_match(shape_a, shape_b):
    """Compare two shape signature structures for equality."""
    if isinstance(shape_a, torch.Size) and isinstance(shape_b, torch.Size):
        return shape_a == shape_b
    elif isinstance(shape_a, dict) and isinstance(shape_b, dict):
        return shape_a.keys() == shape_b.keys() and all(
            shapes_match(shape_a[k], shape_b[k]) for k in shape_a
        )
    elif isinstance(shape_a, (tuple, list)) and isinstance(shape_b, (tuple, list)):
        return len(shape_a) == len(shape_b) and all(
            shapes_match(a, b) for a, b in zip(shape_a, shape_b)
        )
    return shape_a == shape_b


# ---------------------------------------------------------------------------
# StaticBufferLoader
# ---------------------------------------------------------------------------
class StaticBufferLoader:
    """Manages static tensor buffers for pre-loading data before graph replay.

    Maintains one independent static buffer per microbatch slot. Each slot has
    a fixed memory address that persists across iterations — new data is copied
    in via copy_() without reallocating.
    """

    def __init__(self):
        self._buffers: list = None  # list of per-microbatch static buffers

    def load(self, batch, slot_idx):
        """Load a batch into the static buffer at slot_idx.

        First call allocates the buffer; subsequent calls copy_() into it.
        Returns the static buffer reference (same memory address every time).
        """
        if self._buffers is None:
            self._buffers = []

        # Extend buffer list if needed
        while len(self._buffers) <= slot_idx:
            self._buffers.append(None)

        if self._buffers[slot_idx] is None:
            self._buffers[slot_idx] = copy_tensors_in_struct(batch)
        else:
            try:
                clone_tensors_in_struct(self._buffers[slot_idx], batch)
            except Exception as e:
                # Reallocating breaks CUDA graph's captured tensor addresses.
                # Log a warning and raise so the graph wrapper can re-capture.
                logger.warning(
                    f"[StaticBufferLoader] clone failed at slot {slot_idx}: {e}. "
                    "Reallocating buffer — CUDA graph will be invalidated."
                )
                self._buffers[slot_idx] = copy_tensors_in_struct(batch)
                raise RuntimeError(
                    f"Static buffer reallocated at slot {slot_idx} after CUDA graph capture; "
                    "graph must be re-captured."
                ) from e

        return self._buffers[slot_idx]


# ---------------------------------------------------------------------------
# BaseCudaGraphWrapper
# ---------------------------------------------------------------------------
class BaseCudaGraphWrapper(torch.nn.Module):
    """Template-method base class for CUDA graph wrappers around forward_backward_func.

    Phase machine driven by ``__call__``:
        warmup (eager, populates static buffers and shape signature)
        ─► capture (record graph(s))
        ─► replay (load new data, replay graph(s), post-process loss)

    Subclasses MUST override:
        - ``_run_capture(num_microbatches, kwargs)``  : record graph(s) + eager re-run
        - ``_run_replay(num_microbatches, kwargs)``   : load → replay → post-process
        - ``_invalidate_graph_state()``               : drop subclass-specific captured state

    Subclasses MAY override (default no-op):
        - ``_before_warmup_forward()``  : hook before eager forward in warmup
        - ``_after_warmup_forward()``   : hook after eager forward in warmup
        - ``_on_buffer_realloc(...)``   : kwargs builder for recursive __call__
        - ``_suppress_loss_allreduce()``: ctxmgr to disable loss allreduce in graph
        - ``LOG_TAG``                   : log tag
        - ``_warmup_sentinel_attrs``    : submodule attrs to clear on invalidation
    """

    # Class name shown in log lines; subclasses override for clarity.
    LOG_TAG = "BaseCudaGraphWrapper"
    # Submodule attribute names that act as warmup-phase sentinels and must be
    # cleared (set to False) on graph invalidation to prevent stale state from
    # blocking re-capture. Subclasses extend.
    _warmup_sentinel_attrs: tuple[str, ...] = ()

    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=3):
        super().__init__()
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.captured: bool = False
        self._call_count: int = 0
        self._cuda_graph_warmup_steps: int = max(cuda_graph_warmup_steps, 1)
        self._shape_signatures = None
        self._needs_recapture: bool = False
        # Persistent references to graph's static loss output tensors.
        # These are the tensors the graph writes to on each replay — we must
        # always read from THESE after replay, never from a replaced reference.
        self._graph_loss_tensors: list[torch.Tensor] = []
        self._model_ref: list | None = None
        self._config = None

    # ----------------------- shared utilities ---------------------------
    def data_read(self, data_iterator, num_microbatches):
        """Read num_microbatches from data_iterator into per-slot static buffers.

        Each microbatch gets its own static buffer with a fixed memory address.
        Returns a list of static buffer references (or None if iterator missing).
        """
        static_batches = []
        iterator0 = data_iterator[0] if isinstance(data_iterator, list) else data_iterator
        if iterator0 is None:
            return None
        for slot_idx in range(num_microbatches):
            batch = next(iterator0)
            static_buf = self.static_loader.load(batch, slot_idx)
            static_batches.append(static_buf)
        return static_batches

    @staticmethod
    def _set_warmup_flag(model, value: bool):
        """Set ``_in_graph_warmup`` on every submodule. Required because checks
        happen in nested modules that don't inherit attributes from parents."""
        if model is None:
            return
        for _mc in model:
            for _submod in _mc.modules():
                _submod._in_graph_warmup = value

    @staticmethod
    def _zero_grad_buffers(model):
        if model is None:
            return
        for mc in model:
            if hasattr(mc, 'zero_grad_buffer'):
                mc.zero_grad_buffer()
            else:
                mc.zero_grad()

    def _cache_model_config(self, model):
        if self._config is None and model is not None:
            from megatron.core.pipeline_parallel.schedules import get_model_config
            m = model[0] if isinstance(model, list) else model
            self._config = get_model_config(m)

    def _allreduce_and_update_losses(self, result_dicts):
        """Allreduce ``self._graph_loss_tensors`` and write averaged values back
        into the matching ``loss`` entries of ``result_dicts`` in-place."""
        if not self._graph_loss_tensors:
            return
        _avg = average_losses_across_data_parallel_group(self._graph_loss_tensors)
        _loss_idx = 0
        for _r in result_dicts:
            if isinstance(_r, dict) and "loss" in _r:
                _r["loss"] = _avg[_loss_idx]
                _loss_idx += 1

    def _load_replay_data_or_recapture(self, num_microbatches, kwargs):
        """Load replay data into static buffers and detect graph invalidation.

        Handles two re-capture triggers:
          1. ``StaticBufferLoader`` raised buffer-reallocation error.
          2. Input shape changed since capture (``shapes_match`` False).

        Returns ``(static_batches, recapture_kwargs)``:
          - When ``recapture_kwargs is None``: caller proceeds to replay with
            ``static_batches`` (may be ``None`` if data_iterator absent).
          - When ``recapture_kwargs`` is set: caller MUST return
            ``self.__call__(**recapture_kwargs)``; ``_invalidate_graph`` already
            ran inside this method.
        """
        data_iter = kwargs.get("data_iterator")
        if data_iter is None:
            return None, None

        try:
            static_batches = self.data_read(data_iter, num_microbatches)
        except RuntimeError as e:
            if "Static buffer reallocated" in str(e):
                logger.info(f"[{self.LOG_TAG}] Buffer reallocated, re-capturing graph")
                new_kwargs = self._on_buffer_realloc(e, data_iter, num_microbatches, kwargs)
                self._invalidate_graph()
                return None, new_kwargs
            raise

        if static_batches and self._shape_signatures is not None:
            new_shape = extract_tensor_shapes(static_batches[0])
            if not shapes_match(self._shape_signatures, new_shape):
                logger.info(f"[{self.LOG_TAG}] Shape change detected, recapturing")
                self._invalidate_graph()
                new_kwargs = dict(kwargs)
                new_kwargs["data_iterator"] = [iter(static_batches)]
                return None, new_kwargs

        return static_batches, None

    def _on_buffer_realloc(self, error, data_iter, num_microbatches, kwargs):
        """Hook: build kwargs for recursive ``__call__`` after a buffer realloc.

        Default: re-enter with original kwargs. Subclasses may override to
        consume remaining slots from the iterator before recursing (avoids
        stale data when only some microbatches were loaded before the error).
        """
        return dict(kwargs)

    @staticmethod
    def _with_static_data(kwargs, static_batches):
        """Return a fresh kwargs dict with ``data_iterator`` replaced by an
        iterator over ``static_batches``. Returns ``kwargs`` unchanged when
        ``static_batches`` is ``None``."""
        if static_batches is None:
            return kwargs
        new_kwargs = dict(kwargs)
        new_kwargs["data_iterator"] = [iter(static_batches)]
        return new_kwargs

    @contextmanager
    def _preserve_rng_state(self):
        """Context manager: snapshot CUDA RNG state on enter, restore on exit.

        Use around code that consumes the default generator (e.g. graph capture
        with TE RNG tracker, or eager RNG buffer pre-allocation) so the
        externally observable RNG sequence is unaffected.
        """
        state = torch.cuda.get_rng_state()
        try:
            yield
        finally:
            torch.cuda.set_rng_state(state)

    def _suppress_loss_allreduce(self):
        """Context manager: disable cross-DP loss allreduce within the block.

        Default: no-op (``nullcontext``). Subclasses override to plug into a
        model-specific suppression mechanism (e.g. a module-level flag read by
        the loss reduction code path).
        """
        return nullcontext()

    def _clear_warmup_sentinels(self):
        """Walk every submodule in the captured model and reset any attribute
        listed in ``_warmup_sentinel_attrs`` to False. Subclasses extend the
        attribute list to plug in additional sentinels."""
        if self._model_ref is None or not self._warmup_sentinel_attrs:
            return
        for _mc in self._model_ref:
            for _submod in _mc.modules():
                for attr in self._warmup_sentinel_attrs:
                    if hasattr(_submod, attr):
                        setattr(_submod, attr, False)

    # ----------------------- template method ----------------------------
    def __call__(self, *args, **kwargs):
        num_microbatches = kwargs.get("num_microbatches", 1)
        self._call_count += 1
        curr_iter = self._call_count - 1

        if curr_iter < self._cuda_graph_warmup_steps:
            return self._run_warmup(curr_iter, num_microbatches, kwargs)

        if not self.captured:
            return self._run_capture(num_microbatches, kwargs)

        if self._needs_recapture:
            self._invalidate_graph()
            return self.__call__(*args, **kwargs)

        return self._run_replay(num_microbatches, kwargs)

    # ----------------------- warmup (shared) ----------------------------
    def _run_warmup(self, curr_iter, num_microbatches, kwargs):
        data_iter = kwargs.get("data_iterator")
        if data_iter is not None:
            static_batches = self.data_read(data_iter, num_microbatches)
            if static_batches is not None:
                if curr_iter == 0:
                    self._shape_signatures = extract_tensor_shapes(static_batches[0])
                    logger.info(f"[{self.LOG_TAG}] Recorded shape signature from warmup step 0")
                kwargs = dict(kwargs)
                kwargs["data_iterator"] = [iter(static_batches)]

        model = kwargs.get("model")
        if model is not None:
            self._model_ref = model
        self._set_warmup_flag(model, True)

        self._before_warmup_forward()
        result = self.forward_backward_func(**kwargs)
        self._after_warmup_forward()

        self._set_warmup_flag(model, False)
        self._cache_model_config(model)

        return result

    # ----------------------- subclass hooks -----------------------------
    def _before_warmup_forward(self):
        """Hook called before eager forward in warmup. Default no-op."""

    def _after_warmup_forward(self):
        """Hook called after eager forward in warmup. Default no-op."""

    def _run_capture(self, num_microbatches, kwargs):
        raise NotImplementedError

    def _run_replay(self, num_microbatches, kwargs):
        raise NotImplementedError

    def _invalidate_graph(self):
        """Invalidate captured graph state. Calls subclass
        ``_invalidate_graph_state`` for graph-specific cleanup, then resets
        common state and clears submodule warmup sentinels.

        Subclasses should NOT override this directly — override
        ``_invalidate_graph_state`` instead.
        """
        self._invalidate_graph_state()
        self.captured = False
        self._call_count = 0
        self._needs_recapture = True
        self._graph_loss_tensors = []
        self._shape_signatures = None
        self._clear_warmup_sentinels()

    def _invalidate_graph_state(self):
        """Subclass hook: drop subclass-specific captured graph state
        (e.g. CUDAGraph handles, sub-graph buffers). Common state reset is
        handled by ``_invalidate_graph`` after this returns."""
        raise NotImplementedError

    # ----------------------- iteration accounting ----------------------
    @property
    def curr_iter(self):
        """Return the current iteration index."""
        return self._call_count - 1

    def next_iter(self):
        """Increment the call count for the next iteration."""
        self._call_count += 1
