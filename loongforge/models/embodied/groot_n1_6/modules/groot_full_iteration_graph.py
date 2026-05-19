# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GrootFullIterationGraph: Full-iteration CUDA graph for GRooT training.

Captures the entire forward_backward_func (all micro-batches of forward +
backward + gradient sync) into a single CUDA graph, eliminating kernel launch
overhead and Python interpreter overhead between layers.

Controlled by training args: --cuda-graph-impl=local --cuda-graph-scope=full_iteration
"""
from __future__ import annotations

import logging

import torch

from loongforge.models.common.cuda_graph_base import (
    BaseCudaGraphWrapper,
    copy_tensors_in_struct,
)

from .groot_graph_mixin import GrootGraphMixin

logger = logging.getLogger(__name__)


class GrootFullIterationGraph(GrootGraphMixin, BaseCudaGraphWrapper):
    """Full-iteration CUDA Graph wrapper for GRooT VLA training."""

    LOG_TAG = "GrootFullIterationGraph"
    # Submodules (e.g. ViT) flip this on during warmup when they see padding;
    # must be cleared on graph invalidation to allow re-capture.
    _warmup_sentinel_attrs = ("_capture_has_invalid_images",)

    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=3):
        super().__init__(forward_backward_func, cuda_graph_warmup_steps)
        self.cuda_graph: dict[str, torch.cuda.CUDAGraph] = {}
        # Instance-level result storage (NOT class-level) to avoid cross-instance pollution.
        self._graph_result: list = []

    def _invalidate_graph_state(self):
        self.cuda_graph.clear()

    def _run_capture(self, num_microbatches, kwargs):
        training_str = kwargs.get("training_str", "train")
        training_str_key = training_str + "_capture"

        # Load data into static buffers
        data_iter = kwargs.get("data_iterator")
        static_batches = None
        if data_iter is not None:
            static_batches = self.data_read(data_iter, num_microbatches)
            kwargs = self._with_static_data(kwargs, static_batches)

        if training_str_key not in self.cuda_graph:
            self.cuda_graph[training_str_key] = torch.cuda.CUDAGraph()
        cg = self.cuda_graph[training_str_key]

        # Suppress allreduce + isolate RNG consumption inside graph capture.
        with self._suppress_loss_allreduce(), self._preserve_rng_state():
            torch.cuda.synchronize()
            with torch.cuda.graph(cg):
                # Re-wrap static buffers as fresh iterator for capture
                if static_batches is not None:
                    kwargs["data_iterator"] = [iter(static_batches)]
                captured_result = self.forward_backward_func(**kwargs)

            # Save the captured result reference — graph replay updates THESE tensors
            if isinstance(captured_result, list):
                self._graph_result = captured_result
                # Save persistent references to graph's loss output tensors.
                # These addresses are fixed — graph replay writes new values here.
                self._graph_loss_tensors = []
                for _cr in captured_result:
                    if isinstance(_cr, dict) and "loss" in _cr:
                        self._graph_loss_tensors.append(_cr["loss"])

        # Defensive zero-grad between capture and eager re-run.
        # In practice, CUDA graph capture leaves param.grad at zero and DDP's
        # backward hook is skipped during capture (is_graph_capturing() -> return),
        # so main_grad is not polluted. This zero_grad is kept as a safety net
        # in case future PyTorch/Megatron changes alter that behavior.
        self._zero_grad_buffers(kwargs.get("model"))

        # Run one eager step with same data to produce correct output for this iter
        if static_batches is not None:
            kwargs["data_iterator"] = [iter(static_batches)]
        result = self.forward_backward_func(**kwargs)

        self.captured = True
        self._needs_recapture = False
        logger.info(f"[GrootFullIterationGraph] CUDA graph captured for {training_str}")

        return result

    def _on_buffer_realloc(self, error, data_iter, num_microbatches, kwargs):
        """After a buffer realloc, consume remaining slots from the original
        iterator so they don't go stale, then re-enter using the static
        buffers as the new iterator source."""
        import re as _re
        # Parse the slot index from the error to know how many batches
        # were already consumed from the iterator (slots 0..slot_idx).
        _m = _re.search(r"at slot (\d+)", str(error))
        consumed_count = int(_m.group(1)) + 1 if _m else num_microbatches
        # Continue consuming remaining slots from original iterator to avoid stale data
        iterator0 = data_iter[0] if isinstance(data_iter, list) else data_iter
        if iterator0 is not None:
            for remaining_slot in range(consumed_count, num_microbatches):
                try:
                    batch = next(iterator0)
                    self.static_loader._buffers[remaining_slot] = copy_tensors_in_struct(batch)
                except StopIteration:
                    break
        new_kwargs = dict(kwargs)
        if self.static_loader._buffers:
            new_kwargs["data_iterator"] = [iter(self.static_loader._buffers)]
        return new_kwargs

    def _run_replay(self, num_microbatches, kwargs):
        training_str = kwargs.get("training_str", "train")
        training_str_key = training_str + "_capture"

        # Load new data into the SAME static buffers (preserves captured memory addresses)
        # and detect any graph-invalidation trigger.
        _, recapture_kwargs = self._load_replay_data_or_recapture(num_microbatches, kwargs)
        if recapture_kwargs is not None:
            return self.__call__(**recapture_kwargs)

        # Replay the captured graph (reads from static buffers which now contain new data).
        with self._suppress_loss_allreduce():
            self.cuda_graph[training_str_key].replay()
            torch.cuda.synchronize()

        # Read loss from the graph's STATIC output tensors (fixed addresses),
        # NOT from a stale dict reference (which may have been overwritten by
        # previous allreduce results).
        if self._graph_loss_tensors:
            self._allreduce_and_update_losses(self._graph_result)

        return self._graph_result if self._graph_result else []
