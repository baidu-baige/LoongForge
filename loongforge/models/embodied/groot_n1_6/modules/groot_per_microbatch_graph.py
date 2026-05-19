# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GrootPerMicrobatchGraph: Per-microbatch CUDA graph with externalized RNG.

Captures each microbatch's forward+backward as an independent CUDA graph.
Before each sub-graph replay, noise and time are generated eagerly using
Beta.sample(), achieving bit-exact RNG alignment with pure eager mode.

Architecture per iteration replay:
  for each microbatch i:
    1. Generate noise_i + time_i EAGERLY (torch.randn + Beta.sample)
    2. copy_() into static buffers
    3. sub_graph_i.replay()  (forward + backward for microbatch i)
  4. finalize_model_grads (gradient allreduce)
  5. allreduce loss

This preserves the eager RNG consumption order:
  noise0 -> time0 -> [fwd0+bwd0] -> noise1 -> time1 -> [fwd1+bwd1] -> ...

Controlled by: --cuda-graph-impl=local --cuda-graph-scope=per_microbatch
"""
from __future__ import annotations

import contextlib
import logging

import torch
from megatron.training.utils import unwrap_model

from loongforge.models.common.cuda_graph_base import (
    BaseCudaGraphWrapper,
    StaticBufferLoader,
)

from .groot_graph_mixin import GrootGraphMixin

logger = logging.getLogger(__name__)


class GrootPerMicrobatchGraph(GrootGraphMixin, BaseCudaGraphWrapper):
    """Per-microbatch CUDA graph with externalized RNG for bit-exact training.

    Each microbatch gets its own CUDA graph. Before each sub-graph replay,
    noise/time are generated eagerly so the RNG sequence matches pure eager
    exactly (noise_i -> time_i -> dropout_i -> backward_i -> ...).
    """

    LOG_TAG = "GrootPerMicrobatchGraph"
    # Submodules (e.g. ViT) flip this on during warmup when they see padding;
    # must be cleared on graph invalidation to allow re-capture.
    _warmup_sentinel_attrs = ("_capture_has_invalid_images",)

    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=3):
        super().__init__(forward_backward_func, cuda_graph_warmup_steps)

        # Per-microbatch CUDA graphs
        self._sub_graphs: list[torch.cuda.CUDAGraph] = []

        # Static RNG buffers per microbatch
        self._noise_bufs: list[torch.Tensor] = []
        self._time_bufs: list[torch.Tensor] = []

        # Per-microbatch loss output references
        self._sub_graph_results: list[list] = []

        # Cached action_head reference
        self._action_head = None

    # ------------------- action_head plumbing -------------------------
    def _find_action_head(self):
        """Find the action_head module in the model."""
        if self._action_head is not None:
            return self._action_head
        if self._model_ref is None:
            return None
        for _mc in self._model_ref:
            model = unwrap_model(_mc)
            for _name, mod in model.named_modules():
                if hasattr(mod, 'action_encoder') and hasattr(mod, 'sample_time'):
                    self._action_head = mod
                    return mod
        return None

    def _enable_shape_recording(self):
        ah = self._find_action_head()
        if ah is not None:
            ah._split_record_shape = True

    def _disable_shape_recording(self):
        ah = self._find_action_head()
        if ah is not None:
            ah._split_record_shape = False

    # ------------------- RNG buffer management ------------------------
    def _allocate_rng_bufs(self, num_microbatches):
        """Allocate static noise/time buffers using shape recorded during warmup."""
        ah = self._find_action_head()
        if ah is None:
            raise RuntimeError("[GrootPerMicrobatchGraph] Cannot find action_head")

        actions_shape = getattr(ah, '_split_actions_shape', None)
        if actions_shape is None:
            raise RuntimeError(
                "[GrootPerMicrobatchGraph] Action shape not recorded during warmup."
            )
        device = ah._split_actions_device
        dtype = ah._split_actions_dtype

        self._noise_bufs = []
        self._time_bufs = []

        for mb_idx in range(num_microbatches):
            noise_buf = torch.randn(actions_shape, device=device, dtype=dtype)
            batch_size = actions_shape[0]
            t = ah.beta_dist.sample([batch_size]).to(device, dtype=dtype)
            time_buf = (1 - t) * ah.config.noise_s
            self._noise_bufs.append(noise_buf)
            self._time_bufs.append(time_buf)

    def _set_rng_buf_on_model(self, mb_idx):
        """Set the static buffer for a specific microbatch on the action_head."""
        ah = self._find_action_head()
        if ah is not None:
            ah._split_noise_buf = self._noise_bufs[mb_idx]
            ah._split_time_buf = self._time_bufs[mb_idx]

    def _clear_rng_bufs_on_model(self):
        ah = self._find_action_head()
        if ah is not None:
            ah._split_noise_buf = None
            ah._split_time_buf = None

    def _eager_rng_single(self, mb_idx):
        """Generate fresh noise/time for a single microbatch and copy into buffer."""
        ah = self._find_action_head()
        if ah is None:
            return

        noise_buf = self._noise_bufs[mb_idx]
        time_buf = self._time_bufs[mb_idx]

        # Fresh noise (consumes default generator offset)
        noise = torch.randn_like(noise_buf)
        noise_buf.copy_(noise)

        # Fresh time via Beta.sample (consumes default generator offset)
        batch_size = time_buf.shape[0]
        t = ah.beta_dist.sample([batch_size]).to(
            time_buf.device, dtype=time_buf.dtype
        )
        t = (1 - t) * ah.config.noise_s
        time_buf.copy_(t)

    # ------------------- BaseCudaGraphWrapper hooks -------------------
    def _before_warmup_forward(self):
        self._enable_shape_recording()

    def _after_warmup_forward(self):
        self._disable_shape_recording()

    @staticmethod
    def _issue_grad_sync(model_list):
        """Manually dispatch grad reduce after replay.

        Graph replay only re-executes recorded kernels — Python autograd hooks
        do not fire — so DDP's overlap path never gets a chance to call
        ``start_grad_sync()`` on its own. We do it here so the subsequent
        ``finalize_model_grads_func`` can ``finish_grad_sync()`` (wait on the
        handle).
        """
        if model_list is None:
            return
        for mc in model_list:
            if hasattr(mc, "start_grad_sync"):
                mc.start_grad_sync()

    @staticmethod
    def _is_overlap_grad_reduce(model_list):
        if not model_list:
            return False
        cfg = getattr(model_list[0], "ddp_config", None)
        return bool(cfg is not None and getattr(cfg, "overlap_grad_reduce", False))

    @staticmethod
    def _collect_bucket_groups(model_list):
        bgs = []
        if not model_list:
            return bgs
        for mc in model_list:
            bgs.extend(getattr(mc, "bucket_groups", []))
            bgs.extend(getattr(mc, "expert_parallel_bucket_groups", []))
        return bgs

    @contextlib.contextmanager
    def _suppress_grad_sync(self, model_list):
        """Pin DDP and schedule state so capture-time backwards don't fire
        ``start_grad_sync()`` and the schedule's ``no_sync`` toggling is a
        no-op. Restored on exit.
        """
        saved_finalize = self._config.finalize_model_grads_func
        saved_no_sync = self._config.no_sync_func
        bgs = self._collect_bucket_groups(model_list)
        saved_flags = [getattr(bg, "is_last_microbatch", True) for bg in bgs]
        self._config.finalize_model_grads_func = None
        self._config.no_sync_func = contextlib.nullcontext
        for bg in bgs:
            bg.is_last_microbatch = False
        try:
            yield
        finally:
            self._config.finalize_model_grads_func = saved_finalize
            self._config.no_sync_func = saved_no_sync
            for bg, prev in zip(bgs, saved_flags):
                bg.is_last_microbatch = prev

    def _capture_one_microbatch(self, mb_idx, kwargs, static_batches):
        """Capture a single microbatch's forward+backward into a CUDAGraph."""
        self._set_rng_buf_on_model(mb_idx)

        mb_kwargs = dict(kwargs)
        mb_kwargs["num_microbatches"] = 1
        if static_batches is not None:
            mb_kwargs["data_iterator"] = [iter([static_batches[mb_idx]])]

        cg = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()

        with torch.cuda.graph(cg):
            if static_batches is not None:
                mb_kwargs["data_iterator"] = [iter([static_batches[mb_idx]])]
            captured_result = self.forward_backward_func(**mb_kwargs)

        self._sub_graphs.append(cg)

        if isinstance(captured_result, list):
            self._sub_graph_results.append(captured_result)
            for _cr in captured_result:
                if isinstance(_cr, dict) and "loss" in _cr:
                    self._graph_loss_tensors.append(_cr["loss"])

    def _invalidate_graph_state(self):
        """Drop sub-graph handles, RNG buffers, and reset static loader.
        Common state (captured/_call_count/etc.) is reset by base class."""
        self._sub_graphs.clear()
        self._sub_graph_results = []
        self._noise_bufs = []
        self._time_bufs = []
        self.static_loader = StaticBufferLoader()
        self._clear_rng_bufs_on_model()

    def _run_capture(self, num_microbatches, kwargs):
        data_iter = kwargs.get("data_iterator")
        static_batches = None
        if data_iter is not None:
            static_batches = self.data_read(data_iter, num_microbatches)

        _model = kwargs.get("model")
        self._sub_graphs = []
        self._sub_graph_results = []
        self._graph_loss_tensors = []

        # Detect overlap_grad_reduce mode. When enabled, we let the schedule
        # run normally for the LAST sub-graph so that DDP's backward hooks
        # fire, ``start_grad_sync()`` records its NCCL kernels into the
        # graph, and ``finalize_model_grads_func`` records ``cm.wait()`` —
        # giving us the same overlap window full_iteration enjoys.
        _model_list = _model if isinstance(_model, list) else (
            [_model] if _model is not None else None
        )
        bake_grad_sync = (
            self._is_overlap_grad_reduce(_model_list) and num_microbatches > 0
        )

        try:
            with self._preserve_rng_state(), self._suppress_loss_allreduce():
                self._allocate_rng_bufs(num_microbatches)

                # Suppressed range: [0, last) when baking; otherwise all.
                last_idx = num_microbatches - 1
                n_suppressed = last_idx if bake_grad_sync else num_microbatches

                with self._suppress_grad_sync(_model_list):
                    for mb_idx in range(n_suppressed):
                        self._capture_one_microbatch(mb_idx, kwargs, static_batches)

                # Capture last sub-graph OUTSIDE the suppression context so
                # the schedule's own no_sync/finalize flow runs inside
                # torch.cuda.graph — recording NCCL kernels.
                if bake_grad_sync:
                    self._capture_one_microbatch(last_idx, kwargs, static_batches)
        finally:
            self._clear_rng_bufs_on_model()

        # Remember whether grad sync is baked into the last sub-graph so
        # _run_replay knows to skip the eager grad-sync dispatch.
        self._grad_sync_in_graph = bake_grad_sync

        # Zero grad and run eager for correct output this iteration
        self._zero_grad_buffers(_model)

        kwargs = self._with_static_data(kwargs, static_batches)
        result = self.forward_backward_func(**kwargs)

        self.captured = True
        self._needs_recapture = False
        logger.info(
            f"[GrootPerMicrobatchGraph] Captured {num_microbatches} per-microbatch "
            f"sub-graphs for bit-exact RNG alignment"
        )
        return result

    def _run_replay(self, num_microbatches, kwargs):
        # Load new data into static buffers and detect any graph-invalidation trigger.
        _, recapture_kwargs = self._load_replay_data_or_recapture(num_microbatches, kwargs)
        if recapture_kwargs is not None:
            return self.__call__(**recapture_kwargs)

        # Per-microbatch replay with interleaved RNG.
        # The trailing ``torch.cuda.synchronize()`` is intentional: the
        # subsequent optimizer.step must read reduced grads, so the host
        # has to wait for the captured NCCL allreduce regardless. Letting
        # the host run ahead just shifts the wait into the next iter's
        # timer (and we measured it as a small regression).
        try:
            with self._suppress_loss_allreduce():
                for mb_idx in range(num_microbatches):
                    # Generate noise/time eagerly (same RNG order as eager)
                    self._eager_rng_single(mb_idx)
                    # Set buffer on model for this sub-graph
                    self._set_rng_buf_on_model(mb_idx)
                    # Replay this microbatch's graph
                    self._sub_graphs[mb_idx].replay()
                torch.cuda.synchronize()
        finally:
            self._clear_rng_bufs_on_model()

        # Finalize model grads (gradient allreduce across DP).
        # Three paths:
        # (a) overlap + grad sync baked into last sub-graph: replay already
        #     re-launches the NCCL kernels and the captured cm.wait().
        # (b) overlap, grad sync NOT in graph: manually start_grad_sync,
        #     then finalize finishes it (waits on the handle).
        # (c) no overlap: finalize runs the synchronous reduce.
        if self._config is not None and self._config.finalize_model_grads_func is not None:
            _model = kwargs.get("model")
            if _model is not None:
                model_list = _model if isinstance(_model, list) else [_model]
                _overlap = self._is_overlap_grad_reduce(model_list)
                _baked = getattr(self, "_grad_sync_in_graph", False)
                if not (_overlap and _baked):
                    if _overlap:
                        self._issue_grad_sync(model_list)
                    self._config.finalize_model_grads_func(model_list, None)

        # Post-replay: allreduce loss across flattened sub-graph results
        if self._graph_loss_tensors:
            flat_results = [_r for mb_results in self._sub_graph_results for _r in mb_results]
            self._allreduce_and_update_losses(flat_results)

        # Return flattened results
        all_results = []
        for mb_results in self._sub_graph_results:
            all_results.extend(mb_results)
        return all_results if all_results else []
