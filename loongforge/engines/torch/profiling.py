# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Profiler — torch.profiler or nsys, unified class."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class Profiler:
    """Unified profiler covering torch.profiler and nsys (cudaProfilerApi).

    Mode is chosen at construction time:
      - training_args.use_pytorch_profiler  -> torch.profiler with tensorboard_trace_handler
      - training_args.use_nsys_profiler     -> cudaProfilerStart/Stop + emit_nvtx (nsys)
    Both honor --profile-step-start / --profile-step-end / --profile-ranks.

    Non-profile ranks (or when profiling is disabled) get a no-op instance.
    """

    __slots__ = ("training_args", "ctx", "output_dir", "mode", "_prof", "_nvtx_ctx",
                 "_started", "_active")

    def __init__(self, training_args, ctx, output_dir: str):
        self.training_args = training_args
        self.ctx = ctx
        self.output_dir = output_dir
        self._prof = None
        self._nvtx_ctx = None
        self._started = False

        rank = ctx.rank if ctx is not None else 0
        in_ranks = rank in training_args.profile_ranks
        if training_args.use_pytorch_profiler and in_ranks:
            self.mode = "pytorch"
        elif training_args.use_nsys_profiler and in_ranks:
            self.mode = "nsys"
        else:
            self.mode = "off"
        self._active = self.mode != "off"

    @property
    def is_active(self) -> bool:
        """Is profiling active on this rank"""
        return self._active

    def start(self):
        """Build the underlying profiler and start it (pytorch mode).

        For nsys, cudaProfilerStart is deferred to step() so the captured
        range begins precisely at completed_steps == profile_step_start.
        """
        import torch

        if not self._active:
            return

        training_args = self.training_args
        start = max(training_args.profile_step_start, 0)
        end = max(training_args.profile_step_end, start)

        if self.mode == "nsys":
            logger.info(
                f"nsys profiling enabled on profile_ranks={training_args.profile_ranks}: "
                f"steps [{start}, {end})"
            )
            return

        # pytorch mode
        active = max(end - start, 1)
        trace_dir = training_args.profile_output_dir or os.path.join(self.output_dir, "profiler")
        os.makedirs(trace_dir, exist_ok=True)

        self._prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=max(start - 1, 0),
                warmup=1 if start > 0 else 0,
                active=active,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=True,
            with_stack=True,
        )
        self._prof.start()
        self._started = True
        logger.info(
            f"torch.profiler enabled on profile_ranks={training_args.profile_ranks}: "
            f"steps [{start}, {end}), trace_dir={trace_dir}"
        )

    def step(self, completed_steps: int):
        """Per-iteration tick."""
        import torch

        if not self._active:
            return
        if self.mode == "pytorch":
            if self._prof is not None:
                self._prof.step()
        else:  # nsys
            if not self._started and completed_steps == self.training_args.profile_step_start:
                torch.cuda.cudart().cudaProfilerStart()
                self._nvtx_ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True)
                self._nvtx_ctx.__enter__()
                self._started = True

    def should_stop(self, completed_steps: int) -> bool:
        """True iff this step is profile_step_end on a profiling rank."""
        if not self._active:
            return False
        return completed_steps == self.training_args.profile_step_end

    def stop(self):
        """Stop the active profiler (idempotent)."""
        import torch

        if not self._active or not self._started:
            return
        if self.mode == "pytorch":
            self._prof.stop()
        else:  # nsys
            torch.cuda.cudart().cudaProfilerStop()
            if self._nvtx_ctx is not None:
                self._nvtx_ctx.__exit__(None, None, None)
                self._nvtx_ctx = None
        self._started = False
