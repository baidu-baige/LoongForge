# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Misc training utilities."""

import contextlib
import logging
import os
import random
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch


from loongforge.embodied.distributed.utils import is_rank_zero 


def resolve_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype.
    """
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    return mapping[dtype_str]


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed across all sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_precision(allow_tf32):
    """set_precision."""
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def set_deterministic():
    """Enable or disable deterministic algorithms for reproducibility.

    """
    if "PYTHONHASHSEED" not in os.environ:
        logger.warning(
            "PYTHONHASHSEED is not set; --deterministic-mode is best-effort without it. "
            "For full reproducibility, prepend `PYTHONHASHSEED=42` (or any fixed value) "
            "to your launch command ?~@~T Python's hash seed is fixed at interpreter startup "
            "and cannot be set retroactively."
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def setup_logging(output_dir: str, rank: int):
    """Configure logging: rank0 gets INFO + file handler, others get WARNING only.

    All handlers are registered in a single ``basicConfig(force=True)`` call
    so re-invocations (or prior implicit StreamHandlers) do not stack up and
    cause duplicate output.
    """
    level = logging.INFO if rank == 0 else logging.WARNING

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    handlers = [sh]

    if rank == 0 and output_dir:
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers, force=True)


@contextlib.contextmanager
def log_stage(
    tag: str,
    start_msg: str = "",
    end_msg: str = "done in {elapsed}",
    log: Optional[logging.Logger] = None,
):
    """Context manager that wraps a setup stage with start/end logs + timing.

    Emits ``[tag] start_msg`` before entering the block and
    ``[tag] end_msg`` after leaving (the ``{elapsed}`` placeholder is
    interpolated with the formatted duration, e.g. ``"3.21s"``).

    Rank-0 gating is built in: when ``torch.distributed`` is initialized and
    the current rank is not 0, the context manager runs silently.

    Example
    -------
    >>> with log_stage("model", "building", "built in {elapsed}"):
    ...     model = build_model()
    """
    out = log or logger
    is_main = is_rank_zero()

    if is_main and start_msg:
        out.info(f"[{tag}] {start_msg}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if is_main and end_msg:
            elapsed = f"{time.perf_counter() - t0:.2f}s"
            out.info(f"[{tag}] {end_msg.format(elapsed=elapsed)}")

# ═══════════════════════════════════════════════
# Profiler (torch.profiler or nsys profiler, unified class)
# ═══════════════════════════════════════════════

class Profiler:
    """Unified profiler covering torch.profiler and nsys (cudaProfilerApi).

    Mode is chosen at construction time:
      - training_args.use_pytorch_profiler  → torch.profiler with tensorboard_trace_handler
      - training_args.use_nsys_profiler     → cudaProfilerStart/Stop + emit_nvtx (nsys)
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
        # Each profile rank ensures its own trace dir. We can't use a
        # collective ctx.barrier() here because only profile ranks reach this
        # code path — calling barrier on a subset of ranks would deadlock.
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
        """Per-iteration tick.

        - pytorch: forward to prof.step().
        - nsys: at completed_steps == profile_step_start, start cudaProfiler
          and enter emit_nvtx so the captured range covers subsequent steps.
        """
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
