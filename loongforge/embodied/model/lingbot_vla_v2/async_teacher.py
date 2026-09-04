# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen distillation teachers on a side stream and worker thread.

The teachers are a pure function of the sampled images and their outputs are only
consumed by the auxiliary depth/video heads, which run after the backbone. Nothing
in the student forward depends on them, so they can execute concurrently.

Measured cost (CUDA-event probe, 20 steps, no profiler): the region takes 145.7 ms
and its device span equals its host wall, so it is device-bound with no host bubble.
Do not re-derive this from an nsys trace, which inflates host dispatch and suggests
the opposite.

Consequences of it being device-bound: the worker thread alone buys nothing, since
the work still has to occupy the device. What the side stream buys is only the
fraction of the teacher the student's own kernels fail to extract from the
bottleneck resource -- concurrency adds no capacity. Measured on this workload the
teacher's 145.7 ms costs the student forward 96.6 ms, i.e. only ~34% comes for
free, because the teacher is submitted before the forward and therefore runs
against the most tensor-core-saturated phase of the step.

Correctness notes:
- ``current stream``, ``no_grad`` and ``autocast`` are all thread-local in PyTorch,
  so the work must be submitted from inside the worker thread, not configured on
  the caller side.
- Inputs are produced on the caller's stream, so the worker waits on an event
  recorded there, and marks the inputs with ``record_stream`` so the allocator
  cannot recycle them while the side stream still reads them.
- Outputs are allocated on the side stream, so ``result()`` marks them with
  ``record_stream`` on the consuming stream for the same reason. Skipping this
  does not crash, it silently corrupts targets.
"""

from __future__ import annotations

import logging
import queue
import threading

import torch

logger = logging.getLogger(__name__)


def _iter_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensors(value)


class TeacherWorkerError(RuntimeError):
    """The submitted work raised inside the worker thread."""


class TeacherHandle:
    """Pending teacher output. ``result()`` is idempotent."""

    def __init__(self, runner, token, fn, args, kwargs):
        self._runner = runner
        self._token = token
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._resolved = None

    def result(self):
        """Return the teacher output, recomputing inline if the worker failed."""
        if self._resolved is not None:
            return self._resolved
        try:
            self._resolved = self._runner._collect(self._token)
        except TeacherWorkerError as exc:
            # torch.fx replaces nn.Module.__call__ process-wide while the main
            # thread is tracing, and that patch is not thread-local: it can catch
            # this teacher mid-forward and raise. Redoing the work on the calling
            # thread costs a step's overlap but keeps the job alive, which is the
            # right trade for a frozen, side-effect-free teacher.
            self._runner._note_fallback(exc)
            self._resolved = self._fn(*self._args, **self._kwargs)
        return self._resolved


class AsyncTeacherRunner:
    """Single long-lived worker thread bound to one side stream."""

    def __init__(self, device=None, timeout=300):
        # Do NOT set torch._dynamo.config.error_on_nested_fx_trace = False here to
        # dodge the cross-thread FX race. Dynamo then silently falls back to the
        # *eager* flex_attention for the colliding call: on 8 GPUs the same-step
        # configuration regresses 1647.6 -> 2003.2 ms (+21.6%) and loss/grad norm
        # stop reproducing across iterations. A rare loud exception with a caught
        # recompute beats a frequent silent downgrade.
        #
        # torch.cuda.set_device requires an explicit index, and the worker thread
        # cannot resolve a bare "cuda" itself, so pin the concrete index here on the
        # calling thread (which the launcher has already set to the local rank).
        resolved = torch.device(device if device is not None else "cuda")
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self._device = resolved
        self._timeout = timeout
        self._startup_error = None
        self._stream = torch.cuda.Stream(device=self._device)
        self._requests: "queue.Queue" = queue.Queue()
        self._responses: "queue.Queue" = queue.Queue()
        self._token = 0
        self._closed = False
        self._fallbacks = 0
        self._thread = threading.Thread(
            target=self._loop, name="lingbot-teacher", daemon=True
        )
        self._thread.start()

    def _loop(self):
        try:
            torch.cuda.set_device(self._device)
        except BaseException as exc:
            # A dead worker used to leave every caller blocked on the response queue,
            # which stalls the whole job and every other rank behind it. Instead,
            # answer every request with the startup failure so it surfaces at once.
            self._startup_error = exc
            logger.exception("teacher worker failed to start")
            while True:
                item = self._requests.get()
                if item is None:
                    return
                self._responses.put((item[0], None, None, exc))
        while True:
            item = self._requests.get()
            if item is None:
                return
            token, fn, args, kwargs, input_ready = item
            try:
                with torch.cuda.stream(self._stream):
                    if input_ready is not None:
                        self._stream.wait_event(input_ready)
                    out = fn(*args, **kwargs)
                    done = torch.cuda.Event()
                    done.record(self._stream)
                self._responses.put((token, out, done, None))
            except BaseException as exc:  # surfaced to the caller in _collect
                self._responses.put((token, None, None, exc))

    def submit(self, fn, *args, **kwargs) -> TeacherHandle:
        """Queue ``fn`` on the side stream and return a handle to its output."""
        if self._closed:
            raise RuntimeError("AsyncTeacherRunner is closed")
        # Let the side stream wait until the caller's stream has produced the inputs,
        # and keep those inputs alive for the side stream.
        caller_stream = torch.cuda.current_stream(self._device)
        input_ready = torch.cuda.Event()
        input_ready.record(caller_stream)
        for tensor in _iter_tensors((args, kwargs)):
            if tensor.is_cuda:
                tensor.record_stream(self._stream)
        self._token += 1
        self._requests.put((self._token, fn, args, kwargs, input_ready))
        return TeacherHandle(self, self._token, fn, args, kwargs)

    def _note_fallback(self, exc) -> None:
        """Warn on the first few fallbacks; a steady stream means the side stream is
        useless and the feature should be turned off rather than silently paying for
        both paths."""
        self._fallbacks += 1
        if self._fallbacks <= 3 or self._fallbacks % 50 == 0:
            logger.warning(
                "teacher worker failed (%d so far), recomputing on the training "
                "thread for this step: %r. Set model.async_teacher=false to stop "
                "paying for the side stream.",
                self._fallbacks,
                exc.__cause__ if exc.__cause__ is not None else exc,
            )

    def _collect(self, token):
        try:
            token_out, out, done, exc = self._responses.get(timeout=self._timeout)
        except queue.Empty:
            raise RuntimeError(
                "teacher worker produced no result within %ds; it is stuck or died. "
                "Set model.async_teacher=false to fall "
                "back to the synchronous path." % self._timeout
            ) from None
        if token_out != token:
            raise RuntimeError(
                f"teacher response out of order (want {token}, got {token_out})"
            )
        if exc is not None:
            raise TeacherWorkerError("teacher work failed on the worker") from exc
        consumer = torch.cuda.current_stream(self._device)
        consumer.wait_event(done)
        for tensor in _iter_tensors(out):
            if tensor.is_cuda:
                tensor.record_stream(consumer)
        return out

    def close(self):
        """Stop the worker thread; idempotent."""
        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        self._thread.join(timeout=30)


__all__ = ["AsyncTeacherRunner", "TeacherHandle", "TeacherWorkerError"]
