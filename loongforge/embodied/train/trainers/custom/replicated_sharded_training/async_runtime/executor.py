# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Side-stream CUDA execution, decoupled from what is being computed.

This is the distillation-teacher runner generalised: the stream/event/thread
discipline it encodes is not specific to teachers, only to "run a side-effect-free
CUDA function concurrently with the step and hand the outputs back safely".

Correctness notes, all load-bearing and all learned the hard way:
- ``current stream``, ``no_grad`` and ``autocast`` are thread-local in PyTorch, so
  the work must be submitted from inside the worker thread rather than configured
  on the caller side.
- Inputs are produced on the caller's stream, so the worker waits on an event
  recorded there, and the inputs are marked with ``record_stream`` so the caching
  allocator cannot recycle them while the side stream still reads them.
- Outputs are allocated on the side stream, so ``result()`` marks them with
  ``record_stream`` on the consuming stream for the same reason. Skipping this
  does not crash, it silently corrupts the outputs.
- Do NOT set ``torch._dynamo.config.error_on_nested_fx_trace = False`` to dodge the
  cross-thread FX race. Dynamo then falls back to the *eager* kernel for the
  colliding call: measured on 8 GPUs the same-step configuration regressed
  1647.6 -> 2003.2 ms (+21.6%) and stopped reproducing across iterations. A rare
  loud exception with a caught recompute beats a frequent silent downgrade.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque

import torch

logger = logging.getLogger(__name__)

# ``fallback_to_sync`` recomputes on the calling thread, so it is only valid for
# work that is side-effect-free and safe to repeat.
ERROR_POLICIES = ("raise", "fallback_to_sync")


def iter_tensors(obj):
    """Yield every tensor reachable through dicts, lists and tuples."""
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_tensors(value)


class AsyncTaskError(RuntimeError):
    """The submitted work raised inside the worker thread."""


class CudaTaskHandle:
    """Pending side-stream output. ``result()`` is idempotent."""

    def __init__(self, executor, token, fn, args, kwargs):
        self._executor = executor
        self._token = token
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._resolved = None
        self._failed = False

    @property
    def token(self) -> int:
        return self._token

    def done(self) -> bool:
        """True when the output can be taken without blocking."""
        if self._resolved is not None or self._failed:
            return True
        return self._executor._is_ready(self._token)

    def wait(self):
        """Block until the output is available; same return value as ``result``."""
        return self.result()

    def result(self):
        """Return the output, applying the executor's error policy on failure."""
        if self._resolved is not None:
            return self._resolved
        self._resolved = self._executor._resolve(
            self._token, self._fn, self._args, self._kwargs
        )
        return self._resolved


class CudaStreamExecutor:
    """One long-lived worker thread bound to one side stream.

    Responses are FIFO because the worker is a single thread, and that order is
    asserted rather than assumed.
    """

    def __init__(
        self,
        device=None,
        timeout=300,
        error_policy: str = "raise",
        name: str = "loongforge-async-cuda",
    ):
        if error_policy not in ERROR_POLICIES:
            raise ValueError(
                f"unsupported error_policy={error_policy!r}; expected one of "
                f"{ERROR_POLICIES}"
            )
        # torch.cuda.set_device requires an explicit index and the worker thread
        # cannot resolve a bare "cuda" itself, so pin the concrete index here on
        # the calling thread (which the launcher has already set to the local rank).
        resolved = torch.device(device if device is not None else "cuda")
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self._device = resolved
        self._timeout = timeout
        self._error_policy = error_policy
        self._startup_error = None
        self._stream = torch.cuda.Stream(device=self._device)
        self._requests: "queue.Queue" = queue.Queue()
        self._responses: "queue.Queue" = queue.Queue()
        self._outstanding: deque = deque()
        self._ready: dict = {}
        self._token = 0
        self._closed = False
        self._fallbacks = 0
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    # -- worker ---------------------------------------------------------------

    def _loop(self):
        try:
            torch.cuda.set_device(self._device)
        except BaseException as exc:
            # A dead worker used to leave every caller blocked on the response
            # queue, which stalls the whole job and every other rank behind it.
            # Instead, answer every request with the startup failure so it
            # surfaces at once.
            self._startup_error = exc
            logger.exception("async cuda worker failed to start")
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
            except BaseException as exc:  # surfaced to the caller in _take
                self._responses.put((token, None, None, exc))

    # -- submission -----------------------------------------------------------

    def submit(self, fn, *args, **kwargs) -> CudaTaskHandle:
        """Queue ``fn`` on the side stream and return a handle to its output."""
        if self._closed:
            raise RuntimeError("CudaStreamExecutor is closed")
        return self._submit_locked(fn, args, kwargs)

    def _submit_locked(self, fn, args, kwargs) -> CudaTaskHandle:
        """Record the input barrier, keep the inputs alive, enqueue the work."""
        caller_stream = torch.cuda.current_stream(self._device)
        input_ready = torch.cuda.Event()
        input_ready.record(caller_stream)
        for tensor in iter_tensors((args, kwargs)):
            if tensor.is_cuda:
                tensor.record_stream(self._stream)
        self._token += 1
        token = self._token
        self._outstanding.append(token)
        self._requests.put((token, fn, args, kwargs, input_ready))
        return CudaTaskHandle(self, token, fn, args, kwargs)

    def pending(self) -> int:
        """Number of submitted tasks whose output has not been taken yet."""
        return len(self._outstanding) + len(self._ready)

    def wait(self, handle):
        """Block for ``handle`` and return its output."""
        return handle.result()

    # -- collection -----------------------------------------------------------

    def _pump(self, block: bool):
        """Move one response into the ready map, asserting FIFO order."""
        try:
            token, out, done, exc = self._responses.get(
                block=block, timeout=self._timeout if block else None
            )
        except queue.Empty:
            if not block:
                return False
            raise RuntimeError(
                "async cuda worker produced no result within %ds; it is stuck or "
                "died." % self._timeout
            ) from None
        expected = self._outstanding.popleft() if self._outstanding else None
        if expected != token:
            raise RuntimeError(
                f"async cuda response out of order (want {expected}, got {token})"
            )
        self._ready[token] = (out, done, exc)
        return True

    def _is_ready(self, token) -> bool:
        """Non-blocking readiness probe; drains whatever the worker has answered."""
        while token not in self._ready and self._pump(block=False):
            pass
        return token in self._ready

    def _take(self, token):
        """Block until ``token`` is answered, then adopt its outputs locally."""
        while token not in self._ready:
            self._pump(block=True)
        out, done, exc = self._ready.pop(token)
        if exc is not None:
            raise AsyncTaskError("async cuda work failed on the worker") from exc
        consumer = torch.cuda.current_stream(self._device)
        consumer.wait_event(done)
        for tensor in iter_tensors(out):
            if tensor.is_cuda:
                tensor.record_stream(consumer)
        return out

    def _resolve(self, token, fn, args, kwargs):
        """Take the output, applying the configured error policy on failure."""
        try:
            return self._take(token)
        except AsyncTaskError as exc:
            if self._error_policy == "fallback_to_sync":
                # torch.fx replaces nn.Module.__call__ process-wide while the main
                # thread is tracing, and that patch is not thread-local: it can
                # catch this work mid-forward and raise. Redoing it on the calling
                # thread costs a step's overlap but keeps the job alive, which is
                # the right trade for side-effect-free work.
                self._note_fallback(exc, "recomputing on the calling thread")
                return fn(*args, **kwargs)
            raise

    def _note_fallback(self, exc, action: str) -> None:
        """Warn on the first few failures.

        A steady stream of them means the side stream is useless and the feature
        should be turned off rather than silently paying for both paths.
        """
        self._fallbacks += 1
        if self._fallbacks <= 3 or self._fallbacks % 50 == 0:
            logger.warning(
                "async cuda worker failed (%d so far), %s for this step: %r",
                self._fallbacks,
                action,
                exc.__cause__ if exc.__cause__ is not None else exc,
            )

    # -- shutdown -------------------------------------------------------------

    def drain(self) -> None:
        """Take and discard every outstanding output, leaving nothing in flight."""
        while self._outstanding:
            token = self._outstanding[0]
            try:
                self._take(token)
            except AsyncTaskError:
                pass
        self._ready.clear()

    def close(self) -> None:
        """Stop the worker thread; idempotent."""
        if self._closed:
            return
        self._closed = True
        self._outstanding.clear()
        self._ready.clear()
        self._requests.put(None)
        self._thread.join(timeout=30)


__all__ = [
    "ERROR_POLICIES",
    "AsyncTaskError",
    "CudaStreamExecutor",
    "CudaTaskHandle",
    "iter_tensors",
]
