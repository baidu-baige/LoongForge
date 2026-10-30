# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pin-memory hook for DP balance.

Provides a lightweight wrapper around PyTorch's _pin_memory_loop to inject
cross-DP data reordering logic, without forking the entire DataLoader.
"""

from functools import wraps
from megatron.core import mpu
from megatron.training import get_args

from torch._utils import ExceptionWrapper

from loongforge.data.dp_balance.rebalance.balance import (
    reorder_data_across_dp,
)


class _ResortQueueProxy:
    """Proxy for out_queue that applies resort_dataset_fn before putting data.

    The original _pin_memory_loop writes pinned data to out_queue via put().
    This proxy intercepts put() to apply cross-DP data reordering before the
    data reaches the main thread.

    Uses identity check (``is``) to avoid re-running resort on retries, since
    _pin_memory_loop retries put() in a while loop on queue.Full.
    """

    def __init__(self, queue):
        """Initialize the proxy with the original output queue.

        Args:
            queue: The original out_queue used by _pin_memory_loop.
        """
        self._queue = queue
        self._last_input = None
        self._last_output = None

    def put(self, item, **kwargs):
        """Intercept put() to apply cross-DP data reordering before enqueueing.

        Uses identity check (``is``) against _last_input to detect retries from
        _pin_memory_loop's while-loop on queue.Full, so resort_dataset_fn is
        only invoked once per unique item.

        Args:
            item: A tuple of (idx, data) or (idx, ExceptionWrapper).
            **kwargs: Extra keyword arguments forwarded to the real queue.put().
        """
        if item is not self._last_input:
            self._last_input = item
            idx, data = item
            if not isinstance(data, ExceptionWrapper):
                self._last_output = resort_dataset_fn(idx, data)
            else:
                self._last_output = item
        self._queue.put(self._last_output, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attribute accesses to the underlying queue.

        This ensures the proxy is a transparent drop-in replacement for the
        original out_queue, forwarding any method or property not explicitly
        overridden (e.g., qsize, empty, full).

        Args:
            name: The attribute name to look up on the underlying queue.

        Returns:
            The corresponding attribute from the wrapped queue.
        """
        return getattr(self._queue, name)


def pin_memory_loop_wrapper(original_pin_memory_loop):
    """Wrap PyTorch's _pin_memory_loop to inject DP balance resort logic.

    Instead of forking the entire DataLoader to thread a resort_fn parameter,
    this wrapper replaces the out_queue with a _ResortQueueProxy that
    transparently applies cross-DP data reordering after pin_memory.
    """
    @wraps(original_pin_memory_loop)
    def wrapper(in_queue, out_queue, device_id, done_event, device):
        """Replace out_queue with a _ResortQueueProxy and delegate to the original loop.

        Args:
            in_queue: Queue supplying (idx, data) items from worker processes.
            out_queue: Original output queue; wrapped by _ResortQueueProxy to
                inject cross-DP reordering before data reaches the main thread.
            device_id: Target CUDA device id for pin_memory.
            done_event: Threading event signalling the loop to stop.
            device: Target torch.device for pin_memory.
        """
        resort_queue = _ResortQueueProxy(out_queue)
        return original_pin_memory_loop(
            in_queue, resort_queue, device_id, done_event, device
        )

    return wrapper


def resort_dataset_fn(idx, data):
    """Reorder dataset with support for data parallel balancing.

    Determines whether to perform cross-data-parallel reordering of data based
    on training parameters and model parallel state. Reordering is only executed
    when specific conditions are met.

    Caches static conditions (use_dp, is_tp0, is_pp0, warmup_end) on first call
    to avoid redundant lookups per batch.

    Args:
        idx: Data index.
        data: Original data.

    Returns:
        tuple: Returns original index and potentially reordered data.
    """
    state = resort_dataset_fn._state
    if state is None:
        args = get_args()
        state = {
            "active": (
                args.use_vlm_dp_balance
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            ),
            "warmup_end": args.vlm_dp_balance_warmup_iters[-1] + 1,
            "warmup_done": False,
        }
        resort_dataset_fn._state = state

    if state["active"]:
        if state["warmup_done"]:
            data = reorder_data_across_dp(data)
        else:
            args = get_args()
            if args.curr_iteration > state["warmup_end"]:
                state["warmup_done"] = True
                data = reorder_data_across_dp(data)
    return idx, data

resort_dataset_fn._state = None
