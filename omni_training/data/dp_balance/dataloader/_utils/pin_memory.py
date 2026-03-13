# Copyright 2026 The OmniTraining Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from PyTorch under the BSD 3-Clause License.
# Copyright (c) 2016-present, Facebook, Inc and respective contributors.


r"""Contains definitions of the methods used by the _BaseDataLoaderIter to put fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import collections
import copy
import queue

import torch
from torch._utils import ExceptionWrapper

from . import MP_STATUS_CHECK_INTERVAL


def pin_memory_loop(in_queue, out_queue, device_id, done_event, resort_fn, device):
    """
    Perform data pinning operations in a separate thread loop

    This function serves as the memory pinning worker thread for data loaders.
    It is responsible for getting data from the input queue, pinning it to the
    specified device memory, and then putting it into the output queue for use
    by the main thread.

    Args:
        in_queue: Input queue containing data and indices that need memory pinning
        out_queue: Output queue for storing pinned data
        device_id: Device ID for setting current thread's device context
        done_event: Completion event for controlling thread exit
        resort_fn: Optional resort function for handling data indices
        device: Device type ('cuda', 'xpu' or custom backend)

    Note:
        - Thread-local setting torch.set_num_threads(1) prevents memory pinning operations
          from occupying too many CPU cores
        - Supports multiple device types including CUDA, XPU and custom backends
        - Uses exception wrapping mechanism to handle errors during memory pinning
        - Follows data loader multi-process shutdown logic to ensure safe exit
    """
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.multiprocessing._set_thread_name("pt_data_pin")

    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "xpu":
        torch.xpu.set_device(device_id)  # type: ignore[attr-defined]
    elif device == torch._C._get_privateuse1_backend_name():
        custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
        custom_device_mod.set_device(device_id)

    def do_one_step():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                data = pin_memory(data, device)
            except Exception:
                data = ExceptionWrapper(
                    where=f"in pin memory thread for device {device_id}"
                )
            r = (idx, data)
            if resort_fn is not None and not isinstance(data, ExceptionWrapper):
                r = resort_fn(idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        # Make sure that we don't preserve any object from one iteration
        # to the next
        do_one_step()


def pin_memory(data, device=None):
    """
    Pin input data or its elements to memory to accelerate GPU data transfer.

    This method recursively handles various data types including:
    - torch.Tensor: Directly calls its pin_memory method
    - String/bytes: Returns as is
    - Mapping types (dict, etc.): Recursively pins all values
    - Named tuples: Preserves type unchanged
    - Sequence types (list, etc.): Recursively pins all elements
    - Custom types: Calls pin_memory method if implemented

    Args:
        data: Input data to be pinned to memory, supports multiple types
        device: Optional target device, passed to torch.Tensor.pin_memory

    Returns:
        Memory-pinned data, preserving original data structure

    Raises:
        TypeError: If input data type is not supported and doesn't implement pin_memory method
    """
    if isinstance(data, torch.Tensor):
        return data.pin_memory(device)
    elif isinstance(data, (str, bytes)):
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(data)
                clone.update(
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )
                return clone
            else:
                return type(data)(
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {k: pin_memory(sample, device) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(*(pin_memory(sample, device) for sample in data))
    elif isinstance(data, tuple):
        return [
            pin_memory(sample, device) for sample in data
        ]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(data)  # type: ignore[arg-type]
                for i, item in enumerate(data):
                    clone[i] = pin_memory(item, device)
                return clone
            return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `copy()` / `__setitem__(index, item)`
            # or `__init__(iterable)` (e.g., `range`).
            return [pin_memory(sample, device) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data
