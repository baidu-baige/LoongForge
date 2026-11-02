# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU offload utilities for full hetero DP memory optimization.

Provides async GPU-to-CPU and CPU-to-GPU tensor transfers using pinned memory
and dedicated CUDA streams, allowing encoder embeddings and decoder gradients
to be temporarily stored on the host while the decoder runs.
"""

from typing import Dict, Optional, Tuple

import torch


class CpuOffloadManager:
    """Manages async GPU<->CPU tensor transfers via pinned memory and CUDA streams.

    Tensors are offloaded after encoder forward and reloaded before encoder
    backward.  Using a dedicated stream ensures transfers can overlap with
    decoder computation.
    """

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._offload_stream: Optional[torch.cuda.Stream] = None
        self._reload_stream: Optional[torch.cuda.Stream] = None
        # key -> (cpu_tensor, shape, dtype, device)
        self._store: Dict[str, Tuple[torch.Tensor, torch.Size, torch.dtype, torch.device]] = {}
        self._offload_events: Dict[str, torch.cuda.Event] = {}
        self._reload_events: Dict[str, torch.cuda.Event] = {}

    @property
    def enabled(self) -> bool:
        """Whether CPU offload is enabled."""
        return self._enabled

    def _ensure_streams(self) -> None:
        if self._offload_stream is None:
            self._offload_stream = torch.cuda.Stream()
        if self._reload_stream is None:
            self._reload_stream = torch.cuda.Stream()

    def offload(self, key: str, tensor: torch.Tensor) -> None:
        """Async copy *tensor* from GPU to pinned CPU memory.

        After this call returns the GPU tensor can be freed – the D2H copy is
        queued on a background stream.
        """
        if not self._enabled or tensor is None:
            return
        self._ensure_streams()

        cpu_tensor = torch.empty(
            tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
        )

        # Make sure the tensor is ready on the current stream before copying.
        ready_event = torch.cuda.current_stream().record_event()

        with torch.cuda.stream(self._offload_stream):
            self._offload_stream.wait_event(ready_event)
            cpu_tensor.copy_(tensor, non_blocking=True)
            event = self._offload_stream.record_event()

        self._store[key] = (cpu_tensor, tensor.shape, tensor.dtype, tensor.device)
        self._offload_events[key] = event

    def offload_sync(self, key: str) -> None:
        """Block until a specific offload finishes."""
        if key in self._offload_events:
            self._offload_events[key].synchronize()

    def reload(self, key: str) -> Optional[torch.Tensor]:
        """Async copy a tensor back from CPU to GPU.

        Returns the destination GPU tensor.  The caller **must** call
        :meth:`reload_sync` (or synchronize the current stream) before
        reading the tensor contents.
        """
        if not self._enabled or key not in self._store:
            return None
        self._ensure_streams()

        cpu_tensor, shape, dtype, device = self._store[key]

        # Ensure the offload finished before we read the CPU buffer.
        if key in self._offload_events:
            self._offload_events[key].synchronize()
            del self._offload_events[key]

        gpu_tensor = torch.empty(shape, dtype=dtype, device=device)

        with torch.cuda.stream(self._reload_stream):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)
            event = self._reload_stream.record_event()

        self._reload_events[key] = event

        # Release CPU storage eagerly.
        del self._store[key]

        return gpu_tensor

    def reload_sync(self, key: str) -> None:
        """Block until a specific reload finishes."""
        if key in self._reload_events:
            self._reload_events[key].synchronize()
            del self._reload_events[key]

    def wait_all_offloads(self) -> None:
        """Block until every pending offload has completed."""
        for event in self._offload_events.values():
            event.synchronize()
        self._offload_events.clear()

    def wait_all_reloads(self) -> None:
        """Block until every pending reload has completed."""
        for event in self._reload_events.values():
            event.synchronize()
        self._reload_events.clear()

    def clear(self) -> None:
        """Release **all** stored CPU tensors and pending events."""
        self.wait_all_offloads()
        self.wait_all_reloads()
        self._store.clear()
        self._offload_events.clear()
        self._reload_events.clear()


# ---------------------------------------------------------------------------
# Convenience helpers for working with lists of tensors
# ---------------------------------------------------------------------------

def offload_list_items(
    manager: CpuOffloadManager,
    items: list,
    prefix: str,
) -> None:
    """Offload every non-``None`` item in *items* and set slots to ``None``.

    Keys are formatted as ``'{prefix}_{index}'``.
    """
    if not manager.enabled:
        return
    for i, item in enumerate(items):
        if item is not None:
            manager.offload(f"{prefix}_{i}", item)
            items[i] = None


def reload_list_item(
    manager: CpuOffloadManager,
    items: list,
    index: int,
    prefix: str,
) -> Optional[torch.Tensor]:
    """Reload a single item from CPU back into *items[index]*.

    Returns the reloaded tensor (also stored back into the list).
    """
    if not manager.enabled:
        return items[index]
    key = f"{prefix}_{index}"
    tensor = manager.reload(key)
    if tensor is not None:
        manager.reload_sync(key)
        items[index] = tensor
    return items[index]
