# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Full heterogeneous DP (``--enable-full-hetero-dp``): per-step state, CPU offload, strided encoder data.

Encoder ranks run the vision encoder for every microbatch of their decoder DP group, gather the
embeddings, and scatter the gradients back after the decoder backward. ``train_step``, the model's
encoder path and the EP-overlap callables share the state kept here:

- embedding / grad / visual-pos-mask / deepstack lists of the current step;
- the CPU offload manager (``--full-hetero-dp-cpu-offload``);
- the encoder data iterator, built by the method from its dataset;
- the method's batch functions, registered with ``set_batch_fns``.

``clear_full_hetero_info`` resets all of it, plus the method's forward-step state, after each step.
"""

import queue
import threading
from typing import Dict, Optional, Tuple

import torch

from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler

from loongforge.engines.megatron import get_args

embedding_list = []
grad_list = []
visual_pos_masks_list = []
deepstack_visual_embeds_list = []
deepstack_grad_list = []

_cpu_offload_manager = None
_encoder_data_iterator = None
_batch_fns = None


def set_batch_fns(get_batch, create_mock_batch, reset_step_state):
    """Register the method's batch builder, mock-batch builder and forward-step state reset."""
    global _batch_fns
    _batch_fns = (get_batch, create_mock_batch, reset_step_state)


def get_batch_fns():
    """Return ``(get_batch, create_mock_batch)`` registered by the method."""
    assert _batch_fns is not None, "full hetero DP: the training method did not call hetero_dp.set_batch_fns"
    return _batch_fns[0], _batch_fns[1]


def get_cpu_offload_manager():
    """Return the global CpuOffloadManager singleton (lazy-initialized)."""
    global _cpu_offload_manager
    if _cpu_offload_manager is None:
        args = get_args()
        _cpu_offload_manager = CpuOffloadManager(
            enabled=args.full_hetero_dp_cpu_offload
        )
    return _cpu_offload_manager

def get_encoder_data_iterator():
    """Return the encoder data iterator for full_hetero_dp mode."""
    return _encoder_data_iterator

def set_encoder_data_iterator(iterator):
    """Set the encoder data iterator for full_hetero_dp mode."""
    global _encoder_data_iterator
    _encoder_data_iterator = iterator

def get_embedding_list():
    """Return the global embedding list."""
    return embedding_list

def clear_embedding_list():
    """Clear the global embedding list."""
    embedding_list.clear()

def get_grad_list():
    """Return the global gradient list."""
    return grad_list

def clear_grad_list():
    """Clear the global gradient list."""
    grad_list.clear()

def get_visual_pos_masks_list():
    """Return the global visual position masks list."""
    return visual_pos_masks_list

def clear_visual_pos_masks_list():
    """Clear the global visual position masks list."""
    visual_pos_masks_list.clear()

def get_deepstack_visual_embeds_list():
    """Return the global deepstack visual embeddings list."""
    return deepstack_visual_embeds_list

def clear_deepstack_visual_embeds_list():
    """Clear the global deepstack visual embeddings list."""
    deepstack_visual_embeds_list.clear()

def get_deepstack_grad_list():
    """Return the global deepstack gradient list."""
    return deepstack_grad_list

def clear_deepstack_grad_list():
    """Clear the global deepstack gradient list."""
    deepstack_grad_list.clear()

def clear_full_hetero_info():
    _batch_fns[2]()
    clear_embedding_list()
    clear_grad_list()
    clear_visual_pos_masks_list()
    clear_deepstack_visual_embeds_list()
    clear_deepstack_grad_list()
    manager = get_cpu_offload_manager()
    if manager.enabled:
        manager.clear()


# ---------------------------------------------------------------------------
# CPU offload of encoder embeddings and decoder gradients
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Strided encoder data: each PP rank's encoder reads only its own microbatches
# ---------------------------------------------------------------------------

_SENTINEL = object()


class PrefetchIterator:
    """Prefetches items from a source iterator in a background thread.

    Allows the next training step's data to be loaded while the current
    step's encoder/decoder computation is running.
    """

    def __init__(self, source_iter, prefetch_count):
        self._queue = queue.Queue(maxsize=prefetch_count)
        self._source = source_iter
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            while True:
                item = next(self._source)
                self._queue.put(item)
        except StopIteration:
            self._queue.put(_SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is _SENTINEL:
            raise StopIteration
        return item


class EncoderStridedSampler:
    """Yields only microbatches assigned to this PP rank's encoder.

    Internally iterates the same index sequence as the decoder's sampler,
    but only yields batches at positions belonging to this PP rank.

    Processes items in chunks of num_real_microbatch (one step's worth)
    to ensure the position pattern is always step-relative, staying in
    sync with the decoder's DataLoader across steps.
    """

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding,
                 pp_rank, tp_size, model_size, num_real_microbatch):
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.pp_rank = pp_rank
        self.tp_size = tp_size
        self.model_size = model_size
        self.num_real_microbatch = num_real_microbatch

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        base_sampler = MegatronPretrainingRandomSampler(
            self.dataset,
            total_samples=self.total_samples,
            consumed_samples=self.consumed_samples,
            micro_batch_size=self.micro_batch_size,
            data_parallel_rank=self.data_parallel_rank,
            data_parallel_size=self.data_parallel_size,
            data_sharding=self.data_sharding,
        )
        start = self.pp_rank * self.tp_size
        end = start + self.tp_size
        # Process in step-sized chunks to keep position pattern step-relative
        step_buffer = []
        for batch in base_sampler:
            step_buffer.append(batch)
            if len(step_buffer) == self.num_real_microbatch:
                for i, b in enumerate(step_buffer):
                    if start <= (i % self.model_size) < end:
                        yield b
                step_buffer = []
        # Yield remaining partial step
        for i, b in enumerate(step_buffer):
            if start <= (i % self.model_size) < end:
                yield b


class EncoderStridedIterator:
    """Iterator-level strided filter for streaming (Energon/WebDataset) dataloaders.

    Wraps an EnergonDataloader, consumes all microbatches from it but only
    yields those assigned to this PP rank. Logically equivalent to
    EncoderStridedSampler but for iterable (non-indexable) datasets.

    Buffers microbatches in step-sized chunks (num_real_microbatch) to
    maintain correct position-based assignment, identical to the logic in
    EncoderStridedSampler.
    """

    def __init__(self, energon_dataloader, pp_rank, tp_size, model_size, num_real_microbatch):
        self._source = energon_dataloader
        self._pp_rank = pp_rank
        self._tp_size = tp_size
        self._model_size = model_size
        self._num_real_microbatch = num_real_microbatch
        self._gen = self._filter()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)

    def _filter(self):
        start = self._pp_rank * self._tp_size
        end = start + self._tp_size
        step_buffer = []
        while True:
            batch = next(self._source)
            step_buffer.append(batch)
            if len(step_buffer) == self._num_real_microbatch:
                for i, b in enumerate(step_buffer):
                    if start <= (i % self._model_size) < end:
                        yield b
                step_buffer = []
