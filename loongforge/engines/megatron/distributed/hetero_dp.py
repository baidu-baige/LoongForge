# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Full heterogeneous DP (``--enable-full-hetero-dp``): per-step state, CPU offload, strided encoder data.

Encoder ranks run the vision encoder for every microbatch of their decoder DP group, gather the
embeddings, and scatter the gradients back after the decoder backward. ``train_step``, the model's
encoder path and the EP-overlap callables share the state kept here:

- embedding / grad / visual-pos-mask / deepstack lists of the current step;
- the CPU offload manager (``--full-hetero-dp-cpu-offload``);
- the encoder data iterator, assembled here from the dataset or Energon loader the method builds;
- the method's batch functions.

``HeteroStepContext`` owns all of it. The training method creates it, ``train_step``
fills the per-step lists and calls ``end_step`` after the step.
"""

import queue
import threading
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from megatron.core import mpu
from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler


class HeteroStepContext:
    """State shared by the engine, the model and the EP-overlap callables for one full-hetero-DP run.

    The lists live for one training step; the offload manager and the encoder data
    iterator live for the whole run. ``train_step`` fills the lists, the model forward
    and backward hooks read and write them, and ``end_step`` clears them.
    """

    def __init__(self, *, get_batch, create_mock_batch, reset_step_state, offload_enabled: bool):
        self.embedding_list = []
        self.grad_list = []
        self.visual_pos_masks_list = []
        self.deepstack_visual_embeds_list = []
        self.deepstack_grad_list = []
        self.offload = CpuOffloadManager(enabled=offload_enabled)
        self.get_batch = get_batch
        self.create_mock_batch = create_mock_batch
        self.encoder_data_iterator = None
        self._reset_step_state = reset_step_state

    def reload(self, items: list, index: int, prefix: str) -> Optional[torch.Tensor]:
        """Reload ``items[index]`` from CPU when offloading is on."""
        return reload_list_item(self.offload, items, index, prefix)

    def end_step(self) -> None:
        """Reset everything scoped to one step; keeps the offload manager and the iterator."""
        self._reset_step_state()
        for items in (
            self.embedding_list,
            self.grad_list,
            self.visual_pos_masks_list,
            self.deepstack_visual_embeds_list,
            self.deepstack_grad_list,
        ):
            items.clear()
        if self.offload.enabled:
            self.offload.clear()


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


def _encoder_parallel_facts():
    """Return the PP/TP/model-size/microbatch facts the strided sampler needs."""
    from ..initialize import (
        get_model_size,
        get_num_micro_batches_per_decoder_dp,
        get_num_real_micro_batches_per_decoder_dp,
    )

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    _, encoder_rounds = get_num_micro_batches_per_decoder_dp()
    return (
        pp_rank,
        tp_size,
        get_model_size(),
        get_num_real_micro_batches_per_decoder_dp(),
        tp_size * encoder_rounds,
    )


def build_encoder_data_iterator(
    dataset,
    consumed_samples,
    data_collator,
    *,
    micro_batch_size,
    num_workers,
    data_sharding,
):
    """Build the strided encoder DataLoader for full hetero DP.

    Uses ``EncoderStridedSampler`` to yield only microbatches assigned to this PP
    rank, avoiding unnecessary disk IO for microbatches handled by other ranks.
    """
    from megatron.training.training import cyclic_iter

    pp_rank, tp_size, model_size, num_real_microbatch, prefetch_count = _encoder_parallel_facts()
    batch_sampler = EncoderStridedSampler(
        dataset,
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        data_parallel_size=mpu.get_data_parallel_world_size(),
        data_sharding=data_sharding,
        pp_rank=pp_rank,
        tp_size=tp_size,
        model_size=model_size,
        num_real_microbatch=num_real_microbatch,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return PrefetchIterator(iter(cyclic_iter(dataloader)), prefetch_count=prefetch_count)


def build_encoder_energon_iterator(energon_dataloader):
    """Wrap an Energon encoder loader with the strided/prefetch iterator for full hetero DP.

    The loader itself is built by the training method from its data config; this
    function only owns the parallelism-specific wrapping.
    """
    pp_rank, tp_size, model_size, num_real_microbatch, prefetch_count = _encoder_parallel_facts()
    strided_iter = EncoderStridedIterator(
        energon_dataloader=energon_dataloader,
        pp_rank=pp_rank,
        tp_size=tp_size,
        model_size=model_size,
        num_real_microbatch=num_real_microbatch,
    )
    return PrefetchIterator(strided_iter, prefetch_count=prefetch_count)
