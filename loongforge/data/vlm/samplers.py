# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Encoder samplers for full heterogeneous DP (``--enable-full-hetero-dp``).

Each PP rank's encoder reads only the microbatches assigned to it. The training
method passes the parallel sizes from ``engines.megatron.distributed.hetero_dp``.
"""

import queue
import threading

from torch.utils.data import DataLoader

from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler

from .dataloader import _cyclic_iter

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


def build_encoder_data_iterator(
    dataset,
    consumed_samples,
    data_collator,
    *,
    micro_batch_size,
    num_workers,
    data_sharding,
    data_parallel_rank,
    data_parallel_size,
    pp_rank,
    tp_size,
    model_size,
    num_real_microbatch,
    prefetch_count,
):
    """Build the strided encoder DataLoader for full hetero DP.

    Uses ``EncoderStridedSampler`` to yield only microbatches assigned to this PP
    rank, avoiding unnecessary disk IO for microbatches handled by other ranks.
    """
    batch_sampler = EncoderStridedSampler(
        dataset,
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
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
    return PrefetchIterator(iter(_cyclic_iter(dataloader)), prefetch_count=prefetch_count)


def build_encoder_energon_iterator(
    energon_dataloader,
    *,
    pp_rank,
    tp_size,
    model_size,
    num_real_microbatch,
    prefetch_count,
):
    """Wrap an Energon encoder loader with the strided/prefetch iterator for full hetero DP.

    The loader itself is built by the training method from its data config; this
    function only owns the parallelism-specific wrapping.
    """
    strided_iter = EncoderStridedIterator(
        energon_dataloader=energon_dataloader,
        pp_rank=pp_rank,
        tp_size=tp_size,
        model_size=model_size,
        num_real_microbatch=num_real_microbatch,
    )
    return PrefetchIterator(strided_iter, prefetch_count=prefetch_count)