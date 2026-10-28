# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SFT dataloader utilities: config and cyclic data builders.

The tokenized-data branches of the VLM and InternVL methods reuse this module: they
read a dataset pre-sharded per data-parallel rank and wrap it in the same iterators.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

from torch.utils.data import DataLoader

from datasets import Dataset, IterableDataset, load_from_disk
from datasets.distributed import split_dataset_by_node

from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler

from .samplers import ChunkPipeGroupBatchSampler
from .sft.collator import DataCollatorForSupervisedDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SFTLoaderConfig:
    """Data-recipe fields needed to assemble the SFT loader.

    Built by the training method from its arguments; the loader never reads
    backend arguments or global CLI state.
    """

    tokenizer_type: str
    variable_seq_lengths: bool
    enable_chunkpipe: bool
    chunksize: int
    seq_length: int
    sft_chunkpipe_mode: bool
    mtp_num_layers: int
    sft_data_streaming: bool
    streaming_buffer_size: int
    seed: int
    micro_batch_size: int
    global_batch_size: int
    data_sharding: bool
    chunkpipe_enable_synthesis: bool
    num_workers: int
    dataloader_save: Optional[str] = None

    @classmethod
    def from_args(cls, args) -> "SFTLoaderConfig":
        """Build the loader recipe from the training arguments."""
        return cls(
            tokenizer_type=args.tokenizer_type,
            variable_seq_lengths=args.variable_seq_lengths,
            enable_chunkpipe=args.enable_chunkpipe,
            chunksize=args.chunksize,
            seq_length=args.seq_length,
            sft_chunkpipe_mode=getattr(args, "sft_chunkpipe_mode", False),
            mtp_num_layers=args.mtp_num_layers or 0,
            sft_data_streaming=args.sft_data_streaming,
            streaming_buffer_size=args.streaming_buffer_size,
            seed=args.seed,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            data_sharding=args.data_sharding,
            chunkpipe_enable_synthesis=getattr(args, "chunkpipe_enable_synthesis", False),
            num_workers=args.num_workers,
            dataloader_save=args.dataloader_save,
        )


def _cyclic_iter(iterable):
    """Yield from ``iterable`` forever."""
    while True:
        yield from iterable


class _IterableWithState:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.step = 0
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)
            self.step += 1
            return batch
        except StopIteration:
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)
            self.step += 1
            return batch

    def save_state(self):
        """dataloader save state"""
        return {"step": self.step}

    def load_state(self, state):
        """dataloader load state"""
        target = state.get("step", 0)
        if target <= self.step:
            return
        for _ in range(target - self.step):
            next(self._iterator)
        self.step = target


class SavableCyclicIterator:
    """
    Cyclic iterator that:
      - exposes `.iterable` with save_state/load_state (via _IterableWithState)
      - yields batches infinitely
    Compatible with Megatron's maybe_save_dataloader_state().
    """

    def __init__(self, dataloader):
        self.iterable = _IterableWithState(dataloader)
        self._iterator = self._cyclic_iter(self.iterable)

    def _cyclic_iter(self, iterable_with_state):
        while True:
            for batch in iterable_with_state:
                yield batch

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def save_state(self):
        """dataloader save state"""
        return self.iterable.save_state()

    def load_state(self, state):
        """dataloader load state"""
        return self.iterable.load_state(state)


def load_pretokenized_dataset(data_path, dp_rank):
    """Load the dataset pre-sharded for this data-parallel rank on disk."""
    save_path = os.path.join(data_path[0], "preprocess", str(dp_rank))
    print(f"[rank{dp_rank}] loading preprocessed dataset from {save_path}")
    return load_from_disk(save_path)


def _build_cyclic_iterator(
    dataset: Union["Dataset", "IterableDataset"],
    consumed_samples: int,
    data_collator: DataCollatorForSupervisedDataset,
    config: SFTLoaderConfig,
    dp_rank: int,
    dp_size: int,
):
    """build data iterator for sft"""
    if dataset is None:
        return None

    _dataloader_kwargs = {}
    if config.sft_data_streaming:
        # split distributed dataset for streaming
        dataset = split_dataset_by_node(
            dataset=dataset,
            rank=dp_rank,
            world_size=dp_size,
        )

        dataset = dataset.shuffle(
            buffer_size=config.streaming_buffer_size,
            seed=config.seed,
        )

        _dataloader_kwargs = dict(
            batch_size=config.micro_batch_size,
        )
    else:
        # build distribued sampler for non-streaming dataset
        if config.enable_chunkpipe:
            num_microbatches = config.global_batch_size // (
                config.micro_batch_size * dp_size
            )
            _batch_sampler = ChunkPipeGroupBatchSampler(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=config.micro_batch_size,
                data_parallel_rank=dp_rank,
                data_parallel_size=dp_size,
                num_microbatches=num_microbatches,
                seed=config.seed,
                enable_synthesis=config.chunkpipe_enable_synthesis,
            )
        else:
            _batch_sampler = MegatronPretrainingRandomSampler(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,  # not support for streaming now!
                micro_batch_size=config.micro_batch_size,
                data_parallel_rank=dp_rank,
                data_parallel_size=dp_size,
                data_sharding=config.data_sharding,
            )

        _dataloader_kwargs = dict(
            batch_sampler=_batch_sampler,
            persistent_workers=True if config.num_workers > 0 else False,
        )

    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        num_workers=config.num_workers,
        pin_memory=True,
        **_dataloader_kwargs,
    )

    if config.dataloader_save is not None:
        base_iter = SavableCyclicIterator(dataloader)
    else:
        base_iter = iter(_cyclic_iter(dataloader))

    if config.enable_chunkpipe and not config.sft_data_streaming:
        base_iter = _batch_sampler.attach_step_metadata(base_iter)

    return base_iter


def build_sft_cyclic_iterators(
    train_ds: Optional[Union["Dataset", "IterableDataset"]],
    valid_ds: Optional[Union["Dataset", "IterableDataset"]],
    test_ds: Optional[Union["Dataset", "IterableDataset"]],
    data_collator: Optional[DataCollatorForSupervisedDataset],
    config: SFTLoaderConfig,
    dp_rank: int,
    dp_size: int,
    consumed_train_samples: int,
    consumed_valid_samples: int,
    skip_train: bool,
    state=None,
):
    """build data iterators for sft

    ``state`` restores the streaming iterator's saved position. Non-streaming SFT
    resumes from ``consumed_samples`` in the sampler instead, so the saved step is
    not applied there.
    """
    train_iter = _build_cyclic_iterator(
        train_ds, consumed_train_samples, data_collator, config, dp_rank, dp_size
    )
    valid_iter = _build_cyclic_iterator(
        valid_ds, 0 if skip_train else consumed_valid_samples, data_collator, config, dp_rank, dp_size
    )
    test_iter = _build_cyclic_iterator(test_ds, 0, data_collator, config, dp_rank, dp_size)
    if state is not None and config.sft_data_streaming and hasattr(train_iter, "load_state"):
        train_iter.load_state(state)
    return train_iter, valid_iter, test_iter


