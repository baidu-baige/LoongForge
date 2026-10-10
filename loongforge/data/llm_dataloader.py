# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SFT dataloader utilities: iterators, collators, and cyclic data builders."""

import logging
from typing import Optional, Type, Union

from torch.utils.data import DataLoader
from transformers.utils import PaddingStrategy

from datasets import Dataset, IterableDataset
from datasets.distributed import split_dataset_by_node

from megatron.core import mpu

from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler
from megatron.training.training import cyclic_iter

from loongforge import constants
from loongforge.engines.mcore.parallel.chunkpipe import (
    ChunkPipeGroupBatchSampler,
    _bind_chunkpipe_queue_iter,
)
from .llm.sft_data_collator import DataCollatorForSupervisedDataset
from loongforge.engines.mcore.tokenizer import AutoTokenizerFromHF

logger = logging.getLogger(__name__)


def build_sft_data_collator(
    collator_cls: Type[DataCollatorForSupervisedDataset], args, tokenizer, **kwargs
) -> DataCollatorForSupervisedDataset:
    """build data collator for sft"""
    assert isinstance(
        tokenizer, AutoTokenizerFromHF
    ), f"Only support HFTokenizer for sft, but got {args.tokenizer_type}."

    pad_to_multiple_of = 1
    # When using sequence parallel, sequence will further be split by TP size
    # When using context parallel, sequence is split by CP size as well
    pad_to_multiple_of *= (
        args.tensor_model_parallel_size if args.sequence_parallel else 1
    )
    pad_to_multiple_of *= (
        (2 * args.context_parallel_size) if args.context_parallel_size > 1 else 1
    )

    # https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/utils.py#L425
    # https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/gemm/cublaslt_gemm.cu#L151
    pad_to_multiple_of *= 128 if args.fp8 else 1
    if args.enable_chunkpipe and getattr(args, "sft_chunkpipe_mode", False):
        pad_to_multiple_of = args.chunksize

    padding = (
        PaddingStrategy.LONGEST
        if args.variable_seq_lengths
        else PaddingStrategy.MAX_LENGTH
    )

    # When chunkpipe is enabled, all base chunks are already padded to chunksize.
    # If SFT chunkpipe + MTP is enabled, the collator temporarily strips bridge
    # tokens, pads only the base part to pad_to_multiple_of, then appends bridge
    # tokens back.
    max_length = args.chunksize if args.enable_chunkpipe else args.seq_length

    if args.enable_chunkpipe and getattr(args, "sft_chunkpipe_mode", False):
        kwargs["chunkpipe_base_length"] = args.chunksize
        kwargs["chunkpipe_mtp_num_layers"] = args.mtp_num_layers or 0

    data_collator = collator_cls(
        tokenizer=tokenizer.tokenizer,
        label_pad_token_id=constants.IGNORE_INDEX,
        pad_to_multiple_of=pad_to_multiple_of,
        padding=padding,
        max_length=max_length,
        **kwargs,
    )
    return data_collator


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
            # self.step = 0
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


def _build_cyclic_iterator(
    dataset: Union["Dataset", "IterableDataset"],
    consumed_samples: int,
    data_collator: DataCollatorForSupervisedDataset,
    args,
):
    """build data iterator for sft"""
    if dataset is None:
        return None

    _dataloader_kwargs = {}
    if args.sft_data_streaming:
        # split distributed dataset for streaming
        dataset = split_dataset_by_node(
            dataset=dataset,
            rank=mpu.get_data_parallel_rank(),
            world_size=mpu.get_data_parallel_world_size(),
        )

        dataset = dataset.shuffle(
            buffer_size=args.streaming_buffer_size,
            seed=args.seed,
        )

        _dataloader_kwargs = dict(
            batch_size=args.micro_batch_size,
        )
    else:
        # build distribued sampler for non-streaming dataset
        if args.enable_chunkpipe:
            num_microbatches = args.global_batch_size // (
                args.micro_batch_size * mpu.get_data_parallel_world_size()
            )
            _batch_sampler = ChunkPipeGroupBatchSampler(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
                num_microbatches=num_microbatches,
                seed=args.seed,
                enable_synthesis=getattr(args, "chunkpipe_enable_synthesis", False),
            )
        else:
            _batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,  # not support for streaming now!
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
        )

        _dataloader_kwargs = dict(
            batch_sampler=_batch_sampler,
            persistent_workers=True if args.num_workers > 0 else False,
        )

    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        **_dataloader_kwargs,
    )

    if args.dataloader_save is not None:
        base_iter = SavableCyclicIterator(dataloader)
    else:
        base_iter = iter(cyclic_iter(dataloader))

    if args.enable_chunkpipe and not args.sft_data_streaming:
        base_iter = _bind_chunkpipe_queue_iter(
            base_iter,
            _batch_sampler._step_g_queue,
            _batch_sampler._composite_queue,
        )

    return base_iter


def build_sft_cyclic_iterators(
    train_ds: Optional[Union["Dataset", "IterableDataset"]],
    valid_ds: Optional[Union["Dataset", "IterableDataset"]],
    test_ds: Optional[Union["Dataset", "IterableDataset"]],
    data_collator: Optional[DataCollatorForSupervisedDataset],
    args,
):
    """build data iterators for sft"""
    train_iter = _build_cyclic_iterator(
        train_ds, args.consumed_train_samples, data_collator, args
    )
    valid_iter = _build_cyclic_iterator(
        valid_ds, 0 if args.skip_train else args.consumed_valid_samples, data_collator, args
    )
    test_iter = _build_cyclic_iterator(test_ds, 0, data_collator, args)
    return train_iter, valid_iter, test_iter


