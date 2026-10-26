# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Build the LLM SFT loader config from the training arguments.

The data layer receives this frozen config instead of the whole argument
namespace, so it never depends on the backend's argument schema.
"""

from loongforge.data.llm.loader_config import SFTLoaderConfig


def sft_loader_config(args) -> SFTLoaderConfig:
    """Collect the SFT loader recipe from the training arguments."""
    return SFTLoaderConfig(
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
