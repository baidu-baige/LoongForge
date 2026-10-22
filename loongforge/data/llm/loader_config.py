# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Frozen data-recipe config for the SFT loader (no heavy imports)."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SFTLoaderConfig:
    """Data-recipe fields needed to assemble the SFT loader.

    Built by the training method from its arguments; the data layer never reads
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
