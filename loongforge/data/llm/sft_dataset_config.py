# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the SFT dataset."""

import logging
from dataclasses import dataclass
from typing import Optional

from transformers import ProcessorMixin
from megatron.core.utils import log_single_rank

from loongforge import constants
from loongforge.chat_templates.base import ChatTemplate

from .blended_hf_dataset_config import BlendedHuggingFaceDatasetConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_DATASET_NAME = "default"


@dataclass
class SFTDatasetConfig(BlendedHuggingFaceDatasetConfig):
    """The data config in SFT dataset"""

    dataset_config_file: str = ""
    """A path to a yaml file containing the dataset configuration for each dataset"""

    chat_template: Optional[ChatTemplate] = None
    """Template for the instruction dataset."""

    processor: Optional[ProcessorMixin] = None
    """The processor for the dataset"""

    ignore_index: int = constants.IGNORE_INDEX
    """The index to ignore in the dataset"""

    train_on_prompt: bool = False
    """Whether to train on the prompt or not."""

    eod_mask_loss: bool = None
    """Option to enable the EOD/EOS mask loss"""

    history_mask_loss: bool = False
    """Option to enable the history mask loss"""

    is_tokenized: bool = False
    """Whether the dataset is tokenized or not."""

    packing: bool = False
    """Whether to pack the dataset or not."""

    sort_batch: bool = False
    """Whether to sort batch or not"""

    packing_buffer_size: int = 10000
    """Perform packing in batches, deciding how many samples each batch contains"""

    context_parallel_size: Optional[int] = None
    """If packing is enabled, and context-parallel is enabled during the training phase,
     it is necessary to set the corresponding context_parallel_size to correctly pad the data."""

    enable_discard_sample: Optional[bool] = None
    """Sample sequence length bigger than sequence_length will be discarded."""

    enable_chunkpipe: bool = False
    """Whether to enable chunkpipe feature, which splits sequence into multiple chunks."""

    chunksize: Optional[int] = None
    """Size of each chunk when chunkpipe is enabled."""

    mtp_num_layers: int = 0
    """Number of MTP bridge tokens appended to each chunk when chunkpipe is enabled."""

    def _setup_default_dataset(self):
        """Setup default dataset or fix the length of dataset list"""

        def _setup(_blend, _dataset):
            if _blend is not None:
                if _dataset is None:
                    _dataset = [DEFAULT_DATASET_NAME] * len(_blend[0])

                    log_single_rank(
                        logger,
                        logging.WARN,
                        f">>> Not given any dataset name for {_blend}, setting to {_dataset}.",
                    )

                elif len(_dataset) != len(_blend[0]) and len(_dataset) == 1:
                    _dataset = _dataset * len(_blend[0])
                    log_single_rank(
                        logger,
                        logging.WARN,
                        f">>> Only given one dataset name for {_blend}, "
                        "and all datasets will be parsed according to the given dataset format",
                    )

            return _dataset

        # setup config.dataset
        self.dataset = _setup(self.blend, self.dataset)

        # setup config.dataset_per_split
        if self.blend_per_split is not None:
            if self.dataset_per_split is None:
                self.dataset_per_split = [None] * len(self.blend_per_split)

            assert len(self.dataset_per_split) == len(
                self.blend_per_split
            ), f"datset_per_split must contain {len(self.blend_per_split)} items"

            for i in range(len(self.blend_per_split)):
                self.dataset_per_split[i] = _setup(
                    self.blend_per_split[i], self.dataset_per_split[i]
                )

    def __post_init__(self) -> None:
        self._setup_default_dataset()

        assert (
            self.dataset_config_file is not None
        ), "dataset_config_file must be provided"
        assert self.chat_template is not None, "chat_template must be provided"
        assert self.eod_mask_loss is not None, "eod_mask_loss must be provided"

        if self.train_on_prompt and self.chat_template.efficient_eos:
            raise ValueError("Current template does not support `train_on_prompt`.")

        super().__post_init__()
