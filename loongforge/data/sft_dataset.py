# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""sft dataset build on huggingface dataset"""

import os
import logging
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List, Any

import yaml
import torch
from datasets import Dataset, IterableDataset, DatasetDict, load_dataset

from megatron.core.utils import log_single_rank
from megatron.core.datasets.utils import Split
from transformers import ProcessorMixin

from loongforge.utils.constants import (
    SFTDataFormats,
    DEFAULT_DATASET_NAME,
    SFT_SUPPORT_DATA_TYPE,
)

from .blended_hf_dataset_config import BlendedHuggingFaceDatasetConfig
from .hf_dataset import HuggingFaceDataset
from .chat_template import ChatTemplate
from .sft_format_utils import convert_to_unified_format
from .sft_supervised_utils import convert_to_tokenized_data


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Columns(object):
    """The field of the dataset."""

    pass


@dataclass
class AlpacaColumns(Columns):
    """The field of the dataset."""

    system: Optional[str] = None
    """The system prompt for sft"""

    prompt: Optional[str] = "instruction"
    """The prompt column"""

    query: Optional[str] = "input"
    """The query column"""

    response: Optional[str] = "output"
    """The response column"""

    history: Optional[str] = None
    """The history column"""


@dataclass
class ShareGPTColumns(Columns):
    """The field of the dataset."""

    messages: Optional[List[dict]] = field(default_factory=list)
    """The messages for sft"""

    images: Optional[List[str]] = field(default_factory=list)
    """The prompt column"""

    videos: Optional[List[str]] = field(default_factory=list)
    """The prompt column"""

    system: Optional[str] = None
    """The system prompt for sft"""

    tools: Optional[str] = None
    """The tools for sft"""


@dataclass
class ShareGPTTags(object):
    """The tag of the dataset."""

    role_tag: Optional[str] = None
    """The role of the tag"""

    content_tag: Optional[str] = None
    """The content of the tag"""

    user_tag: Optional[str] = None
    """The user of the tag"""

    assistant_tag: Optional[str] = None
    """The assistant of the tag"""

    observation_tag: Optional[str] = None
    """The observaton of the tag"""

    function_tag: Optional[str] = None
    """The function of the tag"""

    system_tag: Optional[str] = None
    """The system of the tag"""


@dataclass
class SFTDataFormat(object):
    """The data format of the dataset."""

    format: Optional[str] = None
    """The format of the dataset."""

    columns: Optional[Columns] = field(init=False, default=None)
    """The columns of the dataset."""

    tags: Optional[ShareGPTTags] = field(init=False, default=None)
    """ optional, used for the sharegpt format """

    def __post_init__(self) -> None:
        assert self.format is not None, "format must be provided"


@dataclass
class SFTDatasetConfig(BlendedHuggingFaceDatasetConfig):
    """The data config in SFT dataset"""

    dataset_config_file: str = ""
    """A path to a yaml file containing the dataset configuration for each dataset"""

    chat_template: Optional[ChatTemplate] = None
    """Template for the instruction dataset."""

    processor: Optional[ProcessorMixin] = None
    """The processor for the dataset"""

    ignore_index: int = -100
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


class SFTDataset(HuggingFaceDataset):
    """Build the sft dataset."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        config: SFTDatasetConfig,
        print_example: bool = True,
    ):
        super().__init__(dataset_name, dataset_path, config)

        self.config = config
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        self.sft_dataset = None
        self.num_samples = 0
        self.split_dataset = [None] * len(Split)

        self.build()

        if print_example:
            self._log_one_example()

    def build(self) -> Optional[Tuple[Union[Dataset, IterableDataset], int]]:
        """
        build the sft dataset.

        Returns:
            dataset: The sft dataset.
            num_samples: The number of samples in the dataset.
        """
        if self.config.is_tokenized:
            return self._build_from_tokenized_data()

        return self._build_from_raw_data()

    def _build_from_raw_data(self):
        """
        Build dataset from raw data.
        """
        self.dataset_format = self._get_format_config()

        path_to_cache = self.config.path_to_cache
        if path_to_cache is None or not torch.distributed.is_initialized():
            return self._build_low_level_dataset(path_to_cache=path_to_cache)

        rank = torch.distributed.get_rank()
        # First, build on rank 0
        if rank == 0:
            self._build_low_level_dataset(path_to_cache=path_to_cache)

        torch.distributed.barrier()

        if rank != 0:
            self._build_low_level_dataset(path_to_cache=path_to_cache)

    def _build_from_tokenized_data(self):
        """
        Build dataset from tokenized data.
        """
        assert (
            os.path.isdir(self.dataset_path) and len(os.listdir(self.dataset_path)) > 0
        ), f"dataset path {self.dataset_path} is not a directory, or empty"

        log_single_rank(
            logger,
            logging.INFO,
            f">>> Loading tokenized dataset from {self.dataset_path}, and will ignore --split flag ...",
        )

        dataset_dict = DatasetDict.load_from_disk(self.dataset_path)

        for i, split in enumerate(Split):
            if split.name in dataset_dict:
                dataset = dataset_dict[split.name]
                log_single_rank(
                    logger, logging.INFO, f">>> {split.name} samples: {len(dataset)}"
                )

                if self.config.streaming:
                    dataset = dataset.to_iterable_dataset()

                self.split_dataset[i] = dataset

    def _get_format_config(self) -> SFTDataFormat:
        """
        Get the format config of the dataset.

        Returns:
            SFTDataFormat: The format config of the dataset.
        """
        config_file = {}
        config_path = Path(self.config.dataset_config_file)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Dataset config file not found: {self.config.dataset_config_file}"
            )

        with open(config_path, "r") as f:
            # read the config file, support yaml only
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                config_file = yaml.safe_load(f) or {}
            else:
                raise ValueError(
                    f"Unsupported dataset config format: {config_path.suffix}. "
                    "Only .yaml/.yml are supported."
                )

        _dataset_desc = config_file.get(self.dataset_name, None)

        if _dataset_desc is None:
            raise ValueError(
                f"Dataset {self.dataset_name} not found "
                f"in config file {self.config.dataset_config_file}"
            )

        _desc_format = _dataset_desc.get("format", None) or _dataset_desc.get(
            "formatting", None
        )
        _desc_columns = _dataset_desc.get("columns", None)
        _desc_tags = _dataset_desc.get("tags", None)

        if _desc_format is None:
            log_single_rank(
                logger,
                logging.WARN,
                f">>> Not found dataset {self.dataset_name} format in config {self.config.dataset_config_file}, "
                f"use default {SFTDataFormats.ALPACA} format.",
            )
            _desc_format = SFTDataFormats.ALPACA  # default alpaca

        # build sft dataset config
        sft_format = SFTDataFormat(format=_desc_format)

        # build sft dataset columns
        if sft_format.format == SFTDataFormats.ALPACA:
            sft_format.columns = AlpacaColumns()
            if _desc_columns is not None:
                sft_format.columns.system = _desc_columns.get("system", None)
                sft_format.columns.prompt = _desc_columns.get("prompt", None)
                sft_format.columns.query = _desc_columns.get("query", None)
                sft_format.columns.response = _desc_columns.get("response", None)
                sft_format.columns.history = _desc_columns.get("history", None)
        elif sft_format.format == SFTDataFormats.SHAREGPT:
            sft_format.columns = ShareGPTColumns()
            sft_format.tags = ShareGPTTags()
            if _desc_columns is not None:
                sft_format.columns.messages = _desc_columns.get("messages", None)
                sft_format.columns.images = _desc_columns.get("images", None)
                sft_format.columns.system = _desc_columns.get("system", None)
                sft_format.columns.tools = _desc_columns.get("tools", None)
                sft_format.tags.role_tag = _desc_tags.get("role_tag", None)
                sft_format.tags.content_tag = _desc_tags.get("content_tag", None)
                sft_format.tags.user_tag = _desc_tags.get("user_tag", None)
                sft_format.tags.assistant_tag = _desc_tags.get("assistant_tag", None)
                sft_format.tags.observation_tag = _desc_tags.get(
                    "observation_tag", None
                )
                sft_format.tags.function_tag = _desc_tags.get("function_tag", None)
                sft_format.tags.system_tag = _desc_tags.get("system_tag", None)
        else:
            raise ValueError(f"Unknown dataset format {sft_format.format}")

        log_single_rank(
            logger,
            logging.INFO,
            f">>> Dataset {self.dataset_name} format: {sft_format}",
        )

        return sft_format

    def _build_low_level_dataset(
        self,
        path_to_cache: Optional[str] = None,
    ) -> Optional[Tuple[Union[Dataset, IterableDataset], int]]:
        """
        Use the huggingface datasets library to build the dataset, and execute data preprocessing with .map() function.

        TODO: Maybe we should support lazy data preprocessing when not using streaming? (It should inheirit from
        the base class, and override the __getitem__() function)

        Returns:
            dataset: The sft dataset.
            num_samples: The number of samples in the dataset.
        """

        # get files
        data_files = []
        if os.path.isdir(self.dataset_path):
            data_files = [
                os.path.join(self.dataset_path, file)
                for file in os.listdir(self.dataset_path)
            ]
        elif os.path.isfile(self.dataset_path):
            data_files = [self.dataset_path]
        else:
            raise ValueError(f"The dataset path [{self.dataset_path}] does not exist")

        # check file type
        data_type = SFT_SUPPORT_DATA_TYPE.get(
            os.path.splitext(data_files[0])[-1][1:], None
        )
        assert (
            data_type is not None
        ), f"Only support file types: {', '.join(SFT_SUPPORT_DATA_TYPE.keys())}"

        if any(
            data_type != SFT_SUPPORT_DATA_TYPE.get(os.path.splitext(file)[-1][1:], None)
            for file in data_files
        ):
            raise ValueError(f"All files must be of the same type.")

        log_single_rank(logger, logging.INFO, f">>> Detected data files: {data_files}")

        dataset = load_dataset(
            data_type,
            data_files=data_files,
            cache_dir=path_to_cache,
            split="train",
            token=False,
            num_proc=self.config.num_preprocess_workers,
        )

        num_samples = len(dataset)

        log_single_rank(
            logger,
            logging.INFO,
            f">>> Loading dataset {self.dataset_path}（{num_samples} samples) "
            f"with {self.dataset_name} config ...",
        )

        if self.config.streaming:
            # FIXME: or just set streaming=True in the load_dataset() function?
            dataset = dataset.to_iterable_dataset()

        # convert the dataset to unified format
        dataset = convert_to_unified_format(
            dataset,
            self.dataset_path,
            self.dataset_format,
            self.config,
            path_to_cache is not None,
        )

        # run sft preprocess
        dataset = convert_to_tokenized_data(
            dataset, self.config, path_to_cache is not None
        )

        if not self.config.streaming:
            # the dataset len may be changed when the dataset is packed in preprocess function,
            # so we need to update it here, but it is not a good idea to do this when streaming is enabled
            if len(dataset) != num_samples:
                log_single_rank(
                    logger,
                    logging.INFO,
                    f">>> The number of samples have been changed from "
                    f"{num_samples} to {len(dataset)} after preprocess.",
                )

                num_samples = len(dataset)

        self.sft_dataset = dataset
        self.num_samples = num_samples

    def _log_one_example(self) -> None:
        """
        Log one example from the dataset.
        """
        example = None
        if not self.config.is_tokenized:
            example = next(iter(self.sft_dataset))
        else:
            for subset in self.split_dataset:
                if subset is not None:
                    example = next(iter(subset))
                    break

        if example is None:
            log_single_rank(
                logger,
                logging.ERROR,
                f">>> No example data found in {self.dataset_path}",
            )
            return

        example_str = (
            f"\n----------------Example Data In {self.dataset_path}----------------\n"
        )
        example_str += f">>> input: \n"
        example_str += f"{self.config.tokenizer.detokenize(example['input_ids'], skip_special_tokens=False)}\n"
        example_str += f">>> input_ids: \n{example['input_ids']}\n"

        _labels = list(
            filter(lambda x: x != self.config.ignore_index, example["labels"])
        )
        example_str += f">>> labels: \n{self.config.tokenizer.detokenize(_labels, skip_special_tokens=False)}\n"
        example_str += f">>> label_ids: \n{example['labels']}\n"
        log_single_rank(logger, logging.INFO, f"{example_str}")

    def split(
        self, split: Optional[List[Tuple[float, float]]]
    ) -> List[Optional[Union[Dataset, IterableDataset]]]:
        """split the dataset into multiple subsets

        Args:
            split (Optional[List[Tuple[float, float]]]): The split of the dataset

        Returns:
            List[Optional[Union[Dataset, IterableDataset]]]: A list containing a dataset instance (or None)
        """
        if self.config.is_tokenized:
            # allready split
            return self.split_dataset

        low_level_dataset = self.sft_dataset
        num_elements = self.num_samples

        # get the split samplers
        split_samplers = []
        for i, _ in enumerate(Split):
            if split[i] is not None:
                beg = int(round(split[i][0] * float(num_elements)))
                end = int(round(split[i][1] * float(num_elements)))
                split_samplers.append(
                    end - beg
                )  #  can also calculate directly based on the proportion
            else:
                split_samplers.append(None)

        split_times = len([s for s in split_samplers if s is not None]) - 1

        for i in range(len(split_samplers)):
            if split_samplers[i] is not None:
                if split_times == 0:
                    self.split_dataset[i] = low_level_dataset
                else:
                    if not self.config.streaming:
                        # for mappable dataset
                        temp_split = low_level_dataset.train_test_split(
                            train_size=split_samplers[i],
                            seed=self.config.random_seed,
                        )
                        self.split_dataset[i] = temp_split["train"]
                        low_level_dataset = temp_split["test"]
                    else:
                        # for iterable dataset
                        self.split_dataset[i] = low_level_dataset.take(
                            split_samplers[i]
                        )
                        low_level_dataset = low_level_dataset.skip(split_samplers[i])

                    # update the split_times
                    split_times -= 1

        return self.split_dataset
