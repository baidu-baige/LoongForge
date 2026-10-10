# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""sft dataset build on huggingface dataset"""

import os
import logging
from pathlib import Path

from typing import Optional, Union, Tuple, List

import yaml
import torch
from datasets import Dataset, IterableDataset, DatasetDict, load_dataset

from megatron.core.utils import log_single_rank
from megatron.core.datasets.utils import Split

from .sft_dataset_config import SFTDatasetConfig
from .sft_format_utils import (
    convert_to_unified_format,
    SFTDataFormats,
    SFTDataFormat,
    AlpacaColumns,
    ShareGPTColumns,
    ShareGPTTags,
    OpenAIChatColumns,
)
from .sft_tokenize_utils import convert_to_tokenized_data


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SFT_SUPPORT_DATA_TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}


class SFTDataset:
    """Build the sft dataset."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        config: SFTDatasetConfig,
        print_example: bool = True,
    ):
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
        elif sft_format.format == SFTDataFormats.OPENAI:
            sft_format.columns = OpenAIChatColumns()
            if _desc_columns is not None:
                sft_format.columns.messages = _desc_columns.get("messages", "messages")
                sft_format.columns.tools = _desc_columns.get("tools", "tools")
                sft_format.columns.images = _desc_columns.get("images", None)
                sft_format.columns.videos = _desc_columns.get("videos", None)
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
            raise ValueError("All files must be of the same type.")

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
        example_str += ">>> input: \n"
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
                        if getattr(self.config, 'enable_chunkpipe', False):
                            # Chunkpipe requires chunk groups (consecutive chunks
                            # from the same long sequence) to stay adjacent.
                            # train_test_split shuffles by default, which would
                            # break that invariant.  Use ordered slicing instead,
                            # snapping the split boundary to a chunk group boundary
                            # so that no group is torn across two splits.
                            group_sizes = low_level_dataset["chunk_group_size"]
                            total = len(low_level_dataset)
                            target = split_samplers[i]
                            boundary = 0
                            while boundary < total and boundary < target:
                                boundary += group_sizes[boundary]
                            self.split_dataset[i] = low_level_dataset.select(
                                range(0, boundary)
                            )
                            low_level_dataset = low_level_dataset.select(
                                range(boundary, total)
                            )
                        else:
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
