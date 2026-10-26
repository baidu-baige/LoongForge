# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLM dataset and DataLoader utilities (Megatron-Energon)."""

import logging
import os
import tempfile

import torch.nn.functional as F
import yaml

from megatron import energon
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.utils import log_single_rank
from .base_task_encoder import print_error_handler
from .flavors import ENERGON_LT_7

from .dataset_config import VLMDataConfig

logger = logging.getLogger(__name__)


def _cyclic_iter(iterable):
    """Yield from ``iterable`` forever."""
    while True:
        yield from iterable


def _energon_read_order_kwargs(config):
    """Resolve optional Energon sample-order controls from CLI flags.

    ``shuffle_buffer_size`` adds sample-level shuffling before cooking and
    encoding. For an offline-packed dataset, each raw sample is a complete pack,
    so this randomizes pack order without changing pack contents.
    ``max_samples_per_sequence`` targets shorter runs from each shard slice,
    giving Energon more sequences to mix through its parallel shard iterators.
    Disabled values are passed as ``None`` to preserve Energon's defaults.
    """
    shuffle_buffer_size = getattr(config, "data_shuffle_buffer_size", 0) or 0
    max_samples_per_sequence = getattr(config, "data_max_samples_per_sequence", 0) or 0
    return {
        "shuffle_buffer_size": shuffle_buffer_size if shuffle_buffer_size > 1 else None,
        "max_samples_per_sequence": (
            max_samples_per_sequence if max_samples_per_sequence > 0 else None
        ),
    }


def _validate_energon_data_paths(paths):
    for path in paths:
        if str(path).lower().endswith((".json", ".jsonl")):
            raise ValueError(
                "VLM training requires an Energon WebDataset directory, not raw "
                f"JSON/JSONL: {path}. Convert it first with "
                "tools/vlm_data_preprocess/convert_to_webdataset.py."
            )


def get_train_dataset(task_encoder, config: "VLMDataConfig", dp_rank, dp_size, dp_group):
    """Get the training dataset"""
    worker_config = energon.WorkerConfig(
        rank=dp_rank,
        world_size=dp_size,
        num_workers=config.num_workers,
        data_parallel_group=dp_group,
        worker_debug_path=None,
        worker_log_level=0,
    )
    shuffle_kwargs = _energon_read_order_kwargs(config)
    log_single_rank(
        logger,
        logging.INFO,
        f"> Energon read order: shuffle_buffer_size="
        f"{shuffle_kwargs['shuffle_buffer_size']}, max_samples_per_sequence="
        f"{shuffle_kwargs['max_samples_per_sequence']}",
    )

    if len(config.data_path) == 1:
        _validate_energon_data_paths(config.data_path)
        train_ds = energon.get_train_dataset(
            config.data_path[0],
            batch_size=config.micro_batch_size,
            task_encoder=task_encoder,
            worker_config=worker_config,
            packing_buffer_size=config.packing_buffer_size,
            handler=print_error_handler,
            image_decode="pil",
            **shuffle_kwargs,
        )
    else:
        data_paths, data_weights = get_blend_from_list(config.data_path)
        _validate_energon_data_paths(data_paths)
        yaml_path = create_metadataset_yaml(data_paths, data_weights, split="train")
        train_ds = energon.get_train_dataset(
            yaml_path,
            batch_size=config.micro_batch_size,
            task_encoder=task_encoder,
            worker_config=worker_config,
            packing_buffer_size=config.packing_buffer_size,
            handler=print_error_handler,
            image_decode="pil",
            **shuffle_kwargs,
        )
    return train_ds


def get_val_dataset(task_encoder, config: "VLMDataConfig", dp_rank, dp_size, dp_group):
    """Build the validation split from the configured Energon dataset."""
    paths = config.valid_data_path or config.data_path
    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    if len(paths) == 1:
        _validate_energon_data_paths(paths)
        path = paths[0]
    else:
        data_paths, data_weights = get_blend_from_list(list(paths))
        _validate_energon_data_paths(data_paths)
        path = create_metadataset_yaml(data_paths, data_weights, split="val")

    worker_config = energon.WorkerConfig(
        rank=dp_rank,
        world_size=dp_size,
        num_workers=config.num_workers,
        data_parallel_group=dp_group,
        worker_debug_path=None,
        worker_log_level=0,
    )
    return energon.get_val_dataset(
        path,
        split_part="val",
        batch_size=config.micro_batch_size,
        task_encoder=task_encoder,
        worker_config=worker_config,
        packing_buffer_size=config.packing_buffer_size,
        handler=print_error_handler,
        image_decode="pil",
    )


def create_metadataset_yaml(data_paths, data_weights, split="train"):
    """
    Create a temporary metadataset.yaml file for multiple datasets

    Args:
        data_paths: List of dataset paths
        data_weights: List of weights corresponding to each dataset
        split: Dataset split name (default: 'train')

    Returns:
        Path to the temporary yaml file
    """
    # Prepare the blend configuration
    blend = []
    for i, path in enumerate(data_paths):
        blend_item = {"path": path}
        # Only add weight if weights are provided
        if data_weights is not None:
            blend_item["weight"] = data_weights[i]
        blend.append(blend_item)

    # Create the metadataset configuration
    metadataset_config = {
        "__module__": "megatron.energon",
        "__class__": "MetadatasetV2",
        "splits": {split: {"blend": blend}},
    }

    # Create a temporary yaml file
    temp_dir = tempfile.gettempdir()
    yaml_path = os.path.join(temp_dir, f"metadataset_{os.getpid()}_{split}.yaml")

    with open(yaml_path, "w") as f:
        yaml.dump(metadataset_config, f, default_flow_style=False)

    return yaml_path


def get_train_loader(train_ds, collator=None, state=None):
    """Get the training loader, restoring the saved state when one is given."""
    if ENERGON_LT_7:
        train_dataloader = energon.get_savable_loader(train_ds)
    else:
        train_dataloader = energon.get_savable_loader(train_ds, watchdog_initial_timeout_seconds=600)

    if state is not None:
        try:
            train_dataloader.restore_state_rank(state)
        except Exception as e:
            print("loading dataset state failed. Skipping. " + str(e))
    return EnergonDataloader(train_dataloader, collator)


class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""

    def __init__(self, dataloader, collator=None):
        self._dataloader = dataloader
        self._collator = collator
        self._iter = iter(_cyclic_iter(dataloader))

    def __next__(self):
        features = self._iter.__next__()
        if self._collator is not None:
            if hasattr(self._collator, "collate_energon"):
                return self._collator.collate_energon(features)
            padded = self._collator.tokenizer.pad(
                {"input_ids": features["tokens"]},
                padding=self._collator.padding,
                max_length=self._collator.max_length,
                pad_to_multiple_of=self._collator.pad_to_multiple_of,
            )
            paded_length = padded["input_ids"].shape[1] - features["tokens"].shape[1]
            features["tokens"] = padded["input_ids"]
            features["labels"] = F.pad(
                features["labels"],
                (0, paded_length),
                "constant",
                self._collator.label_pad_token_id,
            )
            features["attn_mask"] = F.pad(
                features["attn_mask"], (0, paded_length), "constant", True
            )
        return features

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        """Save the current state of this dataloader"""
        return self._dataloader.save_state_rank()
