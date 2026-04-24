# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dataset and DataLoader related utilities"""

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from transformers.utils import PaddingStrategy

from megatron import energon
from megatron.core import parallel_state
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.models.multimodal import context_parallel
from megatron.core.transformer.enums import AttnMaskType
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name
from loongforge.utils import constants, get_model_config
from .base.task_encoder import print_error_handler
from loongforge.train.get_position_idx_func import get_position_ids

IGNORE_INDEX = constants.IGNORE_INDEX
PAD_TOKEN_ID = 151643


def seq_padding_for_cp(data, tp_size=1, cp_size=1, has_sp=False):
    """Sequence padding for CP and/or SP

    Args:
        data (dict): Data from dataloader.
        tp_size (int): Tensor parallel size.
        cp_size (int): Context parallel size.
        has_sp (bool): Model uses sequence parallelism.

    Returns:
        data (dict): Padded data.
    """
    tokens = data["tokens"]
    labels = data["labels"]
    attn_mask = data["attn_mask"]
    cu_lengths = data["cu_lengths"]
    max_lengths = data["max_lengths"]

    valid_tokens = []
    valid_labels = []
    valid_attn_mask = []

    cu_seqlens_padded = [0]
    seq_lengths = cu_lengths[0, 1:] - cu_lengths[0, :-1]
    start = 0
    for length in seq_lengths:
        token = tokens[0, start : start + length]
        label = labels[0, start : start + length]
        mask = attn_mask[0, start : start + length]

        mp_padding_needed = context_parallel.get_padding(
            length, cp_size, tp_size, has_sp
        )

        input_ids = F.pad(token, (0, mp_padding_needed), "constant", PAD_TOKEN_ID)
        label = F.pad(label, (0, mp_padding_needed), "constant", IGNORE_INDEX)
        mask = F.pad(mask, (0, mp_padding_needed), "constant", True)

        valid_tokens.append(input_ids)
        valid_labels.append(label)
        valid_attn_mask.append(mask)

        cu_seqlens_padded.append(
            int(cu_seqlens_padded[-1] + length + mp_padding_needed)
        )

        start += length

    data["tokens"] = torch.cat(valid_tokens, dim=0).unsqueeze(0).to(tokens.dtype)
    data["labels"] = torch.cat(valid_labels, dim=0).unsqueeze(0).to(labels.dtype)
    data["attn_mask"] = (
        torch.cat(valid_attn_mask, dim=0).unsqueeze(0).to(attn_mask.dtype)
    )

    data["cu_lengths"] = torch.tensor(
        cu_seqlens_padded, dtype=cu_lengths.dtype
    ).unsqueeze(0)
    cu_seqlens_padded = torch.tensor(cu_seqlens_padded, dtype=torch.int32)
    seq_lens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    data["max_lengths"] = torch.tensor(
        [seq_lens_padded.max().item()], dtype=max_lengths.dtype
    )

    return data


@dataclass
class VLMPretrainCollator:
    """Collator that performs multimodal padding plus mask/position preprocessing."""

    tokenizer: Any
    model: Optional[Any] = None
    padding: Optional[PaddingStrategy] = None
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = IGNORE_INDEX
    return_tensors: str = "pt"

    def collate_energon(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw Energon batch tensors, pad for TP/CP configs, then build masks/positions."""
        batch = self._ensure_tensor(batch)
        self._pad_sequences(batch)
        args = get_args()
        if args.packing_sft_data and (
            args.context_parallel_size > 1 or args.sequence_parallel
        ):
            seq_padding_for_cp(
                batch,
                tp_size=args.tensor_model_parallel_size,
                cp_size=args.context_parallel_size,
                has_sp=args.sequence_parallel,
            )
        self._build_masks_and_positions(batch)
        return batch

    def _ensure_tensor(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        tensor_specs = (
            ("tokens", torch.long),
            ("labels", torch.long),
            ("attn_mask", torch.bool),
            ("cu_lengths", torch.int32),
            ("max_lengths", torch.int32),
        )
        for key, dtype in tensor_specs:
            value = batch.get(key)
            if value is None:
                continue
            if torch.is_tensor(value):
                if value.dtype != dtype:
                    batch[key] = value.to(dtype)
            else:
                batch[key] = torch.as_tensor(value, dtype=dtype)
        return batch

    def _pad_sequences(self, batch: Dict[str, Any]) -> None:
        tokens = batch["tokens"]
        seq_len = tokens.shape[-1]
        target_len = seq_len
        padding_value = (
            self.padding.value
            if isinstance(self.padding, PaddingStrategy)
            else self.padding
        )
        if padding_value == PaddingStrategy.MAX_LENGTH.value:
            if self.max_length is not None:
                target_len = self.max_length
        if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
            target_len = (
                (target_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        pad_len = target_len - seq_len
        if pad_len <= 0:
            return
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        pad_token_id = PAD_TOKEN_ID if pad_token_id is None else pad_token_id
        batch["tokens"] = F.pad(tokens, (0, pad_len), "constant", pad_token_id)
        batch["labels"] = F.pad(
            batch["labels"], (0, pad_len), "constant", self.label_pad_token_id
        )
        batch["attn_mask"] = F.pad(batch["attn_mask"], (0, pad_len), "constant", True)
        # Extend last packed boundary after right-padding;
        # keep sum(seqlens) == tokens.shape[-1] for apply_mrope split.
        cu_lengths = batch.get("cu_lengths")
        if cu_lengths is not None and cu_lengths.shape != torch.Size([1, 1]):
            batch["cu_lengths"] = cu_lengths.clone()
            batch["cu_lengths"][..., -1] += pad_len

    def _build_masks_and_positions(self, batch: Dict[str, Any]) -> None:
        tokens = batch["tokens"]
        labels = batch["labels"]
        attention_mask = batch["attn_mask"]
        cu_lengths = batch["cu_lengths"]
        model_config = get_model_config()
        get_position_ids_func = getattr(
            model_config, "position_idx_func", get_position_ids
        )
        position_ids, _ = get_position_ids_func(batch)
        batch["position_ids"] = position_ids.to(dtype=torch.long)

        # Shift labels left for next-token prediction and build a loss mask aligned to the shifted labels.
        labels = torch.roll(labels, shifts=-1, dims=1)
        loss_mask = (labels != self.label_pad_token_id).long()

        batch["labels"] = labels

        if attention_mask is not None:
            if cu_lengths.shape == torch.Size([1, 1]):
                for i in range(attention_mask.shape[0]):
                    valid_tokens = (~attention_mask[i]).sum().item()
                    if valid_tokens > 0:
                        loss_mask[i, valid_tokens - 1] = 0
            else:
                for i in range(cu_lengths.shape[0]):
                    for j in range(1, cu_lengths[i].shape[0]):
                        idx = cu_lengths[i][j].item() - 1
                        if 0 <= idx < loss_mask.shape[1]:
                            loss_mask[i, idx] = 0
                assert (
                    cu_lengths.shape[0] == 1
                ), "micro-batch-size must be 1 for packing"

        batch["loss_mask"] = loss_mask


def get_train_dataset(task_encoder):
    """Get the training dataset"""
    args = get_args()
    worker_config = energon.WorkerConfig(
        rank=parallel_state.get_data_parallel_rank(),
        world_size=parallel_state.get_data_parallel_world_size(),
        num_workers=args.num_workers,
        data_parallel_group=parallel_state.get_data_parallel_group(),
        worker_debug_path=None,
        worker_log_level=0,
    )

    if len(args.data_path) == 1:
        train_ds = energon.get_train_dataset(
            args.data_path[0],
            batch_size=args.micro_batch_size,
            task_encoder=task_encoder,
            worker_config=worker_config,
            max_samples_per_sequence=None,
            shuffle_buffer_size=None,
            packing_buffer_size=args.packing_buffer_size,
            handler=print_error_handler,
            image_decode="pil",
        )
    else:
        data_paths, data_weights = get_blend_from_list(args.data_path)
        yaml_path = create_metadataset_yaml(data_paths, data_weights, split="train")
        train_ds = energon.get_train_dataset(
            yaml_path,
            batch_size=args.micro_batch_size,
            task_encoder=task_encoder,
            worker_config=worker_config,
            max_samples_per_sequence=None,
            shuffle_buffer_size=None,
            packing_buffer_size=args.packing_buffer_size,
            handler=print_error_handler,
            image_decode="pil",
        )
    return train_ds


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
    yaml_path = os.path.join(temp_dir, f"metadataset_{os.getpid()}.yaml")

    with open(yaml_path, "w") as f:
        yaml.dump(metadataset_config, f, default_flow_style=False)

    return yaml_path


def get_train_loader(train_ds, collator=None):
    """Get the training loader"""
    args = get_args()
    from importlib.metadata import version
    if version('megatron-energon') < "7.0.0":
        train_dataloader = energon.get_savable_loader(train_ds)
    else:
        train_dataloader = energon.get_savable_loader(train_ds, watchdog_initial_timeout_seconds=600)
    
    if args.load is not None:
        if getattr(args, "dataloader_save", None):
            dp_rank = parallel_state.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                pipeline_rank=0,  # Only the first pipeline parallel rank stores the dataloader checkpoint.
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(data_save_name, map_location="cpu")
                    train_dataloader.restore_state_rank(
                        dataset_state_dict["dataloader_state_dict"]
                    )
                    print(f"restored dataset state from {data_save_name}")
                except Exception as e:
                    print("loading dataset state failed. Skipping. " + str(e))
            else:
                print(f"dataset state {data_save_name} does not exist")
    return EnergonDataloader(train_dataloader, collator)


class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""

    def __init__(self, dataloader, collator=None):
        self._dataloader = dataloader
        self._collator = collator
        self._iter = iter(cyclic_iter(dataloader))

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


def cyclic_iter(iter):
    """Infinite iteration over an iterator"""
    while True:
        for x in iter:
            yield x
