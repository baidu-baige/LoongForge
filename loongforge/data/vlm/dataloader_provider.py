# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dataset and DataLoader related utilities"""

import logging
import os
import tempfile
from dataclasses import dataclass
from math import gcd
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import yaml
from transformers.utils import PaddingStrategy

from megatron import energon
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.models.multimodal import context_parallel
from megatron.core.utils import log_single_rank
from loongforge.data import IGNORE_INDEX, TrainingPhase
from .base.task_encoder import print_error_handler
from loongforge.models.omni_models.get_position_idx_func import get_position_ids

logger = logging.getLogger(__name__)

PAD_TOKEN_ID = 151643


@dataclass(frozen=True)
class VLMDataConfig:
    """Data-recipe fields needed to build the Energon datasets and task encoders.

    Built by the training method; the data layer never reads backend arguments.
    Defaults match the CLI definitions in ``training/megatron_args.py`` and
    ``training/vlm/args.py``.
    """

    # Loader inputs.
    data_path: List[str]
    micro_batch_size: int
    num_workers: int
    seed: int
    seq_length: int
    valid_data_path: Optional[List[str]] = None
    packing_buffer_size: int = 10000
    data_shuffle_buffer_size: int = 0
    data_max_samples_per_sequence: int = 0

    # Dataset shape and preprocessing.
    add_question_in_pretrain: bool = False
    conv_style: str = "internvl2_5"
    down_sample_ratio: float = 0.5
    dynamic_image_size: bool = False
    enable_discard_sample: bool = False
    energon_pack_algo: str = "balanced"
    force_image_size: int = 448
    fps: float = 2.0
    fps_max_frames: int = 768
    fps_min_frames: int = 4
    frame_max_pixels: int = 768 * 28 * 28
    frame_min_pixels: int = 128 * 28 * 28
    group_by_length: bool = False
    hf_tokenizer_path: Optional[str] = None
    image_resolution: Optional[int] = None
    loss_reduction: str = "square"
    max_buffer_size: int = 20
    max_dynamic_patch: int = 12
    max_num_frame: int = 32
    max_packed_tokens: Optional[int] = None
    max_pixels: int = 16384 * 28 * 28
    min_dynamic_patch: int = 1
    min_num_frame: int = 8
    min_pixels: int = 4 * 28 * 28
    normalize_type: str = "imagenet"
    num_images_expected: int = 48
    packing_pretrain_data: bool = False
    packing_sft_data: bool = False
    pad2square: bool = False
    sample_type: Optional[str] = None
    sft_dataset: Optional[List[str]] = None
    sft_dataset_config: Optional[str] = None
    strict_mode: bool = False
    task_encoder: Optional[str] = None
    training_phase: str = TrainingPhase.PRETRAIN
    training_rice_vl_max_answer_length: int = 4096
    use_thumbnail: bool = False
    video_max_pixels: int = 65536 * 28 * 28

    # Fields read by the ERNIE encoder but not defined as CLI arguments.
    # Kept as ``Any`` so the existing call sites keep their current behaviour.
    image_dtype: Any = None
    image_token_len: Any = None
    pad: Any = None
    sft_image_normalize: Any = None
    sft_image_rescale: Any = None
    sft_replace_ids: Any = None
    sft_shift_by_one: Any = None

    @classmethod
    def from_args(cls, args) -> "VLMDataConfig":
        """Build the config from the training arguments."""
        encoder_fields = {
            name: getattr(args, name, None)
            for name in cls.__dataclass_fields__
            if name not in {
                "data_path",
                "valid_data_path",
                "micro_batch_size",
                "num_workers",
                "packing_buffer_size",
                "data_shuffle_buffer_size",
                "data_max_samples_per_sequence",
            }
        }
        return cls(
            data_path=args.data_path,
            valid_data_path=getattr(args, "valid_data_path", None),
            micro_batch_size=args.micro_batch_size,
            num_workers=args.num_workers,
            packing_buffer_size=getattr(args, "packing_buffer_size", 0),
            data_shuffle_buffer_size=getattr(args, "data_shuffle_buffer_size", 0),
            data_max_samples_per_sequence=getattr(args, "data_max_samples_per_sequence", 0),
            **encoder_fields,
        )


def _lcm(lhs, rhs):
    return lhs * rhs // gcd(lhs, rhs)


def _sequence_padding_factor(tp_size=1, cp_size=1, has_sp=False):
    if has_sp and cp_size > 1:
        return tp_size * cp_size * 2
    if cp_size > 1:
        return cp_size * 2
    if has_sp:
        return tp_size
    return 1


def _fp8_padding_factor(fp8_recipe=None):
    if fp8_recipe == "mxfp8":
        return 32
    if fp8_recipe == "blockwise":
        return 128
    return 16


def _has_packed_cu_lengths(batch: Dict[str, Any]) -> bool:
    """Return whether a collated batch carries real packed sequence boundaries."""
    cu_lengths = batch.get("cu_lengths")
    if cu_lengths is None:
        return False
    if not torch.is_tensor(cu_lengths):
        cu_lengths = torch.as_tensor(cu_lengths)
    return cu_lengths.ndim >= 2 and cu_lengths.shape[-1] > 1


def seq_padding_for_cp(
    data,
    tp_size=1,
    cp_size=1,
    has_sp=False,
    fp8_enabled=False,
    fp8_recipe=None,
    pad_token_id=PAD_TOKEN_ID,
):
    """Sequence padding for CP and/or SP

    Args:
        data (dict): Data from dataloader.
        tp_size (int): Tensor parallel size.
        cp_size (int): Context parallel size.
        has_sp (bool): Model uses sequence parallelism.
        fp8_enabled (bool): Model uses FP8 execution.
        fp8_recipe (str): FP8 recipe. Affects required padding.
        pad_token_id (int): Token id used to pad token slots that the
            attention mask will mark as padded. Defaults to the module-level
            ``PAD_TOKEN_ID`` when the caller cannot resolve a tokenizer-aware
            value.

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
        length = int(length)
        token = tokens[0, start : start + length]
        label = labels[0, start : start + length]
        mask = attn_mask[0, start : start + length]

        mp_padding_needed = context_parallel.get_padding(
            length, cp_size, tp_size, has_sp
        )

        input_ids = F.pad(token, (0, mp_padding_needed), "constant", pad_token_id)
        label = F.pad(label, (0, mp_padding_needed), "constant", IGNORE_INDEX)
        mask = F.pad(mask, (0, mp_padding_needed), "constant", True)

        valid_tokens.append(input_ids)
        valid_labels.append(label)
        valid_attn_mask.append(mask)

        cu_seqlens_padded.append(
            int(cu_seqlens_padded[-1] + length + mp_padding_needed)
        )

        start += length

    final_padding_factor = _sequence_padding_factor(tp_size, cp_size, has_sp)
    if fp8_enabled:
        fp8_padding_factor = _fp8_padding_factor(fp8_recipe)
        if has_sp:
            fp8_padding_factor *= tp_size
        final_padding_factor = _lcm(final_padding_factor, fp8_padding_factor)

    final_padding_needed = (
        int(
            (cu_seqlens_padded[-1] + final_padding_factor - 1)
            // final_padding_factor
            * final_padding_factor
        )
        - cu_seqlens_padded[-1]
    )

    if final_padding_needed > 0 and valid_tokens:
        valid_tokens[-1] = F.pad(
            valid_tokens[-1], (0, final_padding_needed), "constant", pad_token_id
        )
        valid_labels[-1] = F.pad(
            valid_labels[-1], (0, final_padding_needed), "constant", IGNORE_INDEX
        )
        valid_attn_mask[-1] = F.pad(
            valid_attn_mask[-1], (0, final_padding_needed), "constant", True
        )
        cu_seqlens_padded[-1] += final_padding_needed

    data["tokens"] = torch.cat(valid_tokens, dim=0).unsqueeze(0).to(tokens.dtype)
    data["labels"] = torch.cat(valid_labels, dim=0).unsqueeze(0).to(labels.dtype)
    data["attn_mask"] = torch.cat(valid_attn_mask, dim=0).unsqueeze(0).to(attn_mask.dtype)

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
    model_config: Any
    tp_size: int
    cp_size: int
    has_sp: bool
    fp8_enabled: bool
    fp8_recipe: Optional[str]
    packing_sft_data: bool
    model: Optional[Any] = None
    padding: Optional[PaddingStrategy] = None
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = IGNORE_INDEX
    return_tensors: str = "pt"

    def _resolve_pad_token_id(self) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        return PAD_TOKEN_ID if pad_token_id is None else pad_token_id

    def collate_energon(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw Energon batch tensors, pad for TP/CP configs, then build masks/positions."""
        batch = self._ensure_tensor(batch)
        self._pad_sequences(batch)
        is_packed = self.packing_sft_data or _has_packed_cu_lengths(batch)
        if is_packed and (
            self.cp_size > 1
            or self.has_sp
            or (self.fp8_enabled and self.fp8_recipe == "blockwise")
        ):
            seq_padding_for_cp(
                batch,
                tp_size=self.tp_size,
                cp_size=self.cp_size,
                has_sp=self.has_sp,
                fp8_enabled=self.fp8_enabled,
                fp8_recipe=self.fp8_recipe,
                pad_token_id=self._resolve_pad_token_id(),
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
        pad_token_id = self._resolve_pad_token_id()
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
        labels = batch["labels"]
        attention_mask = batch["attn_mask"]
        cu_lengths = batch["cu_lengths"]
        model_config = self.model_config
        get_position_ids_func = getattr(
            model_config, "position_idx_func", get_position_ids
        )
        position_ids, _ = get_position_ids_func(batch, model_config)
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
    """Get the training loader"""
    from importlib.metadata import version
    from packaging.version import Version
    if Version(version("megatron-energon")) < Version("7.0.0"):
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


def build_full_hetero_encoder_energon_iterator(
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
    from loongforge.data.vlm.encoder_strided_sampler import EncoderStridedIterator, PrefetchIterator

    strided_iter = EncoderStridedIterator(
        energon_dataloader=energon_dataloader,
        pp_rank=pp_rank,
        tp_size=tp_size,
        model_size=model_size,
        num_real_microbatch=num_real_microbatch,
    )
    return PrefetchIterator(strided_iter, prefetch_count=prefetch_count)
