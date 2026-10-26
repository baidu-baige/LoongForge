# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Collator for Energon VLM batches: padding for CP/SP/FP8 alignment, loss mask, position ids."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers.utils import PaddingStrategy

from loongforge import constants
from loongforge.models.multimodal.position_ids import get_position_ids

IGNORE_INDEX = constants.IGNORE_INDEX
# Fallback pad id (Qwen ``<|endoftext|>``) for tokenizers without a pad token.
PAD_TOKEN_ID = 151643


def _has_packed_cu_lengths(batch: Dict[str, Any]) -> bool:
    """Return whether a collated batch carries real packed sequence boundaries."""
    cu_lengths = batch.get("cu_lengths")
    if cu_lengths is None:
        return False
    if not torch.is_tensor(cu_lengths):
        cu_lengths = torch.as_tensor(cu_lengths)
    return cu_lengths.ndim >= 2 and cu_lengths.shape[-1] > 1


def seq_padding_for_cp(data, padding, pad_token_id=PAD_TOKEN_ID):
    """Pad every packed segment, then the packed total, as ``padding`` prescribes.

    Args:
        data (dict): Data from dataloader.
        padding: ``segment_padding(length)`` and ``total_multiple`` from the training
            method (``engines/megatron/data_padding.VlmPackedPadding``).
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

        mp_padding_needed = padding.segment_padding(length)

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

    final_padding_factor = padding.total_multiple

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
    # Set by the training method from parallelism and precision; None means no alignment.
    packed_padding: Optional[Any]
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
        if self.packed_padding is not None and is_packed:
            seq_padding_for_cp(batch, self.packed_padding, pad_token_id=self._resolve_pad_token_id())
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
