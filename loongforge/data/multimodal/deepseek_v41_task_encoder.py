# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Native DeepSeek-V4.1 multimodal task encoding."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from megatron.energon import CaptioningSample, VQASample
from transformers.utils import PaddingStrategy

from loongforge.data.multimodal.base.task_encoder import (
    _ENERGON_NEEDS_SUBFLAVOR,
    IGNORE_INDEX,
    BaseTaskBatchPacked,
    BaseTaskEncoder,
    BaseTaskSample,
)
from loongforge.data.multimodal.flavors import ChatMixSample
from loongforge.models.foundation.deepseek_v41.deepseek_v41_processing import (
    TEXT,
    DeepseekV41ImageProcessor,
)
from loongforge.train.get_position_idx_func import get_position_ids
from loongforge.utils import constants, get_model_config

DEEPSEEK_IMAGE_TOKEN = "<｜deepseek_image｜>"


@dataclass
class DeepseekV41TaskSample(BaseTaskSample):
    """One sample carrying the native V4.1 vision span metadata."""

    vision_token_types: Optional[torch.Tensor] = None
    image_grid_hws: Optional[torch.Tensor] = None
    pixel_values: Optional[torch.Tensor] = None


@dataclass
class DeepseekV41TaskBatch(BaseTaskBatchPacked):
    """A batch using the native V4.1 image contract."""

    vision_token_types: Optional[torch.Tensor] = None
    image_grid_hws: Optional[torch.Tensor] = None
    pixel_values: Optional[torch.Tensor] = None


class DeepseekV41TaskEncoder(BaseTaskEncoder):
    """Encode Energon samples with the native V4.1 processor."""

    def __init__(self, args):
        del args
        super().__init__()
        self.model_config = get_model_config()
        self.image_processor = DeepseekV41ImageProcessor(self.tokenizer.hf_tokenizer(), self.model_config)
        self.eos_token = self.tokenizer.tokenizer.eos_token

    def _make_sample(
        self,
        sample,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        vision_token_types: torch.Tensor,
        image_grid_hws: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
    ) -> DeepseekV41TaskSample:
        meta = {
            "__key__": sample.__key__,
            "__restore_key__": sample.__restore_key__,
            "__subflavors__": sample.__subflavors__,
        }
        if _ENERGON_NEEDS_SUBFLAVOR:
            meta["__subflavor__"] = None
        return DeepseekV41TaskSample(
            **meta,
            tokens=input_ids,
            labels=labels,
            attn_mask=torch.zeros_like(input_ids, dtype=torch.bool),
            total_len=input_ids.numel(),
            imgs=[],
            num_tiles=[len(image_grid_hws)] if image_grid_hws is not None else [0],
            vision_token_types=vision_token_types,
            image_grid_hws=image_grid_hws,
            pixel_values=pixel_values,
        )

    def _encode_segments(self, segments):
        """Encode independent text segments and concatenate native image spans."""
        ids, labels, types, patches, grids = [], [], [], [], []
        for text, images, train_targets in segments:
            encoded = self.image_processor(
                text,
                images=images,
                padding=False,
            )
            segment_ids = encoded["input_ids"][0]
            segment_types = encoded["vision_token_types"][0]
            segment_labels = segment_ids.clone() if train_targets else torch.full_like(segment_ids, IGNORE_INDEX)
            segment_labels[segment_types >= 0] = IGNORE_INDEX
            ids.append(segment_ids)
            labels.append(segment_labels)
            types.append(segment_types)
            if "pixel_values" in encoded:
                patches.append(encoded["pixel_values"])
                grids.append(encoded["image_grid_hws"])
        input_ids = torch.cat(ids)
        target = torch.cat(labels)
        vision_token_types = torch.cat(types)
        pixel_values = torch.cat(patches) if patches else None
        image_grid_hws = torch.cat(grids) if grids else None
        return input_ids, target, vision_token_types, image_grid_hws, pixel_values

    @staticmethod
    def _replace_image_tokens(text: str) -> str:
        return text.replace(constants.Placeholder.IMAGE, DEEPSEEK_IMAGE_TOKEN)

    def encode_captioning(self, sample: CaptioningSample) -> DeepseekV41TaskSample:
        text = DEEPSEEK_IMAGE_TOKEN + sample.caption + self.eos_token
        encoded = self._encode_segments([(text, [sample.image], True)])
        return self._make_sample(
            sample,
            input_ids=encoded[0],
            labels=encoded[1],
            vision_token_types=encoded[2],
            image_grid_hws=encoded[3],
            pixel_values=encoded[4],
        )

    def encode_vqa(self, sample: VQASample) -> DeepseekV41TaskSample:
        if self.args.training_phase == constants.TrainingPhase.PRETRAIN:
            if self.args.add_question_in_pretrain:
                text = self._replace_image_tokens(sample.context + sample.answers)
            else:
                text = DEEPSEEK_IMAGE_TOKEN + sample.answers
            text += self.eos_token
            encoded = self._encode_segments([(text, [sample.image], True)])
        else:
            segments = [
                (
                    self._replace_image_tokens(sample.context),
                    [sample.image],
                    False,
                ),
                (sample.answers + self.eos_token, [], True),
            ]
            encoded = self._encode_segments(segments)
        return self._make_sample(
            sample,
            input_ids=encoded[0],
            labels=encoded[1],
            vision_token_types=encoded[2],
            image_grid_hws=encoded[3],
            pixel_values=encoded[4],
        )

    def encode_chat_mix(self, sample: ChatMixSample) -> DeepseekV41TaskSample:
        if self.args.training_phase != constants.TrainingPhase.SFT:
            raise NotImplementedError("DeepSeek-V4.1 chat samples require SFT")
        return self.encode_multi_mix_qa(sample)

    def encode_multi_mix_qa(self, sample) -> DeepseekV41TaskSample:
        """Encode multi-mix conversations with native image spans."""
        pretrain = self.args.training_phase == constants.TrainingPhase.PRETRAIN
        messages = list(sample.messages)
        if sample.system:
            messages.insert(0, {"role": constants.DataRoles.SYSTEM, "content": sample.system})
        images = list(sample.image or [])
        image_cursor = 0
        segments = []
        for message in messages:
            role = message["role"]
            content = self._replace_image_tokens(message["content"])
            image_count = content.count(DEEPSEEK_IMAGE_TOKEN)
            message_images = images[image_cursor : image_cursor + image_count]
            image_cursor += image_count
            if role == constants.DataRoles.ASSISTANT:
                text = content + self.eos_token
                train_targets = True
            elif role == constants.DataRoles.USER:
                text = f"<｜User｜>{content}<｜Assistant｜></think>"
                train_targets = pretrain
            elif role == constants.DataRoles.SYSTEM:
                text = content
                train_targets = pretrain
            else:
                raise ValueError(f"Unsupported DeepSeek-V4.1 chat role: {role}")
            segments.append((text, message_images, train_targets))
        if image_cursor != len(images):
            raise ValueError("DeepSeek-V4.1 image placeholders do not match images")
        encoded = self._encode_segments(segments)
        return self._make_sample(
            sample,
            input_ids=encoded[0],
            labels=encoded[1],
            vision_token_types=encoded[2],
            image_grid_hws=encoded[3],
            pixel_values=encoded[4],
        )

    def encode_packed_captioning(self, sample):
        raise NotImplementedError("DeepSeek-V4.1 native packing is not enabled")

    def encode_packed_vqa(self, sample):
        raise NotImplementedError("DeepSeek-V4.1 native packing is not enabled")

    def encode_packed_multi_mix_qa(self, sample):
        raise NotImplementedError("DeepSeek-V4.1 native packing is not enabled")

    def encode_packed_chat_mix(self, sample):
        raise NotImplementedError("DeepSeek-V4.1 native packing is not enabled")

    def batch(self, samples: List[DeepseekV41TaskSample]) -> DeepseekV41TaskBatch:
        max_seq_len = min(max(len(sample.tokens) for sample in samples), self.args.seq_length)
        batch_size = len(samples)
        pad_id = self.tokenizer.pad
        tokens = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long)
        attn_mask = torch.ones((batch_size, max_seq_len), dtype=torch.bool)
        vision_token_types = torch.full((batch_size, max_seq_len), TEXT, dtype=torch.int32)
        patches, grids = [], []
        max_lengths = []
        for row, sample in enumerate(samples):
            if sample.pixel_values is not None and sample.tokens.numel() > self.args.seq_length:
                raise ValueError("DeepSeek-V4.1 image samples must fit seq_length; truncation can split an image span")
            length = min(max_seq_len, sample.tokens.numel())
            tokens[row, :length] = sample.tokens[:length]
            labels[row, :length] = sample.labels[:length]
            attn_mask[row, :length] = False
            vision_token_types[row, :length] = sample.vision_token_types[:length]
            max_lengths.append(length)
            if sample.pixel_values is not None:
                patches.append(sample.pixel_values)
                grids.append(sample.image_grid_hws)
        pixel_values = torch.cat(patches) if patches else None
        image_grid_hws = torch.cat(grids) if grids else None
        return DeepseekV41TaskBatch(
            __key__=[sample.__key__ for sample in samples],
            __restore_key__=[sample.__restore_key__ for sample in samples],
            __subflavors__=[sample.__subflavors__ for sample in samples],
            tokens=tokens,
            labels=labels,
            max_lengths=torch.tensor(max_lengths, dtype=torch.int32),
            cu_lengths=torch.tensor([[0, length] for length in max_lengths], dtype=torch.int32),
            attn_mask=attn_mask,
            imgs=None,
            num_tiles=torch.tensor(
                [len(sample.image_grid_hws) if sample.image_grid_hws is not None else 0 for sample in samples],
                dtype=torch.int32,
            ),
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
            vision_token_types=vision_token_types,
        )

    def encode_batch(self, batch: DeepseekV41TaskBatch) -> dict:
        raw = dataclasses.asdict(batch)
        raw.pop("__subflavors__", None)
        return raw


@dataclass
class DeepseekV41PretrainCollator:
    """Pad native V4.1 fields and expose a valid-token attention mask."""

    tokenizer: object
    padding: Optional[PaddingStrategy] = None
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = IGNORE_INDEX

    def collate_energon(self, batch: dict) -> dict:
        tokens = torch.as_tensor(batch["tokens"], dtype=torch.long)
        labels = torch.as_tensor(batch["labels"], dtype=torch.long)
        attn_mask = torch.as_tensor(batch["attn_mask"], dtype=torch.bool)
        vision_types = torch.as_tensor(batch["vision_token_types"], dtype=torch.int32)
        target_length = tokens.shape[1]
        if self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None:
            target_length = self.max_length
        if self.pad_to_multiple_of and target_length % self.pad_to_multiple_of:
            target_length = (
                (target_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
            )
        pad_len = target_length - tokens.shape[1]
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            tokens = F.pad(tokens, (0, pad_len), value=pad_id)
            labels = F.pad(labels, (0, pad_len), value=self.label_pad_token_id)
            attn_mask = F.pad(attn_mask, (0, pad_len), value=True)
            vision_types = F.pad(vision_types, (0, pad_len), value=TEXT)
        batch["tokens"] = tokens
        batch["labels"] = torch.roll(labels, shifts=-1, dims=1)
        batch["vision_token_types"] = vision_types
        batch["attn_mask"] = attn_mask
        batch["position_ids"], _ = get_position_ids(batch)
        batch["attn_mask"] = ~attn_mask
        target_types = torch.roll(vision_types, shifts=-1, dims=1)
        batch["loss_mask"] = (
            (batch["labels"] != self.label_pad_token_id) & (target_types < 0) & batch["attn_mask"]
        ).long()
        valid_lengths = batch["attn_mask"].sum(dim=1)
        for row, length in enumerate(valid_lengths.tolist()):
            if length:
                batch["loss_mask"][row, length - 1] = 0
        return batch
