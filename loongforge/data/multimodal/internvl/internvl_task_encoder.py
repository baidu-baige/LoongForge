# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Preprocessing pipeline for InternVL models."""

from functools import partial
import torch
from typing import List, Union
import dataclasses
from dataclasses import dataclass, asdict
from typing_extensions import override
from loongforge.data.multimodal import MultiMixQASample
from megatron.energon import stateless
from ..base.task_encoder import (
    BaseTaskSample,
    BaseTaskSamplePacked,
    BaseTaskBatchPacked,
    BaseTaskEncoder,
)
from .internvl_preprocess import InternvlPreprocess, IGNORE_TOKEN_ID, IGNORE_INDEX

from importlib.metadata import version as _energon_version
try:
    _ENERGON_NEEDS_SUBFLAVOR = _energon_version("megatron-energon") < "7.0.0"
except Exception:
    _ENERGON_NEEDS_SUBFLAVOR = False
@dataclass
class MixQATaskSample(BaseTaskSample):
    """Dataclass to store a single unbatched sample."""

    position_ids: torch.Tensor = None
    loss_weight: torch.Tensor = None
    image_flags: torch.Tensor = None


@dataclass
class MixQATaskPackedSample(BaseTaskSamplePacked):
    """Dataclass to store a single packed sample."""

    position_ids: torch.Tensor = None
    loss_weight: torch.Tensor = None
    image_flags: torch.Tensor = None


@dataclass
class MixQATaskBatchPackedSample(BaseTaskBatchPacked):
    """Dataclass storing one batched sample"""

    position_ids: torch.Tensor = None
    loss_weight: torch.Tensor = None
    image_flags: torch.Tensor = None


class InternVLTaskEncoder(BaseTaskEncoder):
    """Task encoder for Intervl."""

    def __init__(self, args, tokenizer):
        super().__init__()
        # Current limitation: SFT-only
        assert args.training_phase == "sft"
        self.preproc = InternvlPreprocess(args, tokenizer)
        self.tokenizer = tokenizer
        self.loss_reduction = args.loss_reduction
        self.seq_length = args.seq_length
        self.strict_mode = args.strict_mode
        self.max_item_length = args.max_packed_tokens if self.strict_mode else 0


    def encode_multi_mix_qa(self, sample: MultiMixQASample) -> MixQATaskSample:
        """Encode multi_mix_qa sample."""
        # Convert standardized messages (role/content) back to internvl's expected format (from/value)
        _role_map: dict[str, str] = {"user": "human", "assistant": "gpt", "system": "system"}
        texts = []
        if sample.system is not None:
            texts.append({"from": "system", "value": sample.system})
        texts += [
            {"from": _role_map.get(msg["role"], msg["role"]), "value": msg["content"]}
            for msg in sample.messages
        ]
        data_item = {"texts": texts}
        # text + images
        if sample.image is not None:
            assert (
                sample.video is None
            ), "Mixed video and image content is not currently supported: sample text:{sample.texts}"
            data_item["image"] = sample.image
            ret = self.preproc.multi_image_get_item(data_item)
        # text + videos
        elif sample.video is not None:
            data_item["videos"] = sample.video
            ret = self.preproc.video_get_item(data_item)
        # only texts
        else:
            ret = self.preproc.pure_text_get_item(data_item)

        sample_args = {
            "__key__": sample.__key__,
            "__restore_key__": sample.__restore_key__,
            "__subflavors__": sample.__subflavors__,
            "tokens": ret["input_ids"],
            "labels": ret["labels"],
            "attn_mask": ret["attention_mask"],
            "position_ids": ret["position_ids"],
            "imgs": ret["pixel_values"],
            "total_len": len(ret["input_ids"]),
            "image_flags": ret["image_flags"],
        }
        if _ENERGON_NEEDS_SUBFLAVOR:
            sample_args["__subflavor__"] = None

        return MixQATaskSample(**sample_args)

    @override
    @stateless
    def pack_selected_samples(
        self, samples: List[MixQATaskSample]
    ) -> MixQATaskPackedSample:
        """Pack selected samples into one big sample."""
        packing_seq_len = self.seq_length
        packed_tokens = []
        packed_labels = []
        packed_masks = []
        packed_pixel_values = []
        packed_pos_ids = []
        packed_data_index = []
        packed_image_flags = []

        current_length = 0
        max_length = 0
        cu_lengths = [0]

        # used for calculate final cu_lengths
        for idx, sample in enumerate(samples):
            sample.cu_lengths = torch.zeros_like(sample.tokens).fill_(idx)

        # Process each sample and build lists that we will concatenate to create the packed sample.
        for _, sample in enumerate(samples):
            sample_len = sample.total_len

            if sample_len > max_length:
                max_length = sample_len

            if current_length + sample_len > packing_seq_len:
                raise ValueError(
                    f"Packed sample exceeds the maximum sequence length of {packing_seq_len}: {samples}"
                )

            # Add the sample's tokens and labels
            packed_tokens.append(sample.tokens)
            packed_labels.append(sample.labels)
            packed_masks.append(sample.attn_mask)
            packed_pos_ids.append(sample.position_ids)
            packed_data_index.append(sample.cu_lengths)
            packed_image_flags.append(sample.image_flags)
            # Add the images
            if sample.imgs is not None:
                packed_pixel_values += sample.imgs

            current_length += sample_len
            cu_lengths.append(current_length)

        # Concatenate packed tokens and labels.
        packed_tokens = torch.cat(packed_tokens, dim=0)
        packed_labels = torch.cat(packed_labels, dim=0)
        packed_masks = torch.cat(packed_masks, dim=0)
        packed_pos_ids = torch.cat(packed_pos_ids, dim=0)
        packed_data_index = torch.cat(packed_data_index, dim=0)
        packed_image_flags = torch.cat(packed_image_flags, dim=0)

        cu_lengths = torch.tensor(cu_lengths, dtype=torch.int32)

        # build_loss_weight
        _, _, curr_loss_weight = self.preproc.get_cu_seqlens_and_indexes(
            data_index=packed_data_index,
            input_ids=packed_tokens,
            labels=packed_labels,
            len2weight=partial(
                self.preproc.len2weight, loss_reduction=self.loss_reduction
            ),
        )
        curr_loss_weight = torch.where(
            packed_labels == IGNORE_TOKEN_ID, 0, curr_loss_weight
        )

        sample_kwargs = {
            "__key__": ",".join([s.__key__ for s in samples]),
            "__restore_key__": (),
            "__subflavors__": samples[0].__subflavors__,
            "tokens": packed_tokens,
            "labels": packed_labels,
            "attn_mask": cu_lengths,
            "position_ids": packed_pos_ids,
            "imgs": packed_pixel_values,
            "cu_lengths": cu_lengths,
            "loss_weight": curr_loss_weight,
            "image_flags": packed_image_flags,
            "max_length": max_length,
        }

        if _ENERGON_NEEDS_SUBFLAVOR:
            sample_kwargs["__subflavor__"] = None

        return MixQATaskPackedSample(**sample_kwargs)

    @override
    def batch(
        self, samples: List[Union[MixQATaskSample, MixQATaskPackedSample]]
    ) -> MixQATaskBatchPackedSample:
        """Batch samples together"""
        batch_lens = [feat.tokens.shape for feat in samples]
        max_item_length = self.max_item_length or max(batch_lens)[0]

        # pad imgs
        if self.strict_mode:
            for s in samples:
                self.preproc.pad_imgs(s, self.num_images_expected)
                if s.cu_lengths[-1] != max_item_length:
                    last = torch.tensor([max_item_length], dtype=s.cu_lengths.dtype)
                    s.cu_lengths = torch.cat([s.cu_lengths, last])

        features = [asdict(f) for f in samples]
        # align with llm
        pad_id = self.tokenizer.pad_token_id if isinstance(samples[0], MixQATaskSample) else 0

        # pad logic, pad all batches to the same length
        first = features[0]

        for idx in range(len(features)):
            feat = features[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_input_ids[: feat["tokens"].shape[0]] = feat["tokens"]
            feat["tokens"] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[: feat["labels"].shape[0]] = feat["labels"]
            feat["labels"] = temp_labels
            if isinstance(samples[0], MixQATaskSample):
                feat["attention_mask"] = feat["tokens"].ne(pad_id)
            else:  # pack
                assert (
                    feat["cu_lengths"] is not None
                ), f'pack mask error: {feat["cu_lengths"]}'
                feat["attn_mask"] = feat["cu_lengths"]

            if "position_ids" in feat:
                temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
                temp_position_ids[: feat["position_ids"].shape[0]] = feat[
                    "position_ids"
                ]
                feat["position_ids"] = temp_position_ids

            if "loss_weight" in feat and feat["loss_weight"] is not None:
                temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
                temp_loss_weight[: feat["loss_weight"].shape[0]] = feat["loss_weight"]
                feat["loss_weight"] = temp_loss_weight

        batch = MixQATaskBatchPackedSample(
            __key__=",".join([s.__key__ for s in samples]),
            __restore_key__=(),
            __subflavors__=samples[0].__subflavors__,
            tokens=None,
            labels=None,
            max_lengths=None,
            cu_lengths=None,
        )

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        if "label" in first and first["label"] is not None:
            label = (
                first["label"].item()
                if isinstance(first["label"], torch.Tensor)
                else first["label"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch.labels = torch.tensor([f["label"] for f in features], dtype=dtype)

        # concat to build batch, ['input_ids', 'labels', 'attention_mask', 'position_ids', 'loss_weight']
        batch.tokens = torch.stack([f["tokens"] for f in features])
        batch.labels = torch.stack([f["labels"] for f in features])
        batch.attn_mask = torch.stack([f["attn_mask"] for f in features])
        batch.position_ids = torch.stack([f["position_ids"] for f in features])
        if isinstance(samples[0], MixQATaskPackedSample):
            batch.loss_weight = torch.stack([f["loss_weight"] for f in features])

        # ['pixel_values', 'image_flags']
        batch.imgs = torch.stack([tensor for f in features for tensor in f["imgs"]])
        batch.image_flags = torch.stack(
            [tensor for f in features for tensor in f["image_flags"]]
        )

        return batch
