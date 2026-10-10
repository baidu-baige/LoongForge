# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tasks related to vision models."""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from loongforge import constants
from loongforge.data.vlm.flavors import (
    ENERGON_LT_7,
    PackedCaptioningSample,
    PackedVQASample,
    PackedMultiMixQASample,
    PackedChatMixSample,
    MultiVidQASample,
    MultiMixQASample,
    ChatMixSample,
)

from megatron.energon import (
    Batch,
    CaptioningSample,
    DefaultTaskEncoder,
    Sample,
    VQASample,
    Cooker,
)
from megatron.energon.task_encoder.base import stateless
from .packer import Packer
from .cookers import (
    bind_args,
    cooker_chat_mix,
    cooker_default,
    cooker_feature_qa,
    cooker_multi_mix_qa,
    cooker_multi_vid_vqa,
    cooker_packed_caption,
    cooker_packed_chat_mix,
    cooker_packed_multi_mix_qa,
    cooker_packed_vqa,
)


@dataclass
class BaseTaskSample(Sample):
    """Dataclass to store a single unbatched sample."""

    __key__: str
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    __subflavors__: Dict
    tokens: torch.Tensor
    # Total token count in the sample, including text and image tokens
    total_len: int
    labels: torch.Tensor = None
    attn_mask: torch.Tensor = None
    # (c, h, w)
    imgs: List[torch.Tensor] = None
    num_tiles: Optional[List[int]]= None
    pixel_values_videos: Optional[torch.Tensor]= None


@dataclass
class BaseTaskSamplePacked(Sample):
    """Dataclass to store a single packed sample (not a batch).

    P = Number of sub-samples in the packed sample
    seq_len = Total sequence length
    num_imgs = Number of images across all samples in the packed sample
    """
    # Sample name
    __key__: str
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    # Sample metadata.
    __subflavors__: Dict
    # Input tokens packed into a single tensor (seq_len,)
    tokens: torch.Tensor
    # Target tokens packed into a single tensor (seq_len,)
    labels: torch.Tensor
    # Maximum length across sub-samples.
    max_length: int
    # Cumulative length of each sub-sample in this packed sample incl. text and image tokens (P,)
    cu_lengths: List[int]
    attn_mask: torch.Tensor = None
    # Input images
    imgs: List[torch.Tensor] = None
    num_tiles: Optional[List[int]]= None
    pixel_values_videos: Optional[torch.Tensor]= None

    def __repr__(self):
        def _shape(x):
            try:
                return tuple(x.shape)
            except Exception:
                return None

        def _list_shapes(lst, n=2):
            if lst is None:
                return None
            return [_shape(x) for x in lst[:n]]

        return (
            f"{self.__class__.__name__}("
            f"key={self.__key__}, "
            f"tokens={_shape(self.tokens)}, "
            f"labels={_shape(self.labels)}, "
            f"attn_mask={_shape(self.attn_mask)}, "
            f"imgs(first2)={_list_shapes(self.imgs)}, "
            f"num_tiles={self.num_tiles})"
        )


# Typing for the resulting batch data after encode_batch()
@dataclass
class BaseTaskBatchPacked(Batch):
    """Dataclass to store a batch of packed samples.

    N = Batch size
    P = Number of samples in the packed sample
    seq_len = Maximum sequence length
    num_imgs = Number of images across all samples in the packed sample
    """

    __key__: List[str]  # Sample names
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    # Sample metadatas.
    __subflavors__: List[Dict]
    # Input tokens packed and padded (N, seq_len)
    tokens: torch.Tensor
    # Target tokens packed and padded (N, seq_len)
    labels: torch.Tensor
    # Maximum length across sub-samples (N,)
    max_lengths: List[int]
    # Cumulative length of each sub-sample in each packed sample of the batch (N, P)
    cu_lengths: List[List[int]]
    attn_mask: torch.Tensor = None
    # All image tiles stacked into a single tensor (num_tiles, C, H, W)
    imgs: torch.Tensor = None
    num_tiles: Optional[List[int]]= None
    pixel_values_videos: Optional[torch.Tensor]= None


def _format_packed_sample_overflow_error(
    samples: List[BaseTaskSample],
    packing_seq_len: int,
    current_length: int,
    next_sample: BaseTaskSample,
    max_items: int = 8,
) -> str:
    """Build a compact error without dumping full tensor values."""

    def _shape(value):
        return tuple(value.shape) if hasattr(value, "shape") else None

    summary = []
    cumulative_length = 0
    for sample in samples[:max_items]:
        cumulative_length += sample.total_len
        summary.append(
            {
                "key": sample.__key__,
                "total_len": sample.total_len,
                "cumulative_len": cumulative_length,
                "tokens_shape": _shape(sample.tokens),
                "labels_shape": _shape(sample.labels),
                "attn_mask_shape": _shape(sample.attn_mask),
                "num_images": len(sample.imgs) if sample.imgs is not None else 0,
                "num_videos": (
                    len(sample.pixel_values_videos)
                    if sample.pixel_values_videos is not None
                    else 0
                ),
            }
        )

    omitted = max(len(samples) - max_items, 0)
    omitted_msg = f", omitted_samples={omitted}" if omitted else ""
    return (
        "Packed sample exceeds the maximum sequence length of "
        f"{packing_seq_len}: current_length={current_length}, "
        f"next_sample_key={next_sample.__key__}, "
        f"next_sample_len={next_sample.total_len}, "
        f"would_be_length={current_length + next_sample.total_len}, "
        f"num_samples={len(samples)}{omitted_msg}, samples_summary={summary}"
    )

class BaseTaskEncoder(DefaultTaskEncoder[BaseTaskSample, BaseTaskSamplePacked, BaseTaskBatchPacked, dict]):
    """A simple task encoder for VLMs."""

    def __init__(self, args, tokenizer, chat_template=None):
        super().__init__()
        self.cookers = [
            Cooker(bind_args(cooker_multi_mix_qa, args), has_subflavors={"sample_type": "multi_mix_qa"}),
            Cooker(bind_args(cooker_chat_mix, args), has_subflavors={"sample_type": "chat_mix"}),
            Cooker(bind_args(cooker_multi_vid_vqa, args), has_subflavors={"sample_type": "multi_vid_vqa"}),
            Cooker(cooker_feature_qa, has_subflavors={"sample_type": "feature_vqa"}),
            Cooker(cooker_packed_caption, has_subflavors={"sample_type": "packed_captioning"}),
            Cooker(cooker_packed_vqa, has_subflavors={"sample_type": "packed_vqa"}),
            Cooker(cooker_packed_multi_mix_qa, has_subflavors={"sample_type": "packed_multi_mix_qa"}),
            Cooker(cooker_packed_chat_mix, has_subflavors={"sample_type": "packed_chat_mix"}),
            Cooker(bind_args(cooker_default, args)),
        ]
        self.args = args

        self.packer = Packer(self.args)
        self.tokenizer = tokenizer
        self.is_packing_enabled = self.args.packing_pretrain_data or self.args.packing_sft_data
        self.max_packed_tokens = self.args.max_packed_tokens
        self.num_images_expected = self.args.num_images_expected
        self.max_buffer_size = self.args.max_buffer_size

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: Union[CaptioningSample, VQASample, MultiVidQASample, MultiMixQASample]):
        """Generates an encoded sample from a raw sample."""
        assert not (
            self.args.packing_sft_data
            and isinstance(sample, (
                PackedCaptioningSample,
                PackedVQASample,
                PackedMultiMixQASample,
                PackedChatMixSample,
            ))
        ), (
            f"Configuration conflict: --packing-sft-data is enabled (online packing), "
            f"but the dataset contains offline-packed samples of type '{type(sample).__name__}'. "
            f"Either disable --packing-sft-data to use offline-packed data, "
            f"or switch to a non-packed dataset for online packing."
        )
        if isinstance(sample, CaptioningSample):
            encoded = self.encode_captioning(sample)
        elif isinstance(sample, VQASample):
            encoded = self.encode_vqa(sample)
        elif isinstance(sample, MultiVidQASample):
            encoded = self.encode_multi_vid_qa(sample)
        elif isinstance(sample, ChatMixSample):
            encoded = self.encode_chat_mix(sample)
        elif isinstance(sample, MultiMixQASample):
            encoded = self.encode_multi_mix_qa(sample)
        elif isinstance(sample, PackedCaptioningSample):
            encoded = self.encode_packed_captioning(sample)
        elif isinstance(sample, PackedVQASample):
            encoded = self.encode_packed_vqa(sample)
        elif isinstance(sample, PackedMultiMixQASample):
            encoded = self.encode_packed_multi_mix_qa(sample)
        elif isinstance(sample, PackedChatMixSample):
            encoded = self.encode_packed_chat_mix(sample)
        else:
            raise NotImplementedError("Sample format not supported", sample)
        if encoded is not None:  # An encoder returns None to drop the sample.
            yield encoded

    def encode_captioning(self, sample: CaptioningSample) -> BaseTaskSample:
        """Generates an encoded captioning sample from a raw sample."""
        raise NotImplementedError("encode_captioning not supported", sample)

    def encode_vqa(self, sample: VQASample) -> BaseTaskSample:
        """Generates an encoded vqa sample from a raw sample."""
        raise NotImplementedError("encode_vqa not supported", sample)

    def encode_multi_mix_qa(self, sample: MultiMixQASample) -> BaseTaskSample:
        """Generates an encoded multi_mix_qa sample from a raw sample."""
        raise NotImplementedError("encode_multi_mix_qa not supported", sample)

    def encode_chat_mix(self, sample: ChatMixSample) -> BaseTaskSample:
        """Generates an encoded chat-format multimodal sample (with tool calling)."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement encode_chat_mix to use "
            f"chat_mix or packed_chat_mix sample types.",
            sample,
        )

    def encode_multi_vid_qa(self, sample: MultiMixQASample) -> BaseTaskSample:
        """Generates an encoded multimodal mix sample from a raw sample."""
        raise NotImplementedError("encode_multi_vid_qa not supported", sample)


    def encode_packed_captioning(self, sample: PackedCaptioningSample) -> BaseTaskSample:
        """Generates an encoded multimodal packed captioning sample from a raw sample."""
        raise NotImplementedError("encode_packed_captioning not supported", sample)


    def encode_packed_vqa(self, sample: PackedVQASample) -> BaseTaskSample:
        """Generates an encoded multimodal packed vqa sample from a raw sample."""
        raise NotImplementedError("encode_packed_vqa not supported", sample)

    def encode_packed_multi_mix_qa(self, sample: PackedMultiMixQASample) -> BaseTaskSample:
        """Generates an encoded multimodal packed multimix sample from a raw sample."""
        raise NotImplementedError("encode_packed_multi_mix_qa not supported", sample)

    def encode_packed_chat_mix(self, sample: PackedChatMixSample) -> BaseTaskSample:
        """Generates an encoded multimodal packed chat (incl. tool calling) sample from a raw sample."""
        raise NotImplementedError("encode_packed_chat_mix not supported", sample)

    def process_images(self, samples: List[Union[BaseTaskSample, BaseTaskSamplePacked]]) -> torch.Tensor:
        """Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image."""
        imgs = [img for s in samples for img in s.imgs]
        if len(imgs) > 0:
            return torch.stack(imgs)
        else:
            return torch.tensor([[0]], dtype=torch.float32)

    def process_videos(self, samples: List[Union[BaseTaskSample, BaseTaskSamplePacked]]) \
                                                                                    -> torch.Tensor:
        """"Process the data to get the model's input"""
        pixel_values_videos = [pixel_values_video for s in samples if s.pixel_values_videos is not None \
                for pixel_values_video in s.pixel_values_videos]
        if len(pixel_values_videos) > 0:
            return torch.cat(pixel_values_videos)
        else:
            return torch.tensor([[0]], dtype=torch.float32)


    def batch(self, samples: List[Union[BaseTaskSample, BaseTaskSamplePacked]]) -> BaseTaskBatchPacked:
        """Generates a batched version of the provided samples."""
        imgs = self.process_images(samples)
        pixel_values_videos = self.process_videos(samples)

        max_seq_len = max(len(s.tokens) for s in samples)
        max_seq_len = min(max_seq_len, self.args.seq_length)

        tokens = np.full((len(samples), max_seq_len), self.tokenizer.pad, dtype=np.int64)
        labels = np.full((len(samples), max_seq_len), constants.IGNORE_INDEX, dtype=np.int64)
        attn_masks = np.full((len(samples), max_seq_len), True, dtype=bool)

        for i, s in enumerate(samples):
            # If the sample/target length exceeds the target sequence length, then truncate.
            text_len = min(max_seq_len, len(s.tokens))
            target_len = min(max_seq_len, len(s.labels))

            tokens[i, :text_len] = s.tokens[:text_len]
            labels[i, :target_len] = s.labels[:target_len]
            attn_masks[i, :text_len] = s.attn_mask[:text_len]

        num_tiles = [n for s in samples for n in s.num_tiles]
        if len(num_tiles) > 0:
            num_tiles = torch.tensor(num_tiles, dtype=torch.int32)
        else:
            num_tiles = torch.tensor([[0]], dtype=torch.int32)

        # Cumulative sample lengths are needed for packing, otherwise use dummy values.
        cu_lengths = torch.tensor([[0]], dtype=torch.int32)
        max_lengths = torch.tensor([[0]], dtype=torch.int32)

        if self.is_packing_enabled:
            cu_lengths = torch.stack([s.cu_lengths for s in samples])
            max_lengths = torch.tensor([s.max_length for s in samples], dtype=torch.int32)

        return BaseTaskBatchPacked(
            __key__=[s.__key__ for s in samples],
            __restore_key__=[s.__restore_key__ for s in samples],
            __subflavors__=samples[0].__subflavors__,
            tokens=tokens,
            labels=labels,
            attn_mask=attn_masks,
            imgs=imgs,
            pixel_values_videos=pixel_values_videos,
            num_tiles=num_tiles,
            cu_lengths=cu_lengths,
            max_lengths=max_lengths,
        )

    def encode_batch(self, batch: BaseTaskBatchPacked) -> dict:
        """Generates a dictionary containing the data required by the model."""
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw

    def select_samples_to_pack(self, samples: List[BaseTaskSample]) -> List[List[BaseTaskSample]]:
        """Selects which samples will be packed together.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html
        """
        packed_samples = self.packer.pack(samples, self.max_packed_tokens,
                        self.num_images_expected, self.max_buffer_size)

        return packed_samples

    @stateless
    def pack_selected_samples(self, samples: List[BaseTaskSample]) -> List[BaseTaskSamplePacked]:
        """
        Function to pack a list of BaseTaskSample into a single BaseTaskSamplePacked.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html

        Args:
            samples: List of BaseTaskSample instances to pack into one sample.

        Returns:
            BaseTaskSamplePacked instance.
        """

        packing_seq_len = self.args.seq_length

        packed_tokens = []
        packed_labels = []
        packed_masks = []
        packed_imgs = []
        packed_videos = []

        current_length = 0
        max_length = 0
        cu_lengths = [0]

        # Process each sample and build lists that we will concatenate to create the packed sample.
        for _, sample in enumerate(samples):
            sample_len = sample.total_len

            if sample_len > max_length:
                max_length = sample_len

            # If adding this sample exceeds the max length, stop.
            # This should not happen.
            # The select_samples_to_pack method should have already ensured that the samples fit.
            if current_length + sample_len > packing_seq_len:
                raise ValueError(
                    _format_packed_sample_overflow_error(
                        samples, packing_seq_len, current_length, sample
                    )
                )

            # Add the sample's tokens and labels
            packed_tokens.append(sample.tokens)
            packed_labels.append(sample.labels)
            packed_masks.append(sample.attn_mask)

            # Add the images
            if sample.imgs is not None:
                packed_imgs += sample.imgs
            if sample.pixel_values_videos is not None:
                packed_videos += sample.pixel_values_videos

            current_length += sample_len
            cu_lengths.append(current_length)

        # Concatenate packed tokens and labels.
        packed_tokens = torch.cat(packed_tokens, dim=0)
        packed_labels = torch.cat(packed_labels, dim=0)
        packed_masks = torch.cat(packed_masks, dim=0)

        if ENERGON_LT_7:
            return BaseTaskSamplePacked(
                __key__=",".join([s.__key__ for s in samples]),
                __restore_key__=(),  # Will be set by energon based on `samples`
                __subflavor__=None,
                __subflavors__=samples[0].__subflavors__,
                tokens=packed_tokens,
                labels=packed_labels,
                attn_mask=packed_masks,
                imgs=packed_imgs,
                pixel_values_videos=packed_videos,
                cu_lengths=torch.tensor(cu_lengths, dtype=torch.int32),
                max_length=max_length,
                num_tiles=[n for s in samples for n in s.num_tiles],
            )
        else:
            return BaseTaskSamplePacked(
                __key__=",".join([s.__key__ for s in samples]),
                __restore_key__=(),  # Will be set by energon based on `samples`
                __subflavors__=samples[0].__subflavors__,
                tokens=packed_tokens,
                labels=packed_labels,
                attn_mask=packed_masks,
                imgs=packed_imgs,
                pixel_values_videos=packed_videos,
                cu_lengths=torch.tensor(cu_lengths, dtype=torch.int32),
                max_length=max_length,
                num_tiles=[n for s in samples for n in s.num_tiles],
            )


def print_error_handler(exc: Exception, key: Optional[str]):
    """Log dataloader sample errors and let Energon skip the sample."""
    logging.warning(
        "skip dataloader sample %s due to %s: %s",
        key,
        type(exc).__name__,
        exc,
    )
