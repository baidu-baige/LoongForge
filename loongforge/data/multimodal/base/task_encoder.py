# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tasks related to vision models."""

from abc import ABC, abstractmethod
import bisect
import dataclasses
import json
import re
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml

from PIL import Image
from torchvision.transforms import ToPILImage
import numpy as np
import torch

from loongforge.data.multimodal import (
    PackedCaptioningSample,
    PackedVQASample,
    PackedMultiMixQASample,
    MultiVidQASample,
    MultiMixQASample,
)

from megatron.energon import (
    Batch,
    CaptioningSample,
    DefaultTaskEncoder,
    OCRSample,
    Sample,
    SimilarityInterleavedSample,
    VQASample,
    MultiChoiceVQASample,
    Cooker,
)
from megatron.energon.task_encoder.base import stateless
from loongforge.utils import get_args, get_tokenizer
from .packer import Packer

IGNORE_INDEX = -100  # ID for labels that should be ignored.
from importlib.metadata import version as _energon_version
try:
    _ENERGON_NEEDS_SUBFLAVOR = _energon_version("megatron-energon") < "7.0.0"
except Exception:
    _ENERGON_NEEDS_SUBFLAVOR = False

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

_vlm_tags_cache: Optional[Dict[str, Dict]] = None


def _load_vlm_tags(section: Optional[str] = None) -> Dict[str, any]:
    """Load and cache the VLM message tags from the dataset config file.

    Reads the ``tags`` block under the given *section* in ``--sft-dataset-config``.
    Falls back to an empty dict (which triggers per-field defaults) when the
    config file is unavailable or the section is missing.

    To add a new data format, define a new named section in the config file and
    pass its name as *section* from the cooker.  Example config sections::

        # role/content fields, user/assistant values (default)
        multimodal:
          tags:
            role_tag: role
            content_tag: content

        # from/value fields, human/gpt values
        multimodal_sharegpt:
          tags:
            role_tag: from
            content_tag: value
            user_tag: human
            assistant_tag: gpt
            system_tag: system
    """
    global _vlm_tags_cache
    args = get_args()
    if section is None:
        section = args.sft_dataset[0] if args.sft_dataset else "multimodal"

    if _vlm_tags_cache is not None and section in _vlm_tags_cache:
        return _vlm_tags_cache[section]

    tags: Dict[str, any] = {}
    if args.sft_dataset_config:
        p = Path(args.sft_dataset_config)
        if p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f) or {}
            tags = cfg.get(section, {}).get("tags", {})

    if _vlm_tags_cache is None:
        _vlm_tags_cache = {}
    _vlm_tags_cache[section] = tags
    return tags


def _parse_messages(raw_messages, section: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
    """Parse a list of raw message dicts into (messages, system).

    Field names and role aliases are read directly from the ``tags`` block of
    *section* in ``--sft-dataset-config``.  No fallback field logic — whatever
    is configured in ``role_tag`` / ``content_tag`` is used as-is.

    For a new data format, add a new named section to the config and pass its
    name via the *section* argument from the relevant cooker function.

    Returns:
        messages: list of dicts with keys ``role`` and ``content``
        system:   system prompt string, or None
    """
    tags = _load_vlm_tags(section)
    role_tag = tags.get("role_tag", "role")
    content_tag = tags.get("content_tag", "content")
    role_map = {
        tags.get("user_tag", "user"): "user",
        tags.get("assistant_tag", "assistant"): "assistant",
        tags.get("system_tag", "system"): "system",
    }

    messages: List[Dict] = []
    system: Optional[str] = None

    for message in raw_messages:
        role = message.get(role_tag)
        content = message.get(content_tag, "")
        role = role_map.get(role, role)
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported role '{role}' in message: {message}")
        if role == "system":
            system = content
            continue
        messages.append({"role": role, "content": content})

    return messages, system


@stateless
def cooker_multi_mix_qa(sample: dict):
    """Convert raw sample dict into a MultiMixQASample. """
    messages, system = _parse_messages(sample["json"]["texts"])
    video = []
    image = []
    if sample["json"]["media"] == "video":
        for name in sample["json"]["name"]:
            video.append(sample.get(name))
    elif sample["json"]["media"] == "image":
        for name in sample["json"]["name"]:
            image.append(sample.get(name))

    if _ENERGON_NEEDS_SUBFLAVOR:
        return MultiMixQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample.get("__subflavors__", {}),
            video=video if len(video) > 0 else None,
            image=image if len(image) > 0 else None,
            system=system,
            messages=messages,
        )
    else:
        return MultiMixQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavors__=sample.get("__subflavors__", {}),
            video=video if len(video) > 0 else None,
            image=image if len(image) > 0 else None,
            system=system,
            messages=messages,
        )

@stateless
def cooker_multi_vid_vqa(sample: dict):
    """Convert raw sample dict into a MultiVidQASample. """
    messages, system = _parse_messages(sample["json"]["texts"])

    video = []
    image = []

    if sample["json"]["media"] == "video":
        for name in sample["json"]["name"]:
            video.append(sample.get(name))
    elif sample["json"]["media"] == "image":
        for name in sample["json"]["name"]:
            image.append(sample.get(name))

    if _ENERGON_NEEDS_SUBFLAVOR:
        return MultiVidQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample.get("__subflavors__", {}),
            video=video if len(video) > 0 else None,
            system=system,
            messages=messages,
        )
    else:
        return MultiVidQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavors__=sample.get("__subflavors__", {}),
            video=video if len(video) > 0 else None,
            system=system,
            messages=messages,
        )


@stateless
def cooker_feature_qa(sample: dict):
    """Convert raw sample dict into a FeatureQASample."""
    # TODO
    pass


@stateless
def cooker_packed_vqa(sample: dict):
    """Convert raw sample dict into a PackedCaptioningSample."""
    data = sample["json"]
    images = [sample.get(f"img{i}.jpg") for i in range(len(data["images"]))]
    captions = data["captions"]
    prompts = data["prompts"]
    if _ENERGON_NEEDS_SUBFLAVOR:
        return PackedVQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample.get("__subflavors__", {}),
            answers=captions,
            contexts=prompts,
            images=images,
        )
    else:
        return PackedVQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavors__=sample.get("__subflavors__", {}),
            answers=captions,
            contexts=prompts,
            images=images,
        )

@stateless
def cooker_packed_multi_mix_qa(sample: dict):
    """
    Convert packed multi-mix qa json into a PackedMultiMixQASample.

    Expected json layout (example):
    {
      "texts": {
        "captions": [...],   # len = N
        "prompts":  [...]    # len = N
      },
      "media_files": [      # length = N
        ["imgA.jpg", "imgB.jpg", ...],
        ["imgC.jpg", ...],
        ...
      ],
      "media_type": "image" | "video"
    }
    """
    data = sample["json"]

    texts = data.get("texts", {})
    prompts = texts.get("prompts", []) or []
    captions = texts.get("captions", []) or []

    if len(captions) != len(prompts):
        raise ValueError(
            f"[cooker_packed_multi_mix_qa] captions/prompts length mismatch for key={sample['__key__']}: "
            f"{len(captions)} vs {len(prompts)}"
        )

    # answers: List[List[str]]
    answers = [
        [c] if isinstance(c, str)
        else (list(c) if isinstance(c, (list, tuple)) else [])
        for c in captions
    ]

    media_files = data.get("media_files", []) or []
    media_type = (data.get("media_type") or "").lower()

    images = None
    videos = None


    if media_type == "image":
        images = []
        for group in media_files:
            image_group = []
            if isinstance(group, (list, tuple)):
                for name in group:
                    img = sample.get(name)
                    if img is not None:
                        image_group.append(img)
            elif isinstance(group, str):
                img = sample.get(group)
                if img is not None:
                    image_group.append(img)
            images.append(image_group)
        if all(len(g) == 0 for g in images):
            images = None
        videos = None
    elif media_type == "video":
        videos = []
        for group in media_files:
            video_group = []
            if isinstance(group, (list, tuple)):
                for name in group:
                    vid = sample.get(name)
                    if vid is not None:
                        video_group.append(vid)
            elif isinstance(group, str):
                vid = sample.get(group)
                if vid is not None:
                    video_group.append(vid)
            videos.append(video_group)

        if all(len(g) == 0 for g in videos):
            videos = None
        images = None

    else:
        raise ValueError(
            f"[cooker_packed_multi_mix_qa] unknown media_type='{media_type}'. "
            f"Expect 'image' or 'video'."
        )
    if _ENERGON_NEEDS_SUBFLAVOR:
        return PackedMultiMixQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample.get("__subflavors__", {}),
            images=images,
            videos=videos,
            contexts=prompts,
            answers=answers,
            answer_weights=None,
        )
    else:
        return PackedMultiMixQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavors__=sample.get("__subflavors__", {}),
            images=images,
            videos=videos,
            contexts=prompts,
            answers=answers,
            answer_weights=None,
        )

@stateless
def cooker_packed_caption(sample: dict):
    """Convert raw sample dict into a PackedCaptioningSample."""
    data = sample["json"]
    images = [sample.get(f"img{i}.jpg") for i in range(len(data["images"]))]
    captions = data["captions"]
    prompts = data["prompts"]
    if _ENERGON_NEEDS_SUBFLAVOR:
        return PackedCaptioningSample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample.get("__subflavors__", {}),
            captions=captions,
            prompts=prompts,
            images=images,
        )
    else:
        return PackedCaptioningSample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavors__=sample.get("__subflavors__", {}),
            captions=captions,
            prompts=prompts,
            images=images,
        )


def cooker_default(sample: dict):
    """Fallback cooker when no subflavor matches, selected by user-defined sample_type."""
    args = get_args()
    if args.sample_type == "multi_mix_qa":
        return cooker_multi_mix_qa(sample)
    elif args.sample_type == "feature_vqa":
        return cooker_feature_qa(sample)
    elif args.sample_type == "packed_captioning":
        return cooker_packed_caption(sample)
    elif args.sample_type == "packed_vqa":
        return cooker_packed_vqa(sample)
    elif args.sample_type == "packed_multi_mix_qa":
        return cooker_packed_multi_mix_qa(sample)
    else:
        raise NotImplementedError("Sample format not supported", sample)


class BaseTaskEncoder(DefaultTaskEncoder[BaseTaskSample, BaseTaskSamplePacked, BaseTaskBatchPacked, dict]):
    """A simple task encoder for VLMs."""
    cookers = [
        Cooker(cooker_multi_mix_qa, has_subflavors={"sample_type": "multi_mix_qa"}),
        Cooker(cooker_multi_vid_vqa, has_subflavors={"sample_type": "multi_vid_vqa"}),
        Cooker(cooker_feature_qa, has_subflavors={"sample_type": "feature_vqa"}),
        Cooker(cooker_packed_caption, has_subflavors={"sample_type": "packed_captioning"}),
        Cooker(cooker_packed_vqa, has_subflavors={"sample_type": "packed_vqa"}),
        Cooker(cooker_packed_multi_mix_qa, has_subflavors={"sample_type": "packed_multi_mix_qa"}),
        Cooker(cooker_default,)
    ]

    def __init__(self):
        super().__init__()

        self.args = get_args()

        self.packer = Packer(self.args)
        self.tokenizer = get_tokenizer()
        self.is_packing_enabled = self.args.packing_pretrain_data or self.args.packing_sft_data
        self.max_packed_tokens = self.args.max_packed_tokens
        self.num_images_expected = self.args.num_images_expected
        self.max_buffer_size = self.args.max_buffer_size

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: Union[CaptioningSample, VQASample, MultiVidQASample, MultiMixQASample]):
        """Generates an encoded sample from a raw sample."""
        assert not (
            self.args.packing_sft_data
            and isinstance(sample, (PackedCaptioningSample, PackedVQASample, PackedMultiMixQASample))
        ), (
            f"Configuration conflict: --packing-sft-data is enabled (online packing), "
            f"but the dataset contains offline-packed samples of type '{type(sample).__name__}'. "
            f"Either disable --packing-sft-data to use offline-packed data, "
            f"or switch to a non-packed dataset for online packing."
        )
        if isinstance(sample, CaptioningSample):
            yield self.encode_captioning(sample)
        elif isinstance(sample, VQASample):
            yield self.encode_vqa(sample)
        elif isinstance(sample, MultiVidQASample):
            yield self.encode_multi_vid_qa(sample)
        elif isinstance(sample, MultiMixQASample):
            yield self.encode_multi_mix_qa(sample)
        elif isinstance(sample, PackedCaptioningSample):
            yield self.encode_packed_captioning(sample)
        elif isinstance(sample, PackedVQASample):
            yield self.encode_packed_vqa(sample)
        elif isinstance(sample, PackedMultiMixQASample):
            yield self.encode_packed_multi_mix_qa(sample)
        else:
            raise NotImplementedError("Sample format not supported", sample)

    def encode_captioning(self, sample: CaptioningSample) -> BaseTaskSample:
        """Generates an encoded captioning sample from a raw sample."""
        raise NotImplementedError("encode_captioning not supported", sample)

    def encode_vqa(self, sample: VQASample) -> BaseTaskSample:
        """Generates an encoded vqa sample from a raw sample."""
        raise NotImplementedError("encode_vqa not supported", sample)

    def encode_multi_mix_qa(self, sample: MultiMixQASample) -> BaseTaskSample:
        """Generates an encoded multi_mix_qa sample from a raw sample."""
        raise NotImplementedError("encode_multi_mix_qa not supported", sample)

    def encode_multi_vid_qa(self, sample: MultiVidQASample) -> BaseTaskSample:
        """Generates an encoded vid_qa sample from a raw sample."""
        raise NotImplementedError("encode_multi_vid_qa not supported", sample)


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
        labels = np.full((len(samples), max_seq_len), IGNORE_INDEX, dtype=np.int64)
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
                raise ValueError(f"Packed sample exceeds the maximum sequence length of {packing_seq_len}: {samples}")

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

        if _ENERGON_NEEDS_SUBFLAVOR:
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
    """Print error handler function called when an exception occurs during loading."""
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()
