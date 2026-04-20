# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Kimi Task Encoder."""

import torch
from loongforge.data.multimodal.vlm_task_encoder import VLMTaskEncoder
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import override
from dataclasses import dataclass

from megatron.energon import (
    CaptioningSample,
    VQASample,
)
from importlib.metadata import version

if version("megatron-energon") < "7.0.0":
    from megatron.energon.flavors.webdataset import VideoData as AVData

    _ENERGON_NEEDS_SUBFLAVOR = True
else:
    from megatron.energon.flavors.webdataset import AVData

    _ENERGON_NEEDS_SUBFLAVOR = False


from loongforge.utils import constants, get_chat_template
from .base.task_encoder import (
    BaseTaskEncoder,
    BaseTaskSample,
    BaseTaskSamplePacked,
    BaseTaskBatchPacked,
)
from .vlm_task_encoder import VLMTaskSample

IGNORE_INDEX = -100  # ID for labels that should be ignored.

# Kimi K2.5 special tokens
MEDIA_BEGIN = "<|media_begin|>"
MEDIA_END = "<|media_end|>"
MEDIA_CONTENT = "<|media_content|>"
MEDIA_PAD = "<|media_pad|>"

# For image: <|media_begin|>image<|media_content|><|media_pad|><|media_end|>
IMAGE_TOKEN_WITH_TAGS = f"{MEDIA_BEGIN}image{MEDIA_CONTENT}{MEDIA_PAD}{MEDIA_END}"
# For video chunk: timestamp<|media_begin|>video<|media_content|><|media_pad|><|media_end|>
VIDEO_TOKEN_WITH_TAGS = f"{MEDIA_BEGIN}video{MEDIA_CONTENT}{MEDIA_PAD}{MEDIA_END}"

# Kimi chat template special tokens
IM_USER = "<|im_user|>"
IM_ASSISTANT = "<|im_assistant|>"
IM_MIDDLE = "<|im_middle|>"
IM_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"


class KimiVLMTaskEncoder(VLMTaskEncoder):
    """VLM Task Encoder for Kimi K2.5 models.

    Kimi K2.5 uses a different tokenization format:
    - Image: <|media_begin|>image<|media_content|><|media_pad|><|media_end|>
    - Video chunk: timestamp<|media_begin|>video<|media_content|><|media_pad|><|media_end|>
    - Chat template: <|im_user|>user<|im_middle|>...<|im_end|><|im_assistant|>assistant\
        <|im_middle|><think></think>...<|im_end|>

    This encoder also expands the single <|media_content|> placeholder token to multiple
    tokens based on the actual image feature length (computed from grid_thws), which is
    the functionality of _merge_input_ids_with_image_features in modeling_kimi_k25.py
    """

    def __init__(self, args):
        super().__init__(args)

        # Initialize chat_template for SFT phase
        if args.training_phase in ['sft']:
            self.chat_template = get_chat_template()

        # Get merge_kernel_size from processor config, default to [2, 2]
        merge_kernel_size = 2  # default
        if (
            hasattr(self.processor, "media_processor")
            and self.processor.media_processor is not None
        ):
            media_proc_cfg = getattr(
                self.processor.media_processor, "media_proc_cfg", {}
            )
            if isinstance(media_proc_cfg, dict):
                merge_kernel_size = media_proc_cfg.get("merge_kernel_size", 2)
            else:
                merge_kernel_size = getattr(media_proc_cfg, "merge_kernel_size", 2)

        if isinstance(merge_kernel_size, int):
            self.merge_kernel_size = [merge_kernel_size, merge_kernel_size]
        else:
            self.merge_kernel_size = list(merge_kernel_size)

    def _get_vision_token_ids(self):
        """Get special token IDs for vision processing."""
        media_begin_id = self.tokenizer.convert_tokens_to_ids(MEDIA_BEGIN)
        media_end_id = self.tokenizer.convert_tokens_to_ids(MEDIA_END)
        media_content_id = self.tokenizer.convert_tokens_to_ids(MEDIA_CONTENT)
        media_pad_id = self.tokenizer.convert_tokens_to_ids(MEDIA_PAD)
        return media_begin_id, media_end_id, media_content_id, media_pad_id

    def _compute_image_tokens_from_grid_thw(self, grid_thw: torch.Tensor) -> int:
        """Compute the number of image tokens from grid_thw after merging.

        Args:
            grid_thw: Tensor of shape (3,) containing [T, H, W] where H and W are
                     in patch units (H = height // patch_size, W = width // patch_size)

        Returns:
            Number of tokens after spatial downsampling with temporal pooling.
            Formula: (H // merge_h) * (W // merge_w)
            Note: T dimension is pooled away (temporal pooling)
        """
        t, h, w = grid_thw.tolist()
        merge_h, merge_w = self.merge_kernel_size
        new_height = h // merge_h
        new_width = w // merge_w
        # Temporal dimension is pooled, so only spatial dimensions matter
        return new_height * new_width

    def _expand_media_content_tokens(
        self,
        input_ids: torch.Tensor,
        target: torch.Tensor,
        attn_mask: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand single <|media_content|> tokens to multiple tokens based on image feature length.

        This implements the core logic of _merge_input_ids_with_image_features from
        modeling_kimi_k25.py, but operates on token IDs instead of embeddings.

        Args:
            input_ids: Token IDs with single <|media_content|> placeholders
            target: Labels tensor
            attn_mask: Attention mask tensor
            grid_thws: Grid dimensions for each image, shape (num_images, 3)

        Returns:
            Expanded input_ids, target, and attn_mask tensors
        """
        media_begin_id, media_end_id, media_content_id, media_pad_id = (
            self._get_vision_token_ids()
        )

        # Handle case where grid_thws is 1D (single image)
        if grid_thws.dim() == 1:
            grid_thws = grid_thws.unsqueeze(0)

        # Compute feature lengths for each image
        feature_lengths = [
            self._compute_image_tokens_from_grid_thw(thw) for thw in grid_thws
        ]

        input_ids_list = input_ids.tolist()
        target_list = target.tolist()
        attn_mask_list = attn_mask.tolist()

        new_input_ids = []
        new_target = []
        new_attn_mask = []

        image_idx = 0
        i = 0
        while i < len(input_ids_list):
            token_id = input_ids_list[i]

            if token_id == media_content_id:
                # Found <|media_content|> token - expand it
                if image_idx < len(feature_lengths):
                    num_tokens = feature_lengths[image_idx]
                    # Add num_tokens copies of media_content_id
                    new_input_ids.extend([media_content_id] * num_tokens)
                    new_target.extend([IGNORE_INDEX] * num_tokens)
                    new_attn_mask.extend(
                        [False] * num_tokens
                    )  # Not masked for attention
                    image_idx += 1
                else:
                    # No more images, keep original token
                    new_input_ids.append(token_id)
                    new_target.append(target_list[i])
                    new_attn_mask.append(attn_mask_list[i])
            else:
                # Regular token - keep as is
                new_input_ids.append(token_id)
                new_target.append(target_list[i])
                new_attn_mask.append(attn_mask_list[i])

            i += 1

        # Convert back to tensors
        expanded_input_ids = torch.tensor(new_input_ids, dtype=input_ids.dtype)
        expanded_target = torch.tensor(new_target, dtype=target.dtype)
        expanded_attn_mask = torch.tensor(new_attn_mask, dtype=attn_mask.dtype)

        return expanded_input_ids, expanded_target, expanded_attn_mask

    def _process(self, image, text):
        """Process the data to get the model's input for Kimi K2.5.

        This method also expands <|media_content|> tokens to match the actual
        image feature length, eliminating the need for _merge_input_ids_with_image_features
        during model forward.

        Args:
            image: PIL Image or None
            text: Formatted text string with media placeholders

        Returns:
            input_ids: Token IDs with expanded image tokens
            target: Labels (with vision tokens masked)
            pixel: List of pixel values tensors
            image_grid_thw: Grid dimensions tensor
            attn_mask: Attention mask (True = masked)
        """
        medias = [{"type": "image", "image": image}] if image is not None else []
        inputs = self.processor(
            text=text,
            medias=medias,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0]
        attn_mask = inputs["attention_mask"][0].logical_not()

        image_grid_thw = None
        pixel = []
        if image is not None:
            # Kimi processor returns 'grid_thws' not 'image_grid_thw'
            image_grid_thw = inputs["grid_thws"]
            pixel = [inputs["pixel_values"]]

        # Build target labels - mask vision-related tokens
        target = input_ids.clone()
        media_begin_id, media_end_id, media_content_id, media_pad_id = (
            self._get_vision_token_ids()
        )

        # Mask all vision tokens from loss computation
        target[target == media_begin_id] = IGNORE_INDEX
        target[target == media_end_id] = IGNORE_INDEX
        target[target == media_content_id] = IGNORE_INDEX
        target[target == media_pad_id] = IGNORE_INDEX

        # Expand <|media_content|> tokens to match actual image feature length
        if image is not None and image_grid_thw is not None:
            input_ids, target, attn_mask = self._expand_media_content_tokens(
                input_ids, target, attn_mask, image_grid_thw
            )

        return input_ids, target, pixel, image_grid_thw, attn_mask

    def _build_kimi_chat_text(self, context, answer, has_image=True):
        """Build Kimi K2.5 chat format text.

        Format:
        <|im_user|>user<|im_middle|>{context}<|im_end|><|im_assistant|>assistant\
            <|im_middle|><think></think>{answer}<|im_end|>
        """
        # Insert image placeholder in context if needed
        if has_image and "<image>" in context:
            context = context.replace("<image>", IMAGE_TOKEN_WITH_TAGS)
        elif has_image and IMAGE_TOKEN_WITH_TAGS not in context:
            # Prepend image placeholder if not present
            context = IMAGE_TOKEN_WITH_TAGS + context

        text = (
            f"{IM_USER}user{IM_MIDDLE}{context}{IM_END}"
            f"{IM_ASSISTANT}assistant{IM_MIDDLE}{THINK_START}{THINK_END}{answer}{IM_END}"
        )
        return text

    def _mask_user_turns_in_target(self, input_ids, target, answer):
        """Mask user turns and special tokens in target, only keep assistant answer for loss.

        For SFT, we only want to compute loss on the assistant's answer portion.
        """
        # Get special token IDs
        im_assistant_id = self.tokenizer.convert_tokens_to_ids(IM_ASSISTANT)
        im_middle_id = self.tokenizer.convert_tokens_to_ids(IM_MIDDLE)
        im_end_id = self.tokenizer.convert_tokens_to_ids(IM_END)
        think_start_id = self.tokenizer.convert_tokens_to_ids(THINK_START)
        think_end_id = self.tokenizer.convert_tokens_to_ids(THINK_END)

        # Find the position after <think></think> in assistant turn
        # Pattern: <|im_assistant|>assistant<|im_middle|><think></think>{answer}<|im_end|>
        input_ids_list = input_ids.tolist()

        # Find last occurrence of think_end_id (end of <think></think>)
        answer_start_pos = None
        for i in range(len(input_ids_list) - 1, -1, -1):
            if input_ids_list[i] == think_end_id:
                answer_start_pos = i + 1
                break

        if answer_start_pos is None:
            # Fallback: find position after last im_middle_id
            for i in range(len(input_ids_list) - 1, -1, -1):
                if input_ids_list[i] == im_middle_id:
                    answer_start_pos = i + 1
                    break

        # Mask everything before answer
        if answer_start_pos is not None:
            target[:answer_start_pos] = IGNORE_INDEX

        # Also mask the final <|im_end|> token
        if input_ids_list[-1] == im_end_id:
            target[-1] = IGNORE_INDEX

        return target

    def process_sft_vqa(self, context, answer, image):
        """Process the data for SFT VQA with Kimi K2.5 format.

        Args:
            context: User question/context
            answer: Assistant answer
            image: PIL Image

        Returns:
            input_ids, target, attn_mask, imgs, image_grid_thw
        """
        text = self._build_kimi_chat_text(
            context, answer, has_image=(image is not None)
        )
        input_ids, target, imgs, image_grid_thw, attn_mask = self._process(image, text)

        # Mask user turns, only compute loss on assistant answer
        target = self._mask_user_turns_in_target(input_ids, target, answer)

        return input_ids, target, attn_mask, imgs, image_grid_thw

    def encode_captioning(self, sample: CaptioningSample) -> BaseTaskSample:
        """Encode CaptioningSample for Kimi K2.5."""
        assert (
            self.args.training_phase == constants.TrainingPhase.PRETRAIN
        ), "Only support PRETRAIN phase"

        # Format: <|media_begin|>image<|media_content|><|media_pad|><|media_end|>{caption}<eos>
        text = (
            IMAGE_TOKEN_WITH_TAGS + sample.caption + self.tokenizer.tokenizer.eos_token
        )

        input_ids, target, imgs, image_grid_thw, attn_mask = self._process(
            sample.image, text
        )
        num_tiles = [len(image_grid_thw)] if image_grid_thw is not None else [0]

        if self.args.enable_discard_sample:
            assert (
                len(input_ids) <= self.args.seq_length
            ), f"{sample.__key__} input length {len(input_ids)}"
        elif image_grid_thw is not None:
            assert (
                image_grid_thw.prod() / 4 <= self.args.seq_length
            ), f"{sample.__key__} thw {image_grid_thw}"

        if _ENERGON_NEEDS_SUBFLAVOR:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavor__=None,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
        else:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )

    def encode_vqa(self, sample: VQASample) -> BaseTaskSample:
        """Encode VQA sample for Kimi K2.5."""
        if self.args.training_phase == constants.TrainingPhase.PRETRAIN:
            if self.args.add_question_in_pretrain:
                # Replace <image> placeholder with Kimi format
                text = (sample.context + sample.answers).replace(
                    "<image>", IMAGE_TOKEN_WITH_TAGS
                )
            else:
                text = IMAGE_TOKEN_WITH_TAGS + sample.answers
            text = text + self.tokenizer.tokenizer.eos_token
            input_ids, target, imgs, image_grid_thw, attn_mask = self._process(
                sample.image, text
            )
        elif self.args.training_phase == constants.TrainingPhase.SFT:
            input_ids, target, attn_mask, imgs, image_grid_thw = self.process_sft_vqa(
                sample.context, sample.answers, sample.image
            )
        else:
            raise NotImplementedError(
                f"Unknown training phase {self.args.training_phase}"
            )

        num_tiles = [len(image_grid_thw)] if image_grid_thw is not None else [0]

        if self.args.enable_discard_sample:
            assert (
                len(input_ids) <= self.args.seq_length
            ), f"{sample.__key__} input length {len(input_ids)}"
        elif image_grid_thw is not None:
            assert (
                image_grid_thw.prod() / 4 <= self.args.seq_length
            ), f"{sample.__key__} grid_thw: {image_grid_thw}"

        if _ENERGON_NEEDS_SUBFLAVOR:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavor__=None,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
        else:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )

    def process_sft_qa(
        self, messages: list, system: str, raw_video: list, raw_image: list
    ):
        """Process multi-turn conversation data for SFT with Kimi K2.5 format.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system: System prompt
            raw_video: List of video data
            raw_image: List of images

        Returns:
            input_ids, target, attn_mask, imgs, image_grid_thw, video, video_grid_thw
        """
        video_grid_thw = None
        pixel_values_videos = []
        image_grid_thw = None
        pixel_values_images = []
        video = []
        image = []

        if raw_image is not None:
            for i in raw_image:
                image.append(self._resize_image(i))

        if raw_video is not None:
            for v in raw_video:
                video.append(self._resize_video(v))

        # Process messages with Kimi plugin
        messages, mm_inputs = self.chat_template.mm_plugin.process_messages(
            messages,
            image if image is not None else [],
            video if raw_video is not None else [],
            self.processor,
        )

        if raw_video is not None and "video_grid_thw" in mm_inputs:
            video_grid_thw = mm_inputs["video_grid_thw"]
            pixel_values_videos = [mm_inputs.get("pixel_values_videos", mm_inputs.get("pixel_values"))]
        if raw_image is not None and "image_grid_thw" in mm_inputs:
            image_grid_thw = mm_inputs["image_grid_thw"]
            pixel_values_images = [mm_inputs["pixel_values"]]

        # Encode multi-turn conversation
        encode_pairs = self.chat_template.encode_multiturn(
            tokenizer=self.tokenizer,
            messages=messages,
            system=system,
        )

        input_ids, target = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(encode_pairs):
            input_ids += source_ids + target_ids
            target += [IGNORE_INDEX] * len(source_ids) + target_ids

        input_ids = torch.tensor(input_ids)
        target = torch.tensor(target)
        attn_mask = torch.zeros_like(input_ids).bool()

        # Expand <|media_content|> tokens to match actual image/video feature length.
        # Images and videos share the same <|media_content|> token ID, so all grid_thws
        # must be passed together in message order (images first, then videos) to match
        # the placeholder appearance order in input_ids. This mirrors the HF design in
        # modeling_kimi_k25.py where a single unified grid_thws covers all media.
        combined_grid_thws = None
        if image_grid_thw is not None and video_grid_thw is not None:
            combined_grid_thws = torch.cat([image_grid_thw, video_grid_thw], dim=0)
        elif image_grid_thw is not None:
            combined_grid_thws = image_grid_thw
        elif video_grid_thw is not None:
            combined_grid_thws = video_grid_thw

        if combined_grid_thws is not None:
            input_ids, target, attn_mask = self._expand_media_content_tokens(
                input_ids, target, attn_mask, combined_grid_thws
            )

        return (
            input_ids,
            target,
            attn_mask,
            pixel_values_images,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
        )

    def encode_multi_mix_qa(self, sample) -> BaseTaskSample:
        """Encode MultiMixQASample for Kimi K2.5."""
        if self.args.training_phase == constants.TrainingPhase.SFT:
            num_tiles = []

            (
                input_ids,
                target,
                attn_mask,
                imgs,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
            ) = self.process_sft_qa(
                sample.messages, sample.system, sample.video, sample.image
            )
            if sample.video is not None:
                num_tiles = [len(video_grid_thw)] if video_grid_thw is not None else []
            elif sample.image is not None:
                num_tiles = [len(image_grid_thw)] if image_grid_thw is not None else []
        else:
            raise NotImplementedError(
                f"Unknown training phase {self.args.training_phase}"
            )

        if self.args.enable_discard_sample:
            assert (
                len(input_ids) <= self.args.seq_length
            ), f"{sample.__key__} input length {len(input_ids)}"
        elif sample.video is not None and video_grid_thw is not None:
            assert (
                video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length
            ), f"{sample.__key__} grid_thw: {video_grid_thw}"
        elif sample.image is not None and image_grid_thw is not None:
            assert (
                image_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length
            ), f"{sample.__key__} grid_thw: {image_grid_thw}"

        if _ENERGON_NEEDS_SUBFLAVOR:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavor__=None,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
        else:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                num_tiles=num_tiles,
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )

    def encode_multi_vid_qa(self, sample) -> BaseTaskSample:
        """Encode video QA sample for Kimi K2.5."""
        if self.args.training_phase == constants.TrainingPhase.SFT:
            (
                input_ids,
                target,
                attn_mask,
                imgs,
                image_grid_thw,
                video,
                video_grid_thw,
            ) = self.process_sft_qa(sample.messages, sample.system, sample.video, None)
        else:
            raise NotImplementedError(
                f"Unknown training phase {self.args.training_phase}"
            )

        if self.args.enable_discard_sample:
            assert (
                len(input_ids) <= self.args.seq_length
            ), f"{sample.__key__} input length {len(input_ids)}"
        elif video_grid_thw is not None:
            assert (
                video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length
            ), f"{sample.__key__} grid_thw: {video_grid_thw}"

        if _ENERGON_NEEDS_SUBFLAVOR:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavor__=None,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=video,
                video_grid_thw=video_grid_thw,
                num_tiles=[len(video_grid_thw)] if video_grid_thw is not None else [],
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
        else:
            return VLMTaskSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,
                __subflavors__=sample.__subflavors__,
                imgs=imgs,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=video,
                video_grid_thw=video_grid_thw,
                num_tiles=[len(video_grid_thw)] if video_grid_thw is not None else [],
                tokens=input_ids,
                labels=target,
                attn_mask=attn_mask,
                total_len=len(input_ids),
            )
