# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Kimi multimodal task encoder."""

import logging
import torch
from loongforge.data.vlm.vlm_task_encoder import VLMTaskEncoder
from typing import Optional, Tuple

from loongforge import constants
from loongforge.chat_templates.base import DataRoles
from loongforge.chat_templates.hf import HFChatTemplate
from loongforge.chat_templates.plugins.mm_plugin import Placeholder

# Kimi multimodal special tokens
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


class KimiTaskEncoder(VLMTaskEncoder):
    """VLM task encoder for Kimi K2.x and K3 models.

    Kimi models use a different tokenization format:
    - Image: <|media_begin|>image<|media_content|><|media_pad|><|media_end|>
    - Video chunk: timestamp<|media_begin|>video<|media_content|><|media_pad|><|media_end|>
    - Chat template: <|im_user|>user<|im_middle|>...<|im_end|><|im_assistant|>assistant\
        <|im_middle|><think></think>...<|im_end|>

    This encoder also expands the single <|media_content|> placeholder token to multiple
    tokens based on the actual image feature length (computed from grid_thws), which is
    the functionality of the Kimi processor's media-token merge step.
    """

    image_token_with_tags = IMAGE_TOKEN_WITH_TAGS
    _grid_key = "grid_thws"  # The Kimi processor names the image grid 'grid_thws'.

    def __init__(self, args, tokenizer, chat_template=None):
        super().__init__(args, tokenizer, chat_template)

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

    def _gate_overlong(
        self,
        sample,
        input_ids,
        *,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> bool:
        """Drop-or-keep gate covering input length and visual-token budget.

        Returns True when the caller should ``return None``.  Two independent
        checks (previously inlined across every encode_xx method):

        - input-length gate: when ``enable_discard_sample`` is on, drop the
          sample if ``len(input_ids)`` exceeds the relevant limit
          (``min(seq_length, max_packed_tokens)`` when packing is enabled,
          else ``seq_length``).
        - visual-tokens trim-safety gate when ``enable_discard_sample`` is
          off.  In that mode the batcher tail-trims silently, so any cut
          point landing inside a media block desyncs ``pixel_values`` from
          the expanded ``<|media_pad|>`` placeholders.  Requiring
          ``visual_tokens <= seq_length`` guarantees at least one trim
          point can preserve every media block intact.

        All overlong cases log a warning and signal a drop instead of
        raising, so upstream pipelines do not need ``except AssertionError``
        to recover.
        """
        if self.args.enable_discard_sample:
            sequence_limit = self.args.seq_length
            packed_limit = getattr(self.args, "max_packed_tokens", None)
            if self.is_packing_enabled and packed_limit is not None:
                sequence_limit = min(sequence_limit, packed_limit)
            if len(input_ids) > sequence_limit:
                logging.warning(
                    "discard overlong sample %s: input length %s > sequence limit %s",
                    sample.__key__,
                    len(input_ids),
                    sequence_limit,
                )
                return True
        else:
            visual_tokens = 0
            if video_grid_thw is not None:
                for thw in video_grid_thw:
                    visual_tokens += self._compute_image_tokens_from_grid_thw(thw)
            if image_grid_thw is not None:
                for thw in image_grid_thw:
                    visual_tokens += self._compute_image_tokens_from_grid_thw(thw)
            if visual_tokens > self.args.seq_length:
                logging.warning(
                    "discard sample %s: visual tokens %s > seq_length %s "
                    "(video_grid_thw=%s, image_grid_thw=%s)",
                    sample.__key__,
                    visual_tokens,
                    self.args.seq_length,
                    video_grid_thw,
                    image_grid_thw,
                )
                return True
        return False

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

    def _expand_media(
        self,
        input_ids: torch.Tensor,
        target: torch.Tensor,
        attn_mask: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match <|media_content|> token count to the image feature count.

        This implements the core logic of _merge_input_ids_with_image_features from
        the Kimi processor's media-token merge step, but operates on token IDs
        instead of embeddings.

        Args:
            input_ids: Token IDs with single placeholders or plugin-expanded tokens
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
        # HF multimodal plugins may already emit one token per visual feature.
        # In that case expanding again would turn N media tokens into 2N-1.
        if input_ids_list.count(media_content_id) == sum(feature_lengths):
            return input_ids, target, attn_mask

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
                    new_target.extend([constants.IGNORE_INDEX] * num_tokens)
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

    def _call_processor(self, image, text, add_special_tokens):
        """Run the Kimi processor. It takes ``medias`` and ignores add_special_tokens."""
        medias = [{"type": "image", "image": image}] if image is not None else []
        return self.processor(text=text, medias=medias, return_tensors="pt")

    def _mask_token_ids(self):
        """Media tokens are masked in the labels."""
        return list(self._get_vision_token_ids())

    def _read_mm_inputs(self, mm_inputs, raw_image, raw_video):
        """Read grids and pixels from the plugin output. Missing keys are tolerated."""
        image_grid_thw = video_grid_thw = None
        pixel_values_images, pixel_values_videos = [], []
        if raw_video is not None and "video_grid_thw" in mm_inputs:
            video_grid_thw = mm_inputs["video_grid_thw"]
            pixel_values_videos = [mm_inputs.get("pixel_values_videos", mm_inputs.get("pixel_values"))]
        if raw_image is not None and "image_grid_thw" in mm_inputs:
            image_grid_thw = mm_inputs["image_grid_thw"]
            pixel_values_images = [mm_inputs["pixel_values"]]
        return pixel_values_images, image_grid_thw, pixel_values_videos, video_grid_thw

    def _finalize_sft_tokens(self, input_ids, target, attn_mask, image_grid_thw, video_grid_thw):
        """Expand <|media_content|> tokens to match the actual image/video feature length.

        Images and videos share the same <|media_content|> token ID, so all grid_thws
        must be passed together in message order (images first, then videos) to match
        the placeholder appearance order in input_ids.
        """
        grids = [g for g in (image_grid_thw, video_grid_thw) if g is not None]
        if not grids:
            return input_ids, target, attn_mask
        grid = torch.cat(grids, dim=0) if len(grids) > 1 else grids[0]
        return self._expand_media(input_ids, target, attn_mask, grid)

    def _num_tiles(self, image_grid_thw, video_grid_thw):
        """Tile counts of video then image, both kept."""
        return [len(g) for g in (video_grid_thw, image_grid_thw) if g is not None]

    def _build_kimi_chat_text(self, context, answer, has_image=True):
        """Build Kimi chat-format text.

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

    def _mask_user_turns_in_target(self, input_ids, target):
        """Mask user turns and special tokens in target, only keep assistant answer for loss.

        For SFT, we only want to compute loss on the assistant's answer portion.
        """
        im_middle_id = self.tokenizer.convert_tokens_to_ids(IM_MIDDLE)
        im_end_id = self.tokenizer.convert_tokens_to_ids(IM_END)
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
            target[:answer_start_pos] = constants.IGNORE_INDEX

        # Also mask the final <|im_end|> token
        if input_ids_list[-1] == im_end_id:
            target[-1] = constants.IGNORE_INDEX

        return target

    def process_sft_vqa(self, context, answer, image):
        """Process VQA data for SFT with Kimi chat formatting.

        Args:
            context: User question/context
            answer: Assistant answer
            image: PIL Image

        Returns:
            input_ids, target, attn_mask, imgs, image_grid_thw
        """
        if isinstance(self.chat_template, HFChatTemplate):
            if image is not None and Placeholder.IMAGE not in context:
                context = Placeholder.IMAGE + context
            messages = [
                {"role": DataRoles.USER, "content": context},
                {"role": DataRoles.ASSISTANT, "content": answer},
            ]
            (
                input_ids,
                target,
                attn_mask,
                imgs,
                image_grid_thw,
                _,
                _,
            ) = self.process_sft_qa(
                messages,
                "",
                None,
                [image] if image is not None else None,
            )
            return input_ids, target, attn_mask, imgs, image_grid_thw

        text = self._build_kimi_chat_text(
            context, answer, has_image=(image is not None)
        )
        input_ids, target, imgs, image_grid_thw, attn_mask = self._process(
            image, text
        )

        target = self._mask_user_turns_in_target(input_ids, target)

        return input_ids, target, attn_mask, imgs, image_grid_thw

    def process_sft_qa(self, messages, system, raw_video, raw_image, tools=None):
        """Check the template has an mm_plugin, then run the shared SFT path."""
        if self.chat_template.mm_plugin is None:
            raise ValueError(
                "KimiTaskEncoder requires a Kimi multimodal chat template. "
                "Use --chat-template kimi-k2.5-hf or kimi-k3-hf."
            )
        return super().process_sft_qa(messages, system, raw_video, raw_image, tools)
