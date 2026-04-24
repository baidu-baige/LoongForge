# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ErnieTaskEncoder — Energon TaskEncoder for ERNIE-VL models.

Bridges the ERNIE-VL offline preprocessing pipeline into the Megatron Energon
online data pipeline, so training can run directly from WebDataset shards
without pre-generating .npz files.

Usage:
    --task-encoder ErnieTaskEncoder --data-path <webdataset_dir>
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from megatron.energon import Cooker
from megatron.energon.task_encoder.base import stateless
from importlib.metadata import version as _energon_version

try:
    _ENERGON_NEEDS_SUBFLAVOR = _energon_version("megatron-energon") < "7.0.0"
except Exception:
    _ENERGON_NEEDS_SUBFLAVOR = False

from loongforge.data.multimodal import MultiMixQASample
from loongforge.data.multimodal.base.task_encoder import (
    BaseTaskEncoder,
    BaseTaskSample,
    BaseTaskSamplePacked,
    BaseTaskBatchPacked,
    _parse_messages,
    IGNORE_INDEX,
)

logger = logging.getLogger(__name__)

# ERNIE components — now directly importable as submodules of the ernie package.
from loongforge.data.multimodal.ernie.tokenizer_vl import (
    Ernie45VLTokenizer,
    special_tokens_info as ERNIE_SPECIAL_TOKENS_INFO,
)
from loongforge.data.multimodal.ernie.chat_template_utils import (
    apply_chat_training_template,
)
from loongforge.data.multimodal.ernie.example_to_feature import (
    ExampleToFeature,
)
from loongforge.data.multimodal.ernie.image_preprocessor import (
    AdaptiveImageProcessor,
)
from loongforge.data.multimodal.ernie.image_modification import (
    ImageModificationProcessor,
)

# ===========================================================================
# Extended dataclasses
# ===========================================================================


@dataclass
class ErnieTaskSample(BaseTaskSample):
    """Single encoded sample with ERNIE-specific fields."""

    token_type_ids: torch.Tensor = None  # (seq_len,) 0=text, 1=image, 2=video
    image_type_ids: torch.Tensor = None  # (num_images,) 0=image, 1=video
    image_grid_thw: torch.Tensor = None  # (num_images, 3)
    position_ids_3d: torch.Tensor = None  # (seq_len, 3)


@dataclass
class ErnieTaskSamplePacked(BaseTaskSamplePacked):
    """Packed sample with ERNIE-specific fields."""

    token_type_ids: torch.Tensor = None
    image_type_ids: torch.Tensor = None
    image_grid_thw: torch.Tensor = None
    position_ids_3d: torch.Tensor = None


@dataclass
class ErnieTaskBatchPacked(BaseTaskBatchPacked):
    """Batch of packed samples with ERNIE-specific fields."""

    token_type_ids: torch.Tensor = None  # (N, seq_len)
    image_type_ids: torch.Tensor = None  # (total_images,)
    image_grid_thw: torch.Tensor = None  # (total_images, 3)
    position_ids_3d: torch.Tensor = None  # (N, seq_len, 3)


# ===========================================================================
# Cooker
# ===========================================================================


@stateless
def cooker_ernie_mix_qa(sample: dict) -> MultiMixQASample:
    """Parse a WebDataset sample into a MultiMixQASample for ERNIE-VL."""
    messages, system_prompt = _parse_messages(sample["json"]["texts"])

    video: list = []
    image: list = []
    media = sample["json"].get("media", "")
    if media == "video":
        for name in sample["json"]["name"]:
            video.append(sample.get(name))
    elif media == "image":
        for name in sample["json"]["name"]:
            image.append(sample.get(name))

    kwargs = dict(
        __key__=sample["__key__"],
        __restore_key__=sample["__restore_key__"],
        __subflavors__=sample.get("__subflavors__", {}),
        video=video if len(video) > 0 else None,
        image=image if len(image) > 0 else None,
        system=system_prompt,
        messages=messages,
    )
    if _ENERGON_NEEDS_SUBFLAVOR:
        kwargs["__subflavor__"] = None
    return MultiMixQASample(**kwargs)


# ===========================================================================
# ErnieTaskEncoder
# ===========================================================================


class ErnieTaskEncoder(BaseTaskEncoder):
    """Energon TaskEncoder for ERNIE-VL models.

    Reuses the existing ERNIE preprocessing components (tokenizer, chat
    template, ExampleToFeature, AdaptiveImageProcessor, 3D RoPE) but wraps
    them in the Energon encode_sample / batch / encode_batch interface.
    """

    cookers = [
        Cooker(cooker_ernie_mix_qa, has_subflavors={"sample_type": "ernie_mix_qa"}),
        # Fall through to base cookers for other sample types.
    ] + BaseTaskEncoder.cookers

    def __init__(self, args):
        super().__init__()

        # ---- Tokenizer ----
        self.ernie_tokenizer = Ernie45VLTokenizer.from_pretrained(
            args.hf_tokenizer_path, trust_remote_code=True,
        )
        # Wrap tokenizer.encode for paddleformers-style API compatibility.
        self._wrap_tokenizer_encode()

        # ---- Image processor ----
        min_pixels = getattr(args, "min_pixels", None) or (56 * 56)
        max_pixels = getattr(args, "max_pixels", None) or (28 * 28 * 1280)
        self.image_processor = AdaptiveImageProcessor(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # ---- ExampleToFeature ----
        self.chat_template = getattr(args, "ernie_chat_template", "ernie_vl")
        self.example_to_feature = ExampleToFeature(
            tokenizer=self.ernie_tokenizer,
            max_seq_length=args.seq_length,
            special_tokens_info=ERNIE_SPECIAL_TOKENS_INFO,
            variable_resolution=True,
            spatial_conv_size=getattr(args, "spatial_conv_size", 2),
            image_processor=self.image_processor,
            rope_3d=True,
            video_min_pixels=getattr(args, "video_min_pixels", None),
            video_max_pixels=getattr(args, "video_max_pixels", None),
            chat_template=self.chat_template,
        )

        # ---- ImageModificationProcessor (for position_ids_for_rope_3d) ----
        # We create a lightweight args-like object for ImageModificationProcessor.
        self._imp = _build_image_modification_processor(
            args, self.ernie_tokenizer, self.image_processor,
        )

        # ---- Video parameters ----
        self.video_fps = getattr(args, "video_fps", 2)
        self.video_min_frames = getattr(args, "video_min_frames", 4)
        self.video_max_frames = getattr(args, "video_max_frames", 768)
        self.video_target_frames = getattr(args, "video_target_frames", -1)
        self.video_frames_sample = getattr(args, "video_frames_sample", "middle")

        # Image patch ID for detecting vision tokens.
        vocab = self.ernie_tokenizer.get_vocab()
        self.im_patch_id = vocab[ERNIE_SPECIAL_TOKENS_INFO["image_placeholder"]]

    # -----------------------------------------------------------------
    # Tokenizer wrapping
    # -----------------------------------------------------------------
    def _wrap_tokenizer_encode(self):
        """Ensure ernie_tokenizer.encode returns a dict (paddleformers compat)."""
        orig_encode = self.ernie_tokenizer.encode

        def _wrapped_encode(text, **kwargs):
            result = orig_encode(text, **kwargs)
            if isinstance(result, dict):
                return result
            # transformers returns a list of ints for encode()
            return {"input_ids": result}

        self.ernie_tokenizer.encode = _wrapped_encode

    # -----------------------------------------------------------------
    # Message conversion helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _messages_to_all_item_list(
        messages: List[dict],
        images: Optional[List[Image.Image]],
        system: Optional[str],
        add_think_prefix: bool = True,
    ) -> dict:
        """Convert Energon-style messages + images into the ERNIE
        ``all_item_list`` format expected by ``apply_chat_training_template``.

        Returns a dict with keys ``all_item_list``, ``is_system``.

        When *add_think_prefix* is True (default), a ``<think>\\n\\n</think>\\n\\n``
        prefix is inserted before the last assistant response with ``tag="mask"``
        (labels=-100), matching the offline pipeline's ``reformat_meta()`` /
        ``processor.py`` behaviour.
        """
        THINK_PREFIX = "<think>\n\n</think>\n\n"

        all_item_list: list = []
        is_system = False
        image_idx = 0

        if system:
            # System prompt is the first "turn" (even index = user side).
            all_item_list.append([{"text": system, "tag": "mask"}])
            # Dummy assistant turn for system (will be skipped by template).
            all_item_list.append([{"text": "", "tag": "mask"}])
            is_system = True

        for msg in messages:
            turn: list = []
            content = msg["content"]

            # Check for image placeholders in content.
            if images and "<image>" in content:
                parts = content.split("<image>")
                for pi, part in enumerate(parts):
                    if part:
                        tag = "mask" if msg["role"] == "user" else "no_mask"
                        turn.append({"text": part, "tag": tag})
                    if pi < len(parts) - 1 and image_idx < len(images):
                        img = images[image_idx]
                        w, h = img.size
                        turn.append([{
                            "image_url": img,
                            "image_width": w,
                            "image_height": h,
                            "is_valid": True,
                            "image_type": "image",
                        }])
                        image_idx += 1
            else:
                tag = "mask" if msg["role"] == "user" else "no_mask"
                turn.append({"text": content, "tag": tag})

            all_item_list.append(turn)

        # If images not yet consumed (no <image> placeholders), prepend to first user turn.
        if images and image_idx == 0:
            img_items = []
            for img in images:
                w, h = img.size
                img_items.append({
                    "image_url": img,
                    "image_width": w,
                    "image_height": h,
                    "is_valid": True,
                    "image_type": "image",
                })
            # Find the first user turn.
            first_user_idx = 1 if is_system else 0
            if first_user_idx < len(all_item_list):
                all_item_list[first_user_idx].insert(0, img_items)

        # ---- Think prefix injection (matches offline pipeline) ----
        # The offline pipeline (processor.py) inserts the think prefix as a
        # separate ``{"text": ..., "tag": "mask"}`` entry at position 0 of the
        # last assistant turn in all_item_list.  It also sets ``label=0`` on
        # all earlier turns so that only the last round participates in loss.
        if add_think_prefix:
            # Find the last assistant turn (odd-indexed in all_item_list,
            # accounting for is_system offset).
            last_asst_idx = None
            for idx in range(len(all_item_list) - 1, -1, -1):
                # Assistant turns are always at odd indices (0-based):
                # [user, asst, user, asst, ...] or [sys, empty, user, asst, ...]
                if idx % 2 == 1:
                    last_asst_idx = idx
                    break

            if last_asst_idx is not None:
                # Check if think tags already present in the turn content.
                has_think = any(
                    isinstance(item, dict) and "<think>" in item.get("text", "")
                    for item in all_item_list[last_asst_idx]
                )
                if not has_think:
                    # Insert think prefix with tag="mask" at position 0.
                    all_item_list[last_asst_idx].insert(
                        0, {"text": THINK_PREFIX, "tag": "mask"}
                    )
                    # Mark all earlier turns with label=0 (masked in loss) and
                    # last turn with label=1, matching offline processor.py
                    # lines 163-171.
                    for tidx in range(len(all_item_list)):
                        for inner_idx, item in enumerate(all_item_list[tidx]):
                            if isinstance(item, dict) and "text" in item:
                                if tidx == last_asst_idx:
                                    all_item_list[tidx][inner_idx]["label"] = 1
                                else:
                                    all_item_list[tidx][inner_idx]["label"] = 0

        return {"all_item_list": all_item_list, "is_system": is_system}

    # -----------------------------------------------------------------
    # Core encode method
    # -----------------------------------------------------------------
    def encode_multi_mix_qa(self, sample: MultiMixQASample) -> ErnieTaskSample:
        """Encode a MultiMixQASample into an ErnieTaskSample.

        Steps:
          1. Convert messages + images to ERNIE internal format.
          2. Apply chat training template.
          3. Run ExampleToFeature for tokenization + placeholder insertion.
          4. Process images with AdaptiveImageProcessor.
          5. Compute 3D RoPE position IDs.
          6. Assemble ErnieTaskSample.
        """
        images = sample.image  # List[PIL.Image] or None
        # TODO: video support via CoarseProcessor frame extraction
        _videos = sample.video  # noqa: F841  List[AVData] or None

        # ---- Step 1: Build all_item_list ----
        data = self._messages_to_all_item_list(
            sample.messages, images, sample.system,
        )

        # ---- Step 2: Apply chat training template ----
        template_result = apply_chat_training_template(
            data=data,
            tokenizer=self.ernie_tokenizer,
            is_training=True,
            chat_template=self.chat_template,
            use_pic_id=True,
        )
        # template_result = {"text_info": [...], "image_info": [...]}

        # ---- Step 3: ExampleToFeature (tokenization) ----
        feature_result = None
        for feat in self.example_to_feature.example_to_feature(template_result):
            feature_result = feat
            break  # Take the first (and typically only) feature.

        if feature_result is None:
            raise ValueError(
                f"ExampleToFeature yielded no features for sample {sample.__key__}"
            )

        ids = feature_result["feature"]["ids"]
        lossmask = feature_result["feature"]["lossmask"]
        ids_type = feature_result["feature"]["ids_type"]
        image_wise_type = feature_result["feature"]["image_wise_type"]
        image_meta = feature_result["meta"]

        # Build input_ids, labels, token_type_ids from the feature.
        input_ids = np.array(ids, dtype=np.int64)
        labels = np.array(
            [IGNORE_INDEX if m == 0 else t for t, m in zip(ids, lossmask)],
            dtype=np.int64,
        )
        token_type_ids = np.array(ids_type, dtype=np.int64)
        image_type_ids_np = np.array(image_wise_type, dtype=np.int64)

        # ---- Align with offline torch pipeline (packing_dataloader) ----
        # 1) Replace SEP token with EOS token in labels (matches
        #    ImageModificationProcessor.mm_example_to_feature for ernie_vl).
        vocab = self.ernie_tokenizer.get_vocab()
        sep_token = self.ernie_tokenizer.special_tokens_map.get(
            "sep_token", "<|endofprompt|>"
        )
        eos_token = self.ernie_tokenizer.special_tokens_map.get(
            "eos_token", "</s>"
        )
        sep_token_id = vocab[sep_token]
        eos_token_id = vocab[eos_token]
        labels[labels == sep_token_id] = eos_token_id

        # 2) Shift-by-one for autoregressive training alignment.
        #    When packing is enabled, skip per-sample shift here — it will be
        #    done globally in pack_selected_samples() after concatenation, to
        #    match the ERNIE reference _concat_samples logic:
        #      concat raw [BOS,...,SEP] → remove last token → labels[1:]
        if not getattr(self.args, "packing_sft_data", False):
            input_ids = input_ids[:-1]
            labels = labels[1:]
            token_type_ids = token_type_ids[:-1]

        # ---- Step 4: Process images via ImageModificationProcessor ----
        # Use the same image_handling_for_adaptive() as the offline pipeline
        # to correctly handle local/global tiling (args_fn positions).
        pixel_values = None
        grid_thw = None
        if images and len(image_meta) > 0:
            from loongforge.data.multimodal.ernie.image_modification import VisionExample

            # image_handling_for_adaptive expects VisionExample with meta=[image_meta]
            example = VisionExample(
                meta=[image_meta],
                ids=ids,
                sids=None,
                task="mm",
                lossmask=lossmask,
                src=-1,
                part=-1,
                info=-1,
                name="dummy",
                data_type=0,
                token_type_ids=ids_type,
                image_type_ids=image_wise_type,
            )

            def _download_pil(url_or_img, need_exif_info=False):
                """Download function that handles PIL images directly."""
                if isinstance(url_or_img, Image.Image):
                    img = url_or_img.convert("RGB")
                else:
                    img = Image.open(url_or_img).convert("RGB")
                return (img, None) if need_exif_info else (img,)

            pixel_values, grid_thw = self._imp.image_handling_for_adaptive(
                example, download_fn=_download_pil,
            )

        # Handle text-only samples.
        if pixel_values is None:
            pixel_values = np.zeros([0, 3 * 14 * 14], dtype=np.float32)
            grid_thw = np.zeros([0, 3], dtype=np.int64)
            image_type_ids_np = np.array([], dtype=np.int64)

        # ---- Step 5: Compute 3D RoPE position IDs ----
        # Use the full (untruncated) ids/token_type_ids for position computation
        # since position_ids_for_rope_3d needs to see image token boundaries,
        # then truncate the result to match the shifted input_ids.
        full_input_ids = np.array(ids, dtype=np.int64)
        full_token_type_ids = np.array(ids_type, dtype=np.int64)
        feature_dict = {
            "input_ids": full_input_ids,
            "images": pixel_values if len(pixel_values) > 0 else None,
            "grid_thw": grid_thw,
            "token_type_ids": full_token_type_ids,
            "image_type_ids": image_type_ids_np,
        }
        feature_dict = self._imp.position_ids_for_rope_3d(feature_dict)
        # When packing, keep full position_ids (truncation done in pack_selected_samples).
        if getattr(self.args, "packing_sft_data", False):
            position_ids = feature_dict["position_ids"]
        else:
            position_ids = feature_dict["position_ids"][:-1]  # truncate to match input_ids

        # ---- Step 6: Assemble ErnieTaskSample ----
        input_ids_t = torch.from_numpy(input_ids)
        labels_t = torch.from_numpy(labels)
        attn_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        token_type_ids_t = torch.from_numpy(token_type_ids)
        image_type_ids_t = torch.from_numpy(image_type_ids_np)
        grid_thw_t = torch.from_numpy(grid_thw).to(torch.int64)
        position_ids_t = torch.from_numpy(position_ids).to(torch.int64)

        # Pixel values: keep as uint8-compatible tensor list (like existing pipeline).
        if len(pixel_values) > 0:
            imgs_t = [torch.from_numpy(pixel_values)]
        else:
            imgs_t = []

        kwargs = dict(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavors__=sample.__subflavors__,
            tokens=input_ids_t,
            labels=labels_t,
            attn_mask=attn_mask,
            total_len=len(input_ids_t),
            imgs=imgs_t,
            num_tiles=[len(grid_thw_t)] if len(grid_thw_t) > 0 else [],
            token_type_ids=token_type_ids_t,
            image_type_ids=image_type_ids_t,
            image_grid_thw=grid_thw_t,
            position_ids_3d=position_ids_t,
        )
        if _ENERGON_NEEDS_SUBFLAVOR:
            kwargs["__subflavor__"] = None
        return ErnieTaskSample(**kwargs)

    # -----------------------------------------------------------------
    # Packing
    # -----------------------------------------------------------------
    @stateless
    def pack_selected_samples(
        self, samples: List[ErnieTaskSample]
    ) -> ErnieTaskSamplePacked:
        """Pack samples with ERNIE-specific concat logic.

        Matches ERNIE reference _concat_samples:
          - First sample: keep full [BOS, ..., SEP]
          - Subsequent samples: remove BOS (first token)
          - Global shift: input_ids[:-1], labels[1:], token_type_ids[:-1]
          - Position IDs: last sample truncate [-1], subsequent [1:] then -1,
            merge with max(prev) + cur (no +1)
        """
        # ---- 1. Concat tokens/labels/token_type_ids per ERNIE reference ----
        token_lists = []
        label_lists = []
        ttype_lists = []
        for i, s in enumerate(samples):
            if i == 0:
                token_lists.append(s.tokens)
                label_lists.append(s.labels)
                ttype_lists.append(s.token_type_ids)
            else:
                # Subsequent samples: remove BOS (first token)
                if len(s.tokens) > 1:
                    token_lists.append(s.tokens[1:])
                    label_lists.append(s.labels[1:])
                    ttype_lists.append(s.token_type_ids[1:])

        packed_tokens = torch.cat(token_lists, dim=0)
        packed_labels = torch.cat(label_lists, dim=0)
        packed_token_type_ids = torch.cat(ttype_lists, dim=0)

        # Global shift-by-one: input_ids[:-1], labels[1:], token_type_ids[:-1]
        packed_tokens = packed_tokens[:-1]
        packed_labels = packed_labels[1:]
        packed_token_type_ids = packed_token_type_ids[:-1]

        # ---- 2. Position IDs: ERNIE reference logic ----
        # Step a: last sample truncate [-1] (remove trailing SEP position)
        # Step b: subsequent samples [1:] (remove BOS position) then -1
        # Step c: merge with max(prev) + cur (no +1)
        pos_lists = []
        for i, s in enumerate(samples):
            pos = s.position_ids_3d.clone()  # (seq_len, 3)
            if i == len(samples) - 1:
                # Last sample: truncate last position
                pos = pos[:-1]
            if i == 0:
                pos_lists.append(pos)
            else:
                # Non-first: remove BOS position, then -1
                if len(pos) > 1:
                    pos = pos[1:]
                pos = pos - 1
                pos_lists.append(pos)

        # Merge with cumulative offset: max(prev) + cur (no +1)
        if len(pos_lists) == 1:
            packed_position_ids_3d = pos_lists[0]
        else:
            merged = pos_lists[0]
            for pos in pos_lists[1:]:
                if pos.numel() > 0:
                    pos = merged.max() + pos
                    merged = torch.cat([merged, pos], dim=0)
            packed_position_ids_3d = merged

        # ---- 3. Image fields: direct concat (not affected by BOS/SEP) ----
        packed_image_type_ids = torch.cat(
            [s.image_type_ids for s in samples if len(s.image_type_ids) > 0],
            dim=0,
        ) if any(len(s.image_type_ids) > 0 for s in samples) else torch.tensor([], dtype=torch.int64)

        packed_grid_thw = torch.cat(
            [s.image_grid_thw for s in samples if len(s.image_grid_thw) > 0],
            dim=0,
        ) if any(len(s.image_grid_thw) > 0 for s in samples) else torch.zeros(0, 3, dtype=torch.int64)

        # ---- 4. Build cu_lengths and collect images ----
        # cu_lengths marks document boundaries in the packed sequence for
        # attention masking.  After the global shift (input_ids[:-1]) the
        # final SEP is removed, so the last sample loses 1 token.
        packing_seq_len = self.args.seq_length
        packed_imgs = []
        cu_lengths = [0]
        current_length = 0
        max_length = 0

        for i, s in enumerate(samples):
            if i == 0:
                sample_len = len(s.tokens)  # full [BOS, ..., SEP]
            else:
                sample_len = len(s.tokens) - 1  # BOS removed

            # Last sample loses 1 token due to global input_ids[:-1]
            if i == len(samples) - 1:
                sample_len -= 1

            if sample_len > max_length:
                max_length = sample_len

            if s.imgs is not None:
                packed_imgs += s.imgs

            current_length += sample_len
            cu_lengths.append(current_length)

        total_packed_len = len(packed_tokens)
        assert cu_lengths[-1] == total_packed_len, (
            f"cu_lengths[-1]={cu_lengths[-1]} != packed_tokens len={total_packed_len}"
        )

        if total_packed_len > packing_seq_len:
            raise ValueError(
                f"Packed sample exceeds max seq length {packing_seq_len}: "
                f"got {total_packed_len}"
            )

        init_kwargs = dict(
            __key__=",".join([s.__key__ for s in samples]),
            __restore_key__=(),
            __subflavors__=samples[0].__subflavors__,
            tokens=packed_tokens,
            labels=packed_labels,
            attn_mask=torch.zeros(len(packed_tokens), dtype=torch.bool),
            imgs=packed_imgs,
            pixel_values_videos=None,
            cu_lengths=torch.tensor(cu_lengths, dtype=torch.int32),
            max_length=max_length,
            num_tiles=[n for s in samples for n in s.num_tiles],
            token_type_ids=packed_token_type_ids,
            image_type_ids=packed_image_type_ids,
            image_grid_thw=packed_grid_thw,
            position_ids_3d=packed_position_ids_3d,
        )
        if _ENERGON_NEEDS_SUBFLAVOR:
            init_kwargs["__subflavor__"] = None
        return ErnieTaskSamplePacked(**init_kwargs)

    # -----------------------------------------------------------------
    # Batching
    # -----------------------------------------------------------------
    def batch(
        self, samples: List[Union[ErnieTaskSample, ErnieTaskSamplePacked]]
    ) -> ErnieTaskBatchPacked:
        """Batch samples together, adding ERNIE-specific padded fields."""
        base_batch = super().batch(samples)

        max_seq_len = base_batch.tokens.shape[-1]
        batch_size = base_batch.tokens.shape[0]

        # Pad token_type_ids (pad with 0 = text type).
        token_type_ids_np = np.zeros(
            (batch_size, max_seq_len), dtype=np.int64
        )
        for i, s in enumerate(samples):
            tlen = min(max_seq_len, len(s.token_type_ids))
            token_type_ids_np[i, :tlen] = s.token_type_ids[:tlen].numpy()

        # Pad position_ids_3d (pad with incrementing values).
        position_ids_np = np.zeros(
            (batch_size, max_seq_len, 3), dtype=np.int64
        )
        for i, s in enumerate(samples):
            tlen = min(max_seq_len, len(s.position_ids_3d))
            position_ids_np[i, :tlen, :] = s.position_ids_3d[:tlen].numpy()
            if tlen < max_seq_len:
                # Fill padding positions with incrementing values.
                last_pos = s.position_ids_3d[tlen - 1].numpy() if tlen > 0 else np.zeros(3)
                for j in range(tlen, max_seq_len):
                    position_ids_np[i, j, :] = last_pos + (j - tlen + 1)

        # Concat image_type_ids and image_grid_thw across batch (no padding).
        all_image_type_ids = []
        all_grid_thw = []
        for s in samples:
            if hasattr(s, "image_type_ids") and s.image_type_ids is not None and len(s.image_type_ids) > 0:
                all_image_type_ids.append(s.image_type_ids)
            if hasattr(s, "image_grid_thw") and s.image_grid_thw is not None and len(s.image_grid_thw) > 0:
                all_grid_thw.append(s.image_grid_thw)
        image_type_ids = (
            torch.cat(all_image_type_ids, dim=0)
            if all_image_type_ids
            else torch.tensor([], dtype=torch.int64)
        )
        image_grid_thw = torch.cat(all_grid_thw, dim=0) if all_grid_thw else torch.zeros(0, 3, dtype=torch.int64)

        init_args = vars(base_batch).copy()
        init_args.update({
            "__key__": base_batch.__key__,
            "__restore_key__": base_batch.__restore_key__,
            "__subflavors__": base_batch.__subflavors__,
            "token_type_ids": token_type_ids_np,
            "image_type_ids": image_type_ids,
            "image_grid_thw": image_grid_thw,
            "position_ids_3d": position_ids_np,
        })
        return ErnieTaskBatchPacked(**init_args)

    # -----------------------------------------------------------------
    # encode_batch — produce the final dict for the model
    # -----------------------------------------------------------------
    def encode_batch(self, batch: ErnieTaskBatchPacked) -> dict:
        """Produce the final dict consumed by ``get_batch()`` in sft_ernie.py.

        Field names are aligned with the existing offline pipeline output.
        """
        tokens = batch.tokens
        if isinstance(tokens, np.ndarray):
            tokens = torch.from_numpy(tokens)

        labels = batch.labels
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        token_type_ids = batch.token_type_ids
        if isinstance(token_type_ids, np.ndarray):
            token_type_ids = torch.from_numpy(token_type_ids)

        position_ids_3d = batch.position_ids_3d
        if isinstance(position_ids_3d, np.ndarray):
            position_ids_3d = torch.from_numpy(position_ids_3d)

        imgs = batch.imgs
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)

        image_grid_thw = batch.image_grid_thw
        if isinstance(image_grid_thw, np.ndarray):
            image_grid_thw = torch.from_numpy(image_grid_thw)

        image_type_ids = batch.image_type_ids
        if isinstance(image_type_ids, np.ndarray):
            image_type_ids = torch.from_numpy(image_type_ids)

        return {
            "input_ids": tokens.to(torch.int64),
            "token_type_ids": token_type_ids.to(torch.int64),
            "position_ids": position_ids_3d.to(torch.int64),
            "images": imgs,
            "grid_thw": image_grid_thw.to(torch.int64),
            "image_type_ids": image_type_ids.to(torch.int64),
            "labels": labels.to(torch.int64),
            "cu_lengths": batch.cu_lengths,
            "max_lengths": batch.max_lengths,
        }

    # -----------------------------------------------------------------
    # Image processing override
    # -----------------------------------------------------------------
    def process_images(
        self, samples: List[Union[ErnieTaskSample, ErnieTaskSamplePacked]]
    ) -> torch.Tensor:
        """Concatenate image patches. ERNIE uses flattened patches, not (C,H,W)."""
        imgs = [img for s in samples if s.imgs is not None for img in s.imgs]
        if len(imgs) > 0:
            return torch.cat(imgs, dim=0)
        else:
            return torch.tensor([[0]], dtype=torch.float32)


# ===========================================================================
# Helper: build a lightweight ImageModificationProcessor for RoPE
# ===========================================================================


@dataclass
class _ImageArgs:
    """Minimal args-like object for ImageModificationProcessor."""
    image_token_len: int = 64
    image_dtype: str = "float32"
    sft_shift_by_one: bool = False
    sft_replace_ids: bool = False
    sft_image_rescale: bool = False
    sft_image_normalize: bool = False
    is_training: bool = True
    is_pretraining: bool = False


def _build_image_modification_processor(args, tokenizer, image_processor):
    """Create an ImageModificationProcessor for position_ids_for_rope_3d only."""
    img_args = _ImageArgs(
        image_token_len=getattr(args, "image_token_len", 64),
        image_dtype=getattr(args, "image_dtype", "float32"),
        sft_shift_by_one=getattr(args, "sft_shift_by_one", False),
        sft_replace_ids=getattr(args, "sft_replace_ids", False),
        sft_image_rescale=getattr(args, "sft_image_rescale", False),
        sft_image_normalize=getattr(args, "sft_image_normalize", False),
    )
    imp = ImageModificationProcessor(img_args, tokenizer, image_processor)
    imp.is_pretraining = False
    imp.should_shift_by_one = False
    return imp
