# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Energon cookers: convert raw WebDataset samples into flavors."""

import functools
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

from megatron.energon.task_encoder.base import stateless

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


_vlm_tags_cache: Optional[Dict[str, Dict]] = None

_IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff")
_VIDEO_EXTS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".webm")


def bind_args(cooker, args):
    """Bind *args* to a cooker, keeping the attributes Energon reads (``__stateless__`` etc.)."""
    return functools.update_wrapper(functools.partial(cooker, args=args), cooker)


def _load_vlm_tags(args, section: Optional[str] = None) -> Dict[str, any]:
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
            tool_tag: tool
            function_tag: function
    """
    global _vlm_tags_cache
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


def _parse_messages(raw_messages, args, section: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
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
    tags = _load_vlm_tags(args, section)
    role_tag = tags.get("role_tag", "role")
    content_tag = tags.get("content_tag", "content")
    role_map = {
        tags.get("user_tag", "user"): "user",
        tags.get("assistant_tag", "assistant"): "assistant",
        tags.get("system_tag", "system"): "system",
        tags.get("tool_tag", "tool"): "tool",
    }

    messages: List[Dict] = []
    system: Optional[str] = None

    for message in raw_messages:
        role = message.get(role_tag)
        content = message.get(content_tag, "")
        role = role_map.get(role, role)
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Unsupported role '{role}' in message: {message}")
        if role == "system":
            system = content
            continue
        normalized = dict(message)
        normalized["role"] = role
        normalized["content"] = content
        messages.append(normalized)

    return messages, system


def _attach_tools(sample_obj, json_data: dict):
    tools = json_data.get("tools")
    if tools:
        setattr(sample_obj, "tools", tools)
    return sample_obj


@stateless
def cooker_multi_mix_qa(sample: dict, args):
    """Convert raw sample dict into a MultiMixQASample. """
    messages, system = _parse_messages(sample["json"]["texts"], args)
    video = []
    image = []
    if sample["json"]["media"] == "video":
        for name in sample["json"]["name"]:
            video.append(sample.get(name))
    elif sample["json"]["media"] == "image":
        for name in sample["json"]["name"]:
            image.append(sample.get(name))

    if ENERGON_LT_7:
        return _attach_tools(MultiMixQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample.get("__subflavors__", {}),
            video=video if len(video) > 0 else None,
            image=image if len(image) > 0 else None,
            system=system,
            messages=messages,
        ), sample["json"])
    else:
        return _attach_tools(MultiMixQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavors__=sample.get("__subflavors__", {}),
            video=video if len(video) > 0 else None,
            image=image if len(image) > 0 else None,
            system=system,
            messages=messages,
        ), sample["json"])

@stateless
def cooker_chat_mix(sample: dict, args) -> ChatMixSample:
    """Convert raw sample dict into a ChatMixSample.

    Expected json layout (one chat session per sample):
      {
        "messages": [...],          # OpenAI Chat Completions-style messages
        "tools": [...],             # optional tool definitions
        "media": "image"|"video"|"text",
        "name": [...]               # media file names referenced by messages
      }
    """
    data = sample["json"]
    raw_messages = data.get("messages")
    if raw_messages is None:
        raise ValueError(
            f"cooker_chat_mix: sample {sample.get('__key__')!r} has no "
            "`messages` field. chat_mix shards must use OpenAI Chat "
            "Completions-style `messages`."
        )

    messages, system = _parse_messages(raw_messages, args)
    tools = data.get("tools")

    video: List = []
    image: List = []
    for name in data.get("name", []) or []:
        obj = sample.get(name)
        if obj is None:
            continue
        lower = name.lower()
        if lower.endswith(_IMAGE_EXTS):
            image.append(obj)
        elif lower.endswith(_VIDEO_EXTS):
            video.append(obj)
        else:
            raise ValueError(
                f"cooker_chat_mix: sample {sample.get('__key__')!r} has media "
                f"{name!r} with unrecognized extension; expected one of "
                f"{_IMAGE_EXTS + _VIDEO_EXTS}"
            )

    init_kwargs = {
        "__key__": sample["__key__"],
        "__restore_key__": sample["__restore_key__"],
        "__subflavors__": sample.get("__subflavors__", {}),
        "messages": messages,
        "image": image if len(image) > 0 else None,
        "video": video if len(video) > 0 else None,
        "system": system,
        "tools": tools if tools else None,
    }
    if ENERGON_LT_7:
        init_kwargs["__subflavor__"] = None
    return ChatMixSample(**init_kwargs)


@stateless
def cooker_multi_vid_vqa(sample: dict, args):
    """Convert raw sample dict into a MultiVidQASample. """
    messages, system = _parse_messages(sample["json"]["texts"], args)

    video = []
    image = []

    if sample["json"]["media"] == "video":
        for name in sample["json"]["name"]:
            video.append(sample.get(name))
    elif sample["json"]["media"] == "image":
        for name in sample["json"]["name"]:
            image.append(sample.get(name))

    if ENERGON_LT_7:
        return _attach_tools(MultiVidQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample.get("__subflavors__", {}),
            video=video if len(video) > 0 else None,
            system=system,
            messages=messages,
        ), sample["json"])
    else:
        return _attach_tools(MultiVidQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavors__=sample.get("__subflavors__", {}),
            video=video if len(video) > 0 else None,
            system=system,
            messages=messages,
        ), sample["json"])


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
    if ENERGON_LT_7:
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
      "media_type": "image" | "video" | "text"
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

    # contexts/answers are List[List[str]]. A single-turn child is a
    # one-element list; multi-turn BMR children keep all turns.
    contexts = [
        [p] if isinstance(p, str)
        else (list(p) if isinstance(p, (list, tuple)) else [])
        for p in prompts
    ]
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

    elif media_type == "text":
        images = None
        videos = None

    else:
        raise ValueError(
            f"[cooker_packed_multi_mix_qa] unknown media_type='{media_type}'. "
            f"Expect 'image', 'video', or 'text'."
        )
    if ENERGON_LT_7:
        return PackedMultiMixQASample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample.get("__subflavors__", {}),
            images=images,
            videos=videos,
            contexts=contexts,
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
            contexts=contexts,
            answers=answers,
            answer_weights=None,
        )


@stateless
def cooker_packed_chat_mix(sample: dict):
    """
    Convert packed full chat/tool-calling json into a PackedChatMixSample.

    Expected json layout (example):
    {
      "messages": [          # len = N
        {
          "messages": [...],  # OpenAI Chat Completions-style messages
          "tools": [...],     # optional tool definitions
          "source": {...}     # optional source metadata
        },
        ...
      ],
      "media_files": [       # length = N
        ["imgA.jpg", "imgB.jpg", ...],
        ["imgC.jpg", ...],
        ...
      ],
      "media_type": "image" | "video" | "text"
    }
    """
    data = sample["json"]

    messages = data.get("messages")
    if messages is None:
        messages = data.get("texts", []) or []
    if not isinstance(messages, list):
        raise ValueError(
            f"[cooker_packed_chat_mix] expected `messages` to be a list "
            f"for key={sample['__key__']}, got {type(messages).__name__}"
        )

    media_files = data.get("media_files", []) or []
    media_type = (data.get("media_type") or "").lower()

    if len(media_files) != len(messages):
        raise ValueError(
            f"[cooker_packed_chat_mix] media_files/messages length mismatch "
            f"for key={sample['__key__']}: {len(media_files)} vs {len(messages)}"
        )

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
    elif media_type == "text":
        images = None
        videos = None
    else:
        raise ValueError(
            f"[cooker_packed_chat_mix] unknown media_type='{media_type}'. "
            f"Expect 'image', 'video', or 'text'."
        )

    if ENERGON_LT_7:
        return PackedChatMixSample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample.get("__subflavors__", {}),
            packed_messages=messages,
            packed_images=images,
            packed_videos=videos,
        )
    else:
        return PackedChatMixSample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavors__=sample.get("__subflavors__", {}),
            packed_messages=messages,
            packed_images=images,
            packed_videos=videos,
        )

@stateless
def cooker_packed_caption(sample: dict):
    """Convert raw sample dict into a PackedCaptioningSample."""
    data = sample["json"]
    images = [sample.get(f"img{i}.jpg") for i in range(len(data["images"]))]
    captions = data["captions"]
    prompts = data["prompts"]
    if ENERGON_LT_7:
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


def cooker_default(sample: dict, args):
    """Fallback cooker when no subflavor matches, selected by user-defined sample_type."""
    if args.sample_type == "multi_mix_qa":
        return cooker_multi_mix_qa(sample, args)
    elif args.sample_type == "chat_mix":
        return cooker_chat_mix(sample, args)
    elif args.sample_type == "feature_vqa":
        return cooker_feature_qa(sample)
    elif args.sample_type == "packed_captioning":
        return cooker_packed_caption(sample)
    elif args.sample_type == "packed_vqa":
        return cooker_packed_vqa(sample)
    elif args.sample_type == "packed_multi_mix_qa":
        return cooker_packed_multi_mix_qa(sample)
    elif args.sample_type == "packed_chat_mix":
        return cooker_packed_chat_mix(sample)
    else:
        raise NotImplementedError("Sample format not supported", sample)
