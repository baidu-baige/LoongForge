# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Scan source WebDataset shards and build a WDS-native packing manifest."""

import io
import hashlib
import json
import logging
import re
import shutil
import tarfile
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from jinja2 import Template
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from wds_pack.core.artifacts import (
    debug_artifacts_enabled,
    keep_intermediate_artifacts,
)
from wds_pack.core.config import get_cfg, parse_args
from wds_pack.core.constants import TEMPLATES, VALID_MEDIA_EXT
from wds_pack.core.paths import (
    get_bins_dir,
    get_combined_token_report_path,
    get_manifest_jsonl_path,
    get_manifest_sqlite_path,
    get_pack_plan_path,
    get_token_report_dir,
    get_token_report_path,
    get_work_dir,
)
from wds_pack.manifest.sqlite import create_manifest, insert_manifest_row
from wds_pack.media import preprocess as media_preprocess_utils

LOG = logging.getLogger(__name__)
SUPPORTED_MEDIA_TYPES = ("text", "image", "video")
DEFAULT_MEDIA_PLACEHOLDER_TOKENS = ("<|media_content|>",)
MEDIA_GRID_KEYS = ("grid_thws", "image_grid_thw", "video_grid_thw")
ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}
FROM_ALIASES = {"user": "human", "assistant": "gpt", "tool": "tool"}
MEDIA_CONTENT_ALIASES = {
    "image": {"image", "image_url"},
    "video": {"video", "video_url"},
}
TOKENIZER_ONLY_KWARGS = {
    "pretrained_model_name_or_path",
    "cache_dir",
    "force_download",
    "local_files_only",
    "revision",
    "trust_remote_code",
    "use_fast",
    "token",
}


@dataclass
class TarMemberRef:
    """Location of one regular tar member in an uncompressed tar shard."""

    name: str
    part: str
    offset_data: int
    size: int


@dataclass
class RawSampleGroup:
    """All tar members belonging to one WDS sample key."""

    base_key: str
    members: Dict[str, TarMemberRef]
    json_bytes: Optional[bytes]


def setup_logging(work_dir: Path, log_level: str) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(work_dir / "scan_wds_manifest.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def get_chat_template(sample_type: str, model_type: str) -> Template:
    task_templates = TEMPLATES.get(sample_type)
    if task_templates is None:
        raise ValueError(f"Unsupported sample_type '{sample_type}'")
    if isinstance(task_templates, str):
        return Template(task_templates)
    template_str = task_templates.get(model_type)
    if template_str is None:
        raise ValueError(
            f"No template for sample_type={sample_type}, model_type={model_type}"
        )
    return Template(template_str)


def normalize_role(role: str) -> str:
    return ROLE_ALIASES.get((role or "").lower(), role or "")


def role_to_from(role: str) -> str:
    return FROM_ALIASES.get(role, role)


def normalize_messages_schema(text_data):
    """Normalize messages to carry both role/content and from/value fields."""
    if not isinstance(text_data, list):
        return text_data
    normalized = []
    for message in text_data:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        merged = dict(message)
        role = normalize_role(message.get("role") or message.get("from") or "")
        if role:
            merged["role"] = role
            merged["from"] = message.get("from") or role_to_from(role)

        content = message.get("content")
        if content is None:
            content = message.get("value")
        if content is not None:
            merged["content"] = content
            merged["value"] = message.get("value") if message.get("value") is not None else content
        normalized.append(merged)
    return normalized


def _first_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return ""
        return _first_text(value[0])
    if isinstance(value, dict):
        return str(value.get("content") or value.get("value") or "")
    return str(value)


def load_messages_and_pair(json_data: dict, template_text_key: str) -> Tuple[list, str, str]:
    """Return normalized messages plus first user prompt and assistant answer."""
    text_data = next(
        (
            json_data.get(key)
            for key in (template_text_key, "messages", "texts")
            if json_data.get(key) is not None
        ),
        None,
    )

    if isinstance(text_data, dict):
        prompts = text_data.get("prompts") or []
        captions = text_data.get("captions") or []
        prompt = _first_text(prompts)
        caption = _first_text(captions)
        messages = [
            {"role": "user", "from": "human", "content": prompt, "value": prompt},
            {"role": "assistant", "from": "gpt", "content": caption, "value": caption},
        ]
        return messages, prompt, caption

    messages = normalize_messages_schema(text_data or [])
    if not isinstance(messages, list):
        messages = []

    prompt = ""
    caption = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = normalize_role(msg.get("role") or msg.get("from") or "")
        content = msg.get("content")
        if content is None:
            content = msg.get("value") or ""
        if role == "user" and not prompt:
            prompt = content
        elif role == "assistant" and not caption:
            caption = content
    return messages, prompt, caption


def flatten_media_files(media_files) -> List[str]:
    def walk(value):
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from walk(item)
            return
        yield str(value)

    return list(walk(media_files))


def infer_media_type_from_part(part: str) -> Optional[str]:
    ext = Path(part).suffix.lower() or f".{part.lower()}"
    for media_type, ext_list in VALID_MEDIA_EXT.items():
        if ext in ext_list:
            return media_type
    return None


def parse_member_name(name: str) -> Optional[Tuple[str, str]]:
    """Split a WDS tar member name into base key and part key."""
    if "/" in name and Path(name).parts[0].startswith("__"):
        return None
    base, dot, part = name.rpartition(".")
    if not dot or "/" in part:
        return None
    return base, part


def iter_tar_groups(tar: tarfile.TarFile) -> Iterable[RawSampleGroup]:
    current_base = None
    current_members: Dict[str, TarMemberRef] = {}
    current_json = None

    def emit():
        if current_base is None:
            return None
        return RawSampleGroup(
            base_key=current_base,
            members=current_members,
            json_bytes=current_json,
        )

    for member in tar:
        if not member.isfile() or member.name is None:
            continue
        parsed = parse_member_name(member.name)
        if parsed is None:
            continue
        base, part = parsed
        if current_base is not None and base != current_base:
            group = emit()
            if group is not None:
                yield group
            current_members = {}
            current_json = None
        current_base = base
        current_members[part] = TarMemberRef(
            name=member.name,
            part=part,
            offset_data=member.offset_data,
            size=member.size,
        )
        if part == "json":
            extracted = tar.extractfile(member)
            current_json = extracted.read() if extracted is not None else None

    group = emit()
    if group is not None:
        yield group


def get_processor_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def install_chat_template(processor, model_cfg: dict):
    chat_template_path = model_cfg.get("chat_template_path")
    if not chat_template_path:
        return processor

    path = Path(chat_template_path)
    if not path.is_file():
        raise FileNotFoundError(f"chat_template_path not found: {path}")
    template_text = path.read_text(encoding="utf-8")

    tokenizer = get_processor_tokenizer(processor)
    setattr(tokenizer, "chat_template", template_text)
    if tokenizer is not processor and hasattr(processor, "chat_template"):
        setattr(processor, "chat_template", template_text)
    return processor


def load_processor(model_cfg: dict):
    kwargs = dict(model_cfg.get("processor_kwargs", {}))
    loader = model_cfg.get("processor_loader", "auto_processor")
    if loader == "auto_tokenizer":
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in TOKENIZER_ONLY_KWARGS
        }
        return install_chat_template(AutoTokenizer.from_pretrained(**kwargs), model_cfg)
    if loader != "auto_processor":
        raise ValueError(f"Unsupported model.processor_loader: {loader}")
    return install_chat_template(AutoProcessor.from_pretrained(**kwargs), model_cfg)


def build_media_preprocess(cfg: dict) -> Dict[str, Callable]:
    funcs = {}
    for media_type, func_name in cfg.get("media_preprocess", {}).items():
        funcs[media_type] = getattr(media_preprocess_utils, func_name)
    return funcs


def preprocess_media_bytes(media_type: str, part: str, data: bytes, funcs: Dict[str, Callable]):
    func = funcs.get(media_type)
    if func is None:
        raise ValueError(f"No media_preprocess function configured for {media_type}")
    if media_type == "image":
        return func(io.BytesIO(data))

    suffix = Path(part).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        return func(Path(tmp.name))


def _input_ids_len(model_inputs) -> int:
    input_ids = model_inputs["input_ids"]
    if not hasattr(input_ids, "shape"):
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    if len(input_ids.shape) == 1:
        return int(input_ids.shape[0])
    return int(input_ids.shape[1])


def _get_config_value(config, *names, default=None):
    for name in names:
        if isinstance(config, dict) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _media_merge_hw(processor) -> Tuple[int, int]:
    candidates = [
        getattr(getattr(processor, "media_processor", None), "media_proc_cfg", None),
        getattr(getattr(processor, "image_processor", None), "media_proc_cfg", None),
        getattr(processor, "image_processor", None),
        processor,
    ]
    merge_size = None
    for candidate in candidates:
        if candidate is None:
            continue
        merge_size = _get_config_value(
            candidate,
            "merge_kernel_size",
            "merge_size",
            "spatial_merge_size",
        )
        if merge_size is not None:
            break

    if merge_size is None:
        merge_size = 2
    if isinstance(merge_size, (list, tuple)):
        return int(merge_size[0]), int(merge_size[1])
    merge_size = int(merge_size)
    return merge_size, merge_size


def _grid_rows(grid_value) -> List[Sequence[int]]:
    if grid_value is None:
        return []
    if hasattr(grid_value, "tolist"):
        grid_value = grid_value.tolist()
    if not grid_value:
        return []
    if not isinstance(grid_value[0], (list, tuple)):
        return [grid_value]
    return grid_value


def _media_feature_lengths_from_grid(processor, model_inputs) -> List[int]:
    merge_h, merge_w = _media_merge_hw(processor)
    lengths = []
    for key in MEDIA_GRID_KEYS:
        for row in _grid_rows(model_inputs.get(key)):
            if len(row) < 3:
                continue
            _, h, w = row[:3]
            lengths.append((int(h) // merge_h) * (int(w) // merge_w))
    return lengths


def _count_media_placeholder_tokens(processor, model_inputs, placeholder_tokens: Sequence[str]) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert_tokens_to_ids is None:
        return 0
    placeholder_ids = {
        convert_tokens_to_ids(token)
        for token in placeholder_tokens
        if token
    }
    placeholder_ids.discard(None)
    if not placeholder_ids:
        return 0

    input_ids = model_inputs["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        return sum(
            1
            for row in input_ids
            for token_id in row
            if token_id in placeholder_ids
        )
    return sum(1 for token_id in input_ids if token_id in placeholder_ids)


def _encode_text_input_ids(processor, text_input: str) -> Optional[List[int]]:
    """Encode rendered chat text with the direct tokenizer API used at runtime."""
    tokenizer = getattr(processor, "tokenizer", processor)
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return None

    try:
        input_ids = encode(text_input, add_special_tokens=False)
    except TypeError:
        input_ids = encode(text_input)

    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def _count_media_placeholder_token_ids(
    processor,
    input_ids: Sequence[int],
    placeholder_tokens: Sequence[str],
) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert_tokens_to_ids is None:
        return 0
    placeholder_ids = {
        convert_tokens_to_ids(token)
        for token in placeholder_tokens
        if token
    }
    placeholder_ids.discard(None)
    if not placeholder_ids:
        return 0
    return sum(1 for token_id in input_ids if token_id in placeholder_ids)


def _as_media_list(media_inputs: dict) -> List[dict]:
    medias = []
    for image in media_inputs.get("images", []):
        medias.append({"type": "image", "image": image})
    for video in media_inputs.get("videos", []):
        medias.append({"type": "video", "video": video})
    return medias


def _processor_model_inputs(processor, text_input: str, media_inputs: dict):
    kwargs = {"padding": True, "return_tensors": "pt"}
    if not media_inputs:
        tokenizer = getattr(processor, "tokenizer", processor)
        try:
            return tokenizer(text_input, return_tensors="pt")
        except Exception:
            try:
                return processor(text=[text_input], **kwargs)
            except TypeError:
                return processor([text_input], **kwargs)

    if getattr(processor, "media_processor", None) is not None:
        try:
            return processor(
                text=text_input,
                medias=_as_media_list(media_inputs),
                return_tensors="pt",
            )
        except TypeError:
            pass

    try:
        return processor(text=[text_input], **media_inputs, **kwargs)
    except TypeError:
        return processor(
            text=text_input,
            medias=_as_media_list(media_inputs),
            return_tensors="pt",
        )


def _compute_processor_token_len(
    processor,
    text_input: str,
    media_inputs: dict,
    placeholder_tokens: Sequence[str] = DEFAULT_MEDIA_PLACEHOLDER_TOKENS,
) -> int:
    direct_input_ids = _encode_text_input_ids(processor, text_input)
    if not media_inputs and direct_input_ids is not None:
        return len(direct_input_ids)

    model_inputs = _processor_model_inputs(processor, text_input, media_inputs)
    token_len = (
        len(direct_input_ids)
        if direct_input_ids is not None
        else _input_ids_len(model_inputs)
    )

    feature_lengths = _media_feature_lengths_from_grid(processor, model_inputs)
    if not feature_lengths:
        return token_len

    if direct_input_ids is not None:
        placeholder_count = _count_media_placeholder_token_ids(
            processor, direct_input_ids, placeholder_tokens
        )
    else:
        placeholder_count = _count_media_placeholder_tokens(
            processor, model_inputs, placeholder_tokens
        )
    replaced_tokens = min(placeholder_count, len(feature_lengths))
    return token_len - replaced_tokens + sum(feature_lengths)


def compute_token_len(processor, text_input: str, media_inputs: dict, model_cfg=None) -> int:
    if isinstance(model_cfg, str):
        model_cfg = {"model_type": model_cfg}
    model_cfg = model_cfg or {}
    placeholder_tokens = model_cfg.get(
        "media_placeholder_tokens", DEFAULT_MEDIA_PLACEHOLDER_TOKENS
    )
    if isinstance(placeholder_tokens, str):
        placeholder_tokens = (placeholder_tokens,)
    return _compute_processor_token_len(
        processor,
        text_input,
        media_inputs,
        placeholder_tokens=placeholder_tokens,
    )


def split_text_with_media_placeholders(content: str, media_type: str, expected_count: int):
    placeholder = f"<{media_type}>"
    matches = list(re.finditer(re.escape(placeholder), content))
    if len(matches) != expected_count:
        raise ValueError(
            f"{media_type} sample expects {expected_count} media placeholder(s), "
            f"found {len(matches)} occurrence(s) of {placeholder!r}"
        )

    parts = []
    cursor = 0
    for match in matches:
        segment = content[cursor:match.start()]
        if segment:
            parts.append({"type": "text", "text": segment})
        parts.append({"type": media_type})
        cursor = match.end()
    tail = content[cursor:]
    if tail:
        parts.append({"type": "text", "text": tail})
    return parts


def count_structured_media_parts(content, media_type: str) -> int:
    if not isinstance(content, list):
        return 0
    aliases = MEDIA_CONTENT_ALIASES.get(media_type, {media_type})
    return sum(
        1
        for part in content
        if isinstance(part, dict) and part.get("type") in aliases
    )


def prepare_messages_for_hf_chat_template(messages: Sequence[dict], media_type: str, media_count: int):
    rendered_messages = []
    remaining_media = media_count
    for message in messages:
        role = normalize_role(message.get("role") or message.get("from") or "")
        content = message.get("content")
        if content is None:
            content = message.get("value") or ""

        if (
            role == "user"
            and media_type in ("image", "video")
            and remaining_media > 0
        ):
            if isinstance(content, str):
                current_count = content.count(f"<{media_type}>")
                if current_count > 0:
                    content = split_text_with_media_placeholders(
                        content, media_type, current_count
                    )
                    remaining_media -= current_count
            else:
                remaining_media -= count_structured_media_parts(content, media_type)

        rendered_message = dict(message)
        rendered_message["role"] = role
        rendered_message["content"] = content
        rendered_messages.append(rendered_message)

    if media_type in ("image", "video") and remaining_media != 0:
        raise ValueError(
            f"{media_type} sample has {media_count} media file(s), "
            f"but only matched {media_count - remaining_media} placeholder(s)"
        )

    return rendered_messages


def should_use_hf_chat_template(processor, cfg: dict) -> bool:
    model_cfg = cfg.get("model", {})
    if model_cfg.get("use_hf_chat_template") is not None:
        return bool(model_cfg.get("use_hf_chat_template"))
    return bool(model_cfg.get("chat_template_path"))


def prepare_tools_for_hf_render(tokenizer, tools: Optional[Sequence[dict]], kwargs: dict):
    if not tools:
        return None, kwargs

    tools = list(tools)
    apply_chat_template = tokenizer.apply_chat_template
    if hasattr(apply_chat_template, "__func__"):
        apply_chat_template = apply_chat_template.__func__
    apply_globals = getattr(apply_chat_template, "__globals__", {})
    deep_sort_dict = apply_globals.get("deep_sort_dict")
    encode_tools_to_typescript_style = apply_globals.get(
        "encode_tools_to_typescript_style"
    )

    if deep_sort_dict is not None:
        tools = deep_sort_dict(tools)

    if (
        "tools_ts_str" not in kwargs
        and encode_tools_to_typescript_style is not None
    ):
        try:
            kwargs["tools_ts_str"] = encode_tools_to_typescript_style(tools)
        except Exception as exc:
            LOG.warning(
                "Failed to render tools_ts_str with HF tokenizer helper; "
                "falling back to raw tools for chat template rendering: %s",
                exc,
            )

    return tools, kwargs


def render_chat_text(
    processor,
    messages: Sequence[dict],
    tools: Optional[Sequence[dict]],
    cfg: dict,
    fallback_template: Optional[Template],
    media_type: str,
    media_count: int,
) -> str:
    if should_use_hf_chat_template(processor, cfg):
        tokenizer = get_processor_tokenizer(processor)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "Configured HF chat template path, but tokenizer has no apply_chat_template"
            )
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "Configured HF chat template rendering, but tokenizer.chat_template is empty"
            )
        rendered_messages = prepare_messages_for_hf_chat_template(
            messages, media_type, media_count
        )
        chat_kwargs = dict(cfg.get("model", {}).get("chat_template_kwargs", {}))
        chat_kwargs.pop("add_generation_prompt", None)
        chat_kwargs.pop("tokenize", None)
        chat_kwargs.pop("tools", None)
        chat_kwargs.setdefault("thinking", False)
        tools, chat_kwargs = prepare_tools_for_hf_render(tokenizer, tools, chat_kwargs)
        if tools:
            chat_kwargs["tools"] = tools
        return tokenizer.apply_chat_template(
            rendered_messages,
            tokenize=False,
            add_generation_prompt=False,
            **chat_kwargs,
        )

    if fallback_template is None:
        raise ValueError("No fallback chat template configured")
    template_text_key = cfg.get("data", {}).get("template_text_key", "messages")
    render_payload = {template_text_key: messages, "messages": messages, "tools": tools}
    return fallback_template.render(**render_payload)


def safe_sample_id(shard: Path, shard_root: Path, base_key: str) -> str:
    rel_shard = str(shard.relative_to(shard_root))
    digest = hashlib.sha1(f"{rel_shard}\0{base_key}".encode("utf-8")).hexdigest()
    return f"{shard.stem}__{digest[:16]}"


def skip_row(reason: str, base_key: str, **fields) -> dict:
    return {"reason": reason, "base_key": base_key, **fields}


def decode_group_json(group: RawSampleGroup) -> Tuple[Optional[dict], Optional[dict]]:
    if group.json_bytes is None:
        return None, skip_row("missing_json", group.base_key)
    try:
        json_data = json.loads(group.json_bytes.decode("utf-8"))
    except Exception as exc:
        return None, skip_row("invalid_json", group.base_key, error=str(exc))
    if not isinstance(json_data, dict):
        return None, skip_row("json_not_object", group.base_key)
    return json_data, None


def get_sample_media_type(json_data: dict) -> str:
    return (json_data.get("media_type") or json_data.get("media") or "text").lower()


def resolve_media_files(json_data: dict, group: RawSampleGroup, media_type: str) -> List[str]:
    raw_media_files = json_data.get("media_files")
    if raw_media_files is None:
        raw_media_files = json_data.get("name")
    media_files = flatten_media_files(raw_media_files)
    if media_type != "text" and not media_files:
        media_files = [
            part
            for part in group.members
            if part != "json" and infer_media_type_from_part(part) == media_type
        ]
    return [] if media_type == "text" else media_files


def build_media_inputs(
    raw_file,
    group: RawSampleGroup,
    media_type: str,
    media_files: Sequence[str],
    media_preprocess: Dict[str, Callable],
) -> dict:
    if media_type not in ("image", "video"):
        return {}

    media_values = []
    for part in media_files:
        member_ref = group.members[part]
        raw_file.seek(member_ref.offset_data)
        media_values.append(
            preprocess_media_bytes(
                media_type,
                part,
                raw_file.read(member_ref.size),
                media_preprocess,
            )
        )
    return {f"{media_type}s": media_values}


def manifest_member_rows(group: RawSampleGroup, media_files: Sequence[str]) -> List[dict]:
    return [
        {
            "part": part,
            "member": group.members[part].name,
            "offset_data": group.members[part].offset_data,
            "size": group.members[part].size,
        }
        for part in media_files
    ]


def build_manifest_row(
    *,
    sample_id: str,
    shard_path: Path,
    shard_root: Path,
    group: RawSampleGroup,
    media_type: str,
    token_len: int,
    prompt: str,
    caption: str,
    media_files: Sequence[str],
    raw_json: dict,
) -> dict:
    return {
        "sample_id": sample_id,
        "media_type": media_type,
        "token_len": token_len,
        "shard": str(shard_path.relative_to(shard_root)),
        "base_key": group.base_key,
        "prompt": prompt,
        "caption": caption,
        "media_files": list(media_files),
        "raw_json": raw_json,
        "members": manifest_member_rows(group, media_files),
    }


def process_group(
    *,
    shard_path: Path,
    shard_root: Path,
    raw_file,
    group: RawSampleGroup,
    chat_template: Optional[Template],
    processor,
    cfg: dict,
    media_preprocess: Dict[str, Callable],
) -> Tuple[Optional[dict], Optional[dict]]:
    json_data, skip = decode_group_json(group)
    if skip is not None:
        return None, skip

    media_type = get_sample_media_type(json_data)
    if media_type not in SUPPORTED_MEDIA_TYPES:
        return None, skip_row(
            "unsupported_media_type", group.base_key, media_type=media_type
        )

    template_text_key = cfg.get("data", {}).get("template_text_key", "messages")
    messages, prompt, caption = load_messages_and_pair(json_data, template_text_key)
    if not messages:
        return None, skip_row("missing_messages", group.base_key)
    tools = json_data.get("tools")

    media_files = resolve_media_files(json_data, group, media_type)
    missing_media = [part for part in media_files if part not in group.members]
    if missing_media:
        return None, skip_row("missing_media", group.base_key, missing_media=missing_media)

    try:
        media_inputs = build_media_inputs(
            raw_file, group, media_type, media_files, media_preprocess
        )
    except Exception as exc:
        return None, {
            "reason": "media_preprocess_failed",
            "base_key": group.base_key,
            "error": str(exc),
        }

    try:
        text_input = render_chat_text(
            processor,
            messages,
            tools,
            cfg,
            chat_template,
            media_type,
            len(media_files),
        )
    except Exception as exc:
        return None, {
            "reason": "render_chat_template_failed",
            "base_key": group.base_key,
            "error": str(exc),
        }

    try:
        token_len = compute_token_len(
            processor,
            text_input,
            media_inputs,
            cfg.get("model", {}),
        )
    except Exception as exc:
        return None, {
            "reason": "tokenize_failed",
            "base_key": group.base_key,
            "error": str(exc),
        }

    max_token_len = int(cfg["sample"]["max_token_len"])
    sample_id = safe_sample_id(shard_path, shard_root, group.base_key)
    if token_len > max_token_len:
        return None, skip_row(
            "overlong",
            group.base_key,
            sample_id=sample_id,
            media_type=media_type,
            token_len=token_len,
            max_token_len=max_token_len,
        )

    return build_manifest_row(
        sample_id=sample_id,
        shard_path=shard_path,
        shard_root=shard_root,
        group=group,
        media_type=media_type,
        token_len=token_len,
        prompt=prompt,
        caption=caption,
        media_files=media_files,
        raw_json=json_data,
    ), None


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scan_shard(shard_path_str: str, cfg: dict) -> dict:
    shard_path = Path(shard_path_str)
    wds_dir = Path(cfg["data"]["wds_dir"])
    work_dir = get_work_dir(cfg)
    keep_debug = debug_artifacts_enabled(cfg)
    part_dir = work_dir / "manifest_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    media_dir = part_dir / shard_path.stem
    if keep_debug:
        media_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor(cfg.get("model", {}))
    chat_template = None
    if not should_use_hf_chat_template(processor, cfg):
        chat_template = get_chat_template(
            cfg["sample"]["sample_type"], cfg.get("model", {}).get("model_type", "")
        )
    media_preprocess = build_media_preprocess(cfg)

    rows = []
    skipped = []
    token_rows = {media_type: [] for media_type in SUPPORTED_MEDIA_TYPES} if keep_debug else {}

    with shard_path.open("rb") as raw_file, tarfile.open(shard_path, "r:") as tar:
        for group in iter_tar_groups(tar):
            row, skip = process_group(
                shard_path=shard_path,
                shard_root=wds_dir,
                raw_file=raw_file,
                group=group,
                chat_template=chat_template,
                processor=processor,
                cfg=cfg,
                media_preprocess=media_preprocess,
            )
            if row is not None:
                rows.append(row)
                if keep_debug:
                    token_rows[row["media_type"]].append(
                        f"{row['sample_id']}: {row['token_len']}\n"
                    )
            elif skip is not None:
                skip["shard"] = str(shard_path.relative_to(wds_dir))
                skipped.append(skip)

    manifest_part = part_dir / f"{shard_path.stem}.manifest.jsonl"
    skipped_part = part_dir / f"{shard_path.stem}.skipped.jsonl"
    write_jsonl(manifest_part, rows)
    write_jsonl(skipped_part, skipped)
    token_paths = {}
    if keep_debug:
        for media_type, lines in token_rows.items():
            path = media_dir / f"sample_len_report_{media_type}.txt"
            path.write_text("".join(lines), encoding="utf-8")
            token_paths[media_type] = str(path)

    return {
        "shard": str(shard_path),
        "manifest": str(manifest_part),
        "skipped": str(skipped_part),
        "token_paths": token_paths,
        "rows": len(rows),
        "skipped_rows": len(skipped),
    }


def merge_outputs(cfg: dict, shard_results: Sequence[dict]) -> None:
    work_dir = get_work_dir(cfg)
    keep_debug = debug_artifacts_enabled(cfg)
    manifest_jsonl = get_manifest_jsonl_path(cfg)
    manifest_sqlite = get_manifest_sqlite_path(cfg)
    combined_report = get_combined_token_report_path(cfg)
    skipped_path = work_dir / "skipped_samples.jsonl"
    overlong_path = work_dir / "skipped_overlong.jsonl"

    conn = create_manifest(manifest_sqlite)
    seen = set()
    with ExitStack() as stack:
        manifest_out = (
            stack.enter_context(manifest_jsonl.open("w", encoding="utf-8"))
            if keep_debug else None
        )
        combined_out = (
            stack.enter_context(combined_report.open("w", encoding="utf-8"))
            if keep_debug else None
        )
        skipped_out = (
            stack.enter_context(skipped_path.open("w", encoding="utf-8"))
            if keep_debug else None
        )
        overlong_out = stack.enter_context(overlong_path.open("w", encoding="utf-8"))

        if keep_debug:
            for media_type in SUPPORTED_MEDIA_TYPES:
                get_token_report_path(cfg, media_type).write_text("", encoding="utf-8")

        for result in shard_results:
            with Path(result["manifest"]).open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample_id = row["sample_id"]
                    if sample_id in seen:
                        raise ValueError(f"Duplicate sample_id: {sample_id}")
                    seen.add(sample_id)
                    if keep_debug:
                        manifest_out.write(line)
                        combined_out.write(f"{sample_id}: {row['token_len']}\n")
                    insert_manifest_row(conn, row)

            with Path(result["skipped"]).open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if keep_debug:
                        skipped_out.write(line)
                    row = json.loads(line)
                    if row.get("reason") == "overlong":
                        overlong_out.write(line)

            if keep_debug:
                for media_type, token_path in result["token_paths"].items():
                    target = get_token_report_path(cfg, media_type)
                    with target.open("a", encoding="utf-8") as out, Path(token_path).open(
                        "r", encoding="utf-8"
                    ) as src:
                        for line in src:
                            out.write(line)

    conn.commit()
    conn.close()
    if not keep_intermediate_artifacts(cfg):
        shutil.rmtree(work_dir / "manifest_parts", ignore_errors=True)


def reset_work_artifacts(cfg: dict) -> None:
    work_dir = get_work_dir(cfg)
    for path in (
        work_dir / "manifest_parts",
        get_token_report_dir(cfg),
        get_bins_dir(cfg),
    ):
        if path.exists():
            shutil.rmtree(path)

    for path in (
        get_manifest_jsonl_path(cfg),
        get_manifest_sqlite_path(cfg),
        get_combined_token_report_path(cfg),
        get_pack_plan_path(cfg),
        work_dir / "skipped_samples.jsonl",
        work_dir / "skipped_overlong.jsonl",
        work_dir / "unpacked_samples.jsonl",
    ):
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    cfg = get_cfg(args.config)
    work_dir = get_work_dir(cfg)
    log_level = cfg.get("log", {}).get("level", "INFO")
    setup_logging(work_dir, log_level)

    sample_type = cfg.get("sample", {}).get("sample_type")
    supported_sample_types = {"packed_multi_mix_qa", "packed_chat_mix"}
    if sample_type not in supported_sample_types:
        raise ValueError(
            "WDS-native V1 only supports sample.sample_type in "
            f"{sorted(supported_sample_types)}, got {sample_type!r}"
        )

    model_cfg = cfg.get("model", {})
    if (
        sample_type == "packed_chat_mix"
        and not model_cfg.get("use_hf_chat_template")
        and not model_cfg.get("chat_template_path")
    ):
        raise ValueError(
            "packed_chat_mix requires HF chat template rendering. "
            "Set model.use_hf_chat_template=true or model.chat_template_path."
        )

    wds_dir = Path(cfg["data"]["wds_dir"])
    shards = sorted(wds_dir.glob("*.tar"))
    max_shards = int(cfg.get("process", {}).get("max_shards", 0) or 0)
    if max_shards > 0:
        shards = shards[:max_shards]
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {wds_dir}")

    reset_work_artifacts(cfg)
    if debug_artifacts_enabled(cfg):
        get_token_report_dir(cfg)
    get_bins_dir(cfg)
    get_pack_plan_path(cfg).parent.mkdir(parents=True, exist_ok=True)

    workers = int(cfg.get("process", {}).get("workers", 1) or 1)
    LOG.info("Scanning %d shards with %d worker(s)", len(shards), workers)

    if workers <= 1:
        shard_results = [
            scan_shard(str(shard), cfg)
            for shard in tqdm(shards, desc="scan shards", unit="shard")
        ]
    else:
        import multiprocessing as mp

        with mp.Pool(processes=workers) as pool:
            shard_results = list(
                tqdm(
                    pool.starmap(scan_shard, [(str(shard), cfg) for shard in shards]),
                    total=len(shards),
                    desc="scan shards",
                    unit="shard",
                )
            )

    merge_outputs(cfg, shard_results)
    if debug_artifacts_enabled(cfg):
        LOG.info(
            "Manifest ready: %s, %s",
            get_manifest_jsonl_path(cfg),
            get_manifest_sqlite_path(cfg),
        )
    else:
        LOG.info("Manifest ready: %s", get_manifest_sqlite_path(cfg))


if __name__ == "__main__":
    main()
