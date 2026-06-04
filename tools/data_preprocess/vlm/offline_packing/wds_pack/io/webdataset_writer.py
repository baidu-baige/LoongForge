# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Script for converting packed samples to WebDataset format."""

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import yaml
import webdataset as wds
from tqdm import tqdm

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from wds_pack.core.config import get_cfg, parse_args
from wds_pack.core.paths import (
    get_init_file,
    get_manifest_sqlite_path,
    get_pack_plan_path,
)
from wds_pack.manifest.sqlite import load_manifest_for_packing

LOG = logging.getLogger(__name__)


class ShardReaderCache:
    """Read byte ranges from source tar shards without extracting files."""

    def __init__(self, root: Path):
        self.root = root
        self._files = {}

    def read(self, shard: str, offset: int, size: int) -> bytes:
        path = self.root / shard
        fh = self._files.get(path)
        if fh is None:
            fh = path.open("rb")
            self._files[path] = fh
        fh.seek(offset)
        return fh.read(size)

    def close(self):
        for fh in self._files.values():
            fh.close()
        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    
def _flatten_media_paths(media: Sequence) -> List[str]:
    """Flatten nested media lists to a simple list of string paths."""
    flattened: List[str] = []
    for item in media:
        if isinstance(item, (list, tuple)):
            flattened.extend(_flatten_media_paths(item))
        elif isinstance(item, str):
            flattened.append(item)
    return flattened


def stream_samples_caption(src_dir: Path) -> Iterator[dict]:
    if not src_dir.exists():
        raise FileNotFoundError(f"Packed json directory not found: {src_dir}")

    for json_path in sorted(src_dir.glob("*.json")):
        sample_id = json_path.stem
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        texts = raw.get("texts", {})
        captions = raw.get("captions") or texts.get("captions") or []
        prompts = raw.get("prompts") or texts.get("prompts") or []
        images = raw.get("images") or []
        images = _flatten_media_paths(images)

        yield {
            "id": sample_id,
            "images": images,
            "prompts": prompts,
            "captions": captions,
        }


def stream_samples_packed_multi_mix_qa(src_dir: Path) -> Iterator[dict]:
    """Stream packed multi-mix qa entries with cooker-ready fields."""
    if not src_dir.exists():
        raise FileNotFoundError(f"Packed json directory not found: {src_dir}")

    for json_path in sorted(src_dir.glob("*.json")):
        sample_id = json_path.stem
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        texts = raw.get("texts", {})
        prompts = raw.get("prompts") or texts.get("prompts") or []
        captions = raw.get("captions") or texts.get("captions") or []
        media_files = raw.get("media_files") or []
        media_type = (raw.get("media_type") or raw.get("media") or "").lower()

        if media_type not in {"image", "video", "text"}:
            raise ValueError(
                f"[{sample_id}] unsupported media_type='{media_type}', "
                f"expected 'image', 'video', or 'text'"
            )
        if len(prompts) != len(captions):
            raise ValueError(
                f"[{sample_id}] prompts/captions length mismatch: {len(prompts)} vs {len(captions)}"
            )
        if media_type == "text":
            media_files = []
        elif len(media_files) != len(prompts):
            raise ValueError(
                f"[{sample_id}] media_files length mismatch: {len(media_files)} vs {len(prompts)} prompts"
            )

        yield {
            "id": sample_id,
            "prompts": prompts,
            "captions": captions,
            "media_files": media_files,
            "media_type": media_type,
        }


def _candidate_media_paths(raw_path: str, sample_id: str) -> List[Path]:
    """
    Build a list of possible on-disk paths for a media entry.

    Some pipelines keep the original filename (Bk1Gh...jpg) in JSON but store the
    physical file with the JSON stem prefixed (ps_00000000.Bk1Gh...jpg). To make
    the lookup robust we try both variants.
    """
    base_candidate = Path(raw_path)
    candidates: List[Path] = [base_candidate]

    if sample_id:
        prefixed_name = base_candidate.with_name(f"{sample_id}.{base_candidate.name}")
        if prefixed_name not in candidates:
            candidates.append(prefixed_name)

    return candidates

def _dedup_paths(paths: Sequence[Optional[Path]]) -> List[Path]:
    seen = set()
    deduped: List[Path] = []
    for path in paths:
        if path is None:
            continue
        resolved = Path(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _resolve_media_path(
    raw_path: str,
    search_roots: Sequence[Path],
    sample_id: str,
) -> Path:
    candidates = _candidate_media_paths(raw_path, sample_id)

    # 1) Absolute paths are honored directly.
    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate

    # 2) Try each search root with the provided relative path.
    for root in search_roots:
        for candidate in candidates:
            combined = root / candidate
            if combined.exists():
                return combined

    # 3) Fallback: glob by suffix within each search root to handle prefixed filenames.
    for root in search_roots:
        for candidate in candidates:
            matches = sorted(root.glob(f"*{candidate.name}"))
            if not matches:
                continue
            if len(matches) > 1:
                chosen = matches[0]
                LOG.warning(
                    "Multiple files ending with '%s' found in '%s' for sample '%s'. "
                    "Selecting '%s'.",
                    candidate.name,
                    root,
                    sample_id,
                    chosen,
                )
                return chosen
            return matches[0]

    # 4) Fall back to the current working directory lookup if nothing else matched.
    for candidate in candidates:
        if candidate.exists():
            return candidate

    search_hint = ", ".join(str(r) for r in search_roots) or "current directory"
    raise FileNotFoundError(
        f"Image '{raw_path}' referenced by sample '{sample_id}' not found in {search_hint}"
    )


def construct_sample(entry, image_roots: Optional[Sequence[Path]] = None):
    """Pack the entire sample"""
    sample_id = entry["id"]
    sample = {"__key__": sample_id, "__restore_key__": sample_id}
    roots = _dedup_paths(image_roots or [])
    normalized_images = []
    for idx, img_path in enumerate(entry["images"]):
        resolved_path = _resolve_media_path(img_path, roots, entry["id"])
        suffix = resolved_path.suffix or Path(img_path).suffix or ".jpg"
        target_name = f"img{idx}{suffix}"
        normalized_images.append(target_name)
        with resolved_path.open("rb") as f:
            sample[target_name] = f.read()

    payload = {
        "prompts": entry["prompts"],
        "captions": entry["captions"],
        "images": normalized_images,
    }
    sample["json"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return sample


def _normalize_media_name(name: str, sample_id: str) -> str:
    """Preserve media filename for storage while ensuring it is a string."""
    return str(name)


def _normalize_media_files(media_files, sample_id: str):
    """Apply _normalize_media_name to every media entry while preserving structure."""
    normalized = []
    for item in media_files:
        if isinstance(item, (list, tuple)):
            normalized.append(_normalize_media_files(item, sample_id))
        else:
            normalized.append(_normalize_media_name(str(item), sample_id))
    return normalized


def _log_json_presence(sample_id: str, sample: dict, index: int, log_every: int = 2000) -> None:
    """
    Log whether the packed sample contains json bytes.

    Emits a warning if json is missing or not bytes; otherwise logs basic info
    for the first few samples and then periodically.
    """
    json_blob = sample.get("json")
    media_keys = [k for k in sample.keys() if k not in {"__key__", "__restore_key__", "json"}]
    if not isinstance(json_blob, (bytes, bytearray)):
        LOG.warning(
            "Sample '%s' missing json payload or wrong type (got %s). Keys=%s",
            sample_id,
            type(json_blob).__name__,
            media_keys,
        )
        return

    if index < 3 or (log_every and index % log_every == 0):
        LOG.info(
            "Sample '%s' json bytes=%d, media keys=%s",
            sample_id,
            len(json_blob),
            media_keys,
        )


def construct_sample_packed_multi_mix_qa(entry, media_roots: Optional[Sequence[Path]] = None):
    """Pack multi-mix qa sample using cooker-aligned json layout."""
    sample_id = entry["id"]
    roots = _dedup_paths(media_roots or [])

    normalized_media_files = _normalize_media_files(entry["media_files"], sample_id)
    flat_raw = _flatten_media_paths(entry["media_files"])
    flat_norm = _flatten_media_paths(normalized_media_files)
    if len(flat_raw) != len(flat_norm):
        raise ValueError(f"[{sample_id}] media flatten mismatch: {len(flat_raw)} vs {len(flat_norm)}")

    sample = {"__key__": sample_id, "__restore_key__": sample_id}
    for raw_name, target_name in zip(flat_raw, flat_norm):
        resolved_path = _resolve_media_path(raw_name, roots, sample_id)
        with resolved_path.open("rb") as f:
            sample[target_name] = f.read()

    payload = {
        "texts": {
            "captions": entry["captions"],
            "prompts": entry["prompts"],
        },
        "media_files": normalized_media_files,
        "media_type": entry["media_type"],
    }
    sample["json"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return sample


def construct_sample_caption(cfg: "PackedWDSConfig", vision, path, entry):
    """ construct webdataset sample """
    assert vision == 'image' or vision == 'video'
    directory = cfg.image_dir if vision == 'image' else cfg.video_dir
    if directory is None:
        raise ValueError(f"Missing directory for vision mode '{vision}'")

    with open(os.path.join(directory, path), "rb") as vision_file:
        vision_data = vision_file.read()
    sample = {
        "__key__": entry.get('id', path).replace('.', '_'),
        "jpg" if vision == 'image' else 'mp4': vision_data,
        "json": json.dumps(entry[cfg.columns_messages]).encode("utf-8"),
    }
    return sample


@dataclass
class PackedWDSConfig:
    output_dir: Path
    json_dir: Path
    image_dir: Optional[Path]
    video_dir: Optional[Path]
    maxcount: int
    maxsize: int
    media: str
    mode: str
    columns_messages: str
    sample_type: str
    image_search_dirs: List[Path]


def build_runtime_config(cfg: dict) -> PackedWDSConfig:
    """Adapt omni packing config.yaml to the fields required by this script."""
    _, _, packed_files_dir, wds_dir = get_init_file(cfg)
    packed_root = Path(packed_files_dir)
    raw_wds_dir = Path(wds_dir)
    data_cfg = cfg.get("data", {})
    packed_wds_cfg = cfg.get("packed_wds", {})

    output_dir = Path(
        data_cfg.get("packed_wds_dir")
        or (packed_root / "packed_wds")
    )
    json_dir = Path(
        packed_wds_cfg.get("json_dir") or packed_root / "row_packing_jsons"
    )
    if not json_dir.exists():
        raise FileNotFoundError(f"Packed json directory not found: {json_dir}")

    image_dir = packed_wds_cfg.get("image_dir")
    if image_dir is None:
        image_dir = packed_root / "row_packing_images"
    else:
        image_dir = Path(image_dir)

    video_dir = packed_wds_cfg.get("video_dir")
    if video_dir:
        video_dir = Path(video_dir)

    maxcount = int(packed_wds_cfg.get("maxcount", 1000000))
    maxsize = int(packed_wds_cfg.get("maxsize", 100000000))
    media = packed_wds_cfg.get("media", "image")
    mode = packed_wds_cfg.get("mode", "caption_pack")
    columns_messages = data_cfg.get("template_text_key", "messages")
    sample_cfg = cfg.get("sample", {})
    sample_type = sample_cfg.get("sample_type", "packed_captioning")

    image_search_dirs = _dedup_paths([image_dir, video_dir, raw_wds_dir])

    return PackedWDSConfig(
        output_dir=output_dir,
        json_dir=json_dir,
        image_dir=image_dir,
        video_dir=video_dir,
        maxcount=maxcount,
        maxsize=maxsize,
        media=media,
        mode=mode,
        columns_messages=columns_messages,
        sample_type=sample_type,
        image_search_dirs=image_search_dirs,
    )


def convert_to_wds(cfg: PackedWDSConfig):
    """ Convert dataset to wds format """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    tar_pattern = cfg.output_dir / "pretrain-%06d.tar"
    if cfg.mode != "caption_pack":
        raise ValueError(f"Unsupported mode '{cfg.mode}' in config-driven workflow.")

    if cfg.sample_type == "packed_multi_mix_qa":
        stream_fn = stream_samples_packed_multi_mix_qa
        construct_fn = construct_sample_packed_multi_mix_qa
    else:
        stream_fn = stream_samples_caption
        construct_fn = lambda entry, roots: construct_sample(entry, roots)

    with wds.ShardWriter(str(tar_pattern), maxcount=cfg.maxcount, maxsize=cfg.maxsize) as sink:
        for idx, entry in enumerate(tqdm(stream_fn(cfg.json_dir))):
            sample = construct_fn(entry, cfg.image_search_dirs)
            _log_json_presence(entry.get("id", "unknown"), sample, idx)
            sink.write(sample)

    write_config(
        EPath(cfg.output_dir).absolute(),
        cfg.media,
        sample_type=cfg.sample_type,
    )
    print("Dataset successfully converted to wds")


def _safe_target_name(part: str, sample_index: int, used_names: set) -> str:
    name = Path(part).name.replace(" ", "_")
    if not name:
        name = f"media_{sample_index:03d}.bin"
    target = f"s{sample_index:03d}_{name}"
    if target not in used_names:
        used_names.add(target)
        return target
    stem = Path(target).stem
    suffix = Path(target).suffix
    counter = 1
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def construct_native_packed_sample(plan: dict, samples: dict, members: dict, reader: ShardReaderCache):
    media_type = plan["media_type"]
    sample_ids = plan["sample_ids"]
    packed = {"__key__": plan["pack_id"], "__restore_key__": plan["pack_id"]}
    prompts = []
    captions = []
    token_lens = []
    packed_media_files = []
    used_names = set()

    for sample_index, sample_id in enumerate(sample_ids):
        if sample_id not in samples:
            raise KeyError(f"Sample {sample_id} not found in manifest")
        sample = samples[sample_id]
        if sample.media_type != media_type:
            raise ValueError(
                f"Pack {plan['pack_id']} mixes media types: "
                f"{sample_id} is {sample.media_type}, expected {media_type}"
            )

        prompts.append(sample.prompt)
        captions.append(sample.caption)
        token_lens.append(sample.token_len)

        if media_type == "text":
            packed_media_files.append([])
            continue

        sample_members = members.get(sample_id, {})
        current_media = []
        for part in sample.media_files:
            if part not in sample_members:
                raise KeyError(f"Member {part} for sample {sample_id} missing from manifest")
            member = sample_members[part]
            target_name = _safe_target_name(part, sample_index, used_names)
            packed[target_name] = reader.read(
                sample.shard,
                member.offset_data,
                member.size,
            )
            current_media.append(target_name)
        packed_media_files.append(current_media)

    packed["json"] = json.dumps(
        {
            "texts": {"prompts": prompts, "captions": captions},
            "media_files": packed_media_files,
            "media_type": media_type,
            "_meta": {
                "pack_id": plan["pack_id"],
                "sample_ids": sample_ids,
                "token_lens": token_lens,
                "total_token_len": sum(token_lens),
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    return packed


def convert_native_to_wds(cfg: dict):
    data_cfg = cfg.get("data", {})
    output_dir = Path(data_cfg["packed_wds_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("pretrain-*.tar", "pretrain-*.tar.idx"):
        for stale_path in output_dir.glob(pattern):
            stale_path.unlink()
    meta_dir = output_dir / MAIN_FOLDER_NAME
    if meta_dir.exists():
        shutil.rmtree(meta_dir)

    manifest_sqlite = get_manifest_sqlite_path(cfg)
    pack_plan = get_pack_plan_path(cfg)
    if not manifest_sqlite.exists():
        raise FileNotFoundError(f"Manifest sqlite not found: {manifest_sqlite}")
    if not pack_plan.exists():
        raise FileNotFoundError(f"Pack plan not found: {pack_plan}")

    samples, members = load_manifest_for_packing(manifest_sqlite)
    maxcount = int(cfg.get("packed_wds", {}).get("maxcount", 1000000))
    maxsize = int(cfg.get("packed_wds", {}).get("maxsize", 100000000))
    tar_pattern = output_dir / "pretrain-%06d.tar"

    source_root = Path(data_cfg["wds_dir"])
    with ShardReaderCache(source_root) as reader, \
        wds.ShardWriter(str(tar_pattern), maxcount=maxcount, maxsize=maxsize) as sink, \
        pack_plan.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="write packed wds", unit="pack")):
            if not line.strip():
                continue
            plan = json.loads(line)
            sample = construct_native_packed_sample(plan, samples, members, reader)
            _log_json_presence(plan["pack_id"], sample, idx)
            sink.write(sample)

    write_config(
        EPath(output_dir).absolute(),
        sample_type=cfg.get("sample", {}).get("sample_type", "packed_multi_mix_qa"),
    )
    print("WDS-native packed dataset successfully converted to wds")

def write_config(path: EPath, media=None, sample_type=None):
    (path / MAIN_FOLDER_NAME).mkdir(exist_ok=True)
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]

    dataset_definition = {
        "__module__": "megatron.energon",
        "__class__": "CrudeWebdataset",
        "subflavors": {
            "sample_type": sample_type
        }
    }
    with (path / MAIN_FOLDER_NAME / "dataset.yaml").open("w") as f:
        yaml.dump(dataset_definition, f, sort_keys=False)

    BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=[("train", 1.0), ("val", 0), ("test", 0)],
        tar_index_only=False,
        workers=32,
    )


def main():
    """main function"""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    cfg = get_cfg(args.config)
    if cfg.get("data", {}).get("input_format", "wds") == "wds":
        convert_native_to_wds(cfg)
    else:
        runtime_cfg = build_runtime_config(cfg)
        convert_to_wds(runtime_cfg)


if __name__ == '__main__':
    main()
