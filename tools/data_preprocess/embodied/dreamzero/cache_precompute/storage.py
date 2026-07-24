# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Storage writers, manifests, and statistics for DreamZero cache precomputation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from loongforge.embodied.model.dreamzero.precomputed_cache.artifact import (
    TENSOR_SHARDS_FORMAT,
    sha256_file,
    tensor_to_storage_array,
)

_CSV_FIELDNAMES = [
    "index",
    "trajectory_id",
    "base_index",
    "path",
    "max_abs_diff",
    "mean_abs_diff",
    "first_frame_max_abs_diff",
    "first_frame_mean_abs_diff",
    "prompt_max_abs_diff",
    "prompt_mean_abs_diff",
    "shape",
    "dtype",
    "first_frame_shape",
    "first_frame_dtype",
    "prompt_shape",
    "prompt_dtype",
    "sum",
    "abs_sum",
    "sq_sum",
    "mean",
    "std",
    "bytes",
    "sha256",
]

_COMPARE_RESULT_KEYS = {
    "video_latents": ("max_abs_diff", "mean_abs_diff"),
    "first_frame_latents": (
        "first_frame_max_abs_diff",
        "first_frame_mean_abs_diff",
    ),
    "prompt_embs": ("prompt_max_abs_diff", "prompt_mean_abs_diff"),
}


@dataclass
class _PrecomputeStats:
    processed: int = 0
    saved: int = 0
    compared: int = 0
    skipped_partial: int = 0
    cache_files: list[dict[str, Any]] = field(default_factory=list)
    compare: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            feature: {"max_abs_diff": 0.0, "max_mean_abs_diff": 0.0}
            for feature in _COMPARE_RESULT_KEYS
        }
    )

    def record(self, result: dict[str, Any]) -> None:
        self.cache_files.append(result["cache_file"])
        self.saved += int(result["saved"])
        self.compared += int(result["compared"])
        self.processed += 1
        for feature, (max_key, mean_key) in _COMPARE_RESULT_KEYS.items():
            feature_stats = self.compare[feature]
            feature_stats["max_abs_diff"] = max(
                feature_stats["max_abs_diff"],
                float(result[max_key]),
            )
            feature_stats["max_mean_abs_diff"] = max(
                feature_stats["max_mean_abs_diff"],
                float(result[mean_key]),
            )

    def compare_summary(self, args: argparse.Namespace) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "enabled": bool(args.compare_existing),
            "compare_atol": float(args.compare_atol),
        }
        for feature, feature_stats in self.compare.items():
            summary[feature] = dict(feature_stats)
        summary["first_frame_latents"]["enabled"] = bool(
            args.include_first_frame_latents
        )
        summary["prompt_embs"]["enabled"] = bool(args.include_prompt_embs)
        return summary


def _cache_path(
    output_dir: Path, template: str, index: int, trajectory_id: int, base_index: int
) -> Path:
    rendered = Path(
        template.format(
            index=int(index),
            trajectory_id=int(trajectory_id),
            base_index=int(base_index),
        )
    )
    return rendered if rendered.is_absolute() else output_dir / rendered


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    f32 = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sum": float(f32.sum().item()),
        "abs_sum": float(f32.abs().sum().item()),
        "sq_sum": float(f32.square().sum().item()),
        "mean": float(f32.mean().item()),
        "std": float(f32.std().item()),
    }


def _load_cached_latents(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("video_latents", "latents"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        raise TypeError(
            f"cached latent payload in {path} is {type(payload)!r}, expected tensor"
        )
    return payload.detach().cpu()


def _load_cached_tensor(path: Path, keys: tuple[str, ...]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        if "video_latents" in keys or "latents" in keys:
            return payload.detach().cpu()
        raise KeyError(f"{path} stores a bare tensor and cannot satisfy keys {keys!r}")
    if not isinstance(payload, dict):
        raise TypeError(f"cached payload in {path} is {type(payload)!r}, expected dict")
    for key in keys:
        if key in payload:
            value = payload[key]
            if torch.is_tensor(value):
                return value.detach().cpu()
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
            raise TypeError(
                f"cached {key!r} in {path} is {type(value)!r}, expected tensor"
            )
    raise KeyError(f"{path} does not contain any of {keys!r}")


def _record_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    row = {key: record.get(key, "") for key in _CSV_FIELDNAMES}
    row["shape"] = (
        json.dumps(record["shape"]) if record.get("shape") is not None else ""
    )
    row["first_frame_shape"] = (
        json.dumps(record["first_frame_shape"])
        if record["first_frame_shape"] is not None
        else ""
    )
    row["prompt_shape"] = (
        json.dumps(record["prompt_shape"])
        if record.get("prompt_shape") is not None
        else ""
    )
    return row


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, path)


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        raise FileNotFoundError(f"rank manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_manifest_files(
    manifest_path: Path, csv_path: Path, records: list[dict[str, Any]]
) -> None:
    tmp_manifest_path = manifest_path.with_suffix(".jsonl.tmp")
    tmp_csv_path = csv_path.with_suffix(".csv.tmp")
    with tmp_manifest_path.open("w", encoding="utf-8") as mf:
        for record in records:
            mf.write(json.dumps(record, sort_keys=True) + "\n")
    with tmp_csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(_record_for_csv(record))
    os.replace(tmp_manifest_path, manifest_path)
    os.replace(tmp_csv_path, csv_path)


def _rank_manifest_path(output_dir: Path, rank: int, suffix: str) -> Path:
    return output_dir / f"manifest.rank{rank}.{suffix}"


def _rank_summary_path(output_dir: Path, rank: int) -> Path:
    return output_dir / f"precompute.rank{rank}.summary.json"


def _compare_tensor(
    *,
    cached: torch.Tensor,
    current: torch.Tensor,
    path: Path,
    feature: str,
    atol: float,
) -> tuple[float, float]:
    if cached.shape != current.shape:
        raise ValueError(
            f"{path} {feature} shape {tuple(cached.shape)} does not match "
            f"online {tuple(current.shape)}"
        )
    diff = (cached.to(dtype=current.dtype) - current).abs().float()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    if max_abs_diff > atol:
        raise ValueError(
            f"{path} {feature} max_abs_diff {max_abs_diff:.6e} exceeds "
            f"--compare-atol {atol:.6e}"
        )
    return max_abs_diff, mean_abs_diff


def _process_cache_sample(
    *,
    output_dir: Path,
    out_path: Path,
    dataset_index: int,
    trajectory_id: int,
    base_index: int,
    sample_latents: torch.Tensor | None,
    sample_first_frame_latents: torch.Tensor | None,
    sample_prompt_embs: torch.Tensor | None,
    overwrite: bool,
    compare_existing: bool,
    compare_atol: float,
) -> dict[str, Any]:
    compare = {key: 0.0 for keys in _COMPARE_RESULT_KEYS.values() for key in keys}
    saved = 0
    compared = 0

    if out_path.exists() and compare_existing:
        if sample_latents is not None:
            compare["max_abs_diff"], compare["mean_abs_diff"] = _compare_tensor(
                cached=_load_cached_latents(out_path),
                current=sample_latents,
                path=out_path,
                feature="video_latents",
                atol=compare_atol,
            )
        if sample_first_frame_latents is not None:
            (
                compare["first_frame_max_abs_diff"],
                compare["first_frame_mean_abs_diff"],
            ) = _compare_tensor(
                cached=_load_cached_tensor(
                    out_path,
                    ("first_frame_latents", "image_latents", "y_latents"),
                ),
                current=sample_first_frame_latents,
                path=out_path,
                feature="first_frame_latents",
                atol=compare_atol,
            )
        if sample_prompt_embs is not None:
            compare["prompt_max_abs_diff"], compare["prompt_mean_abs_diff"] = (
                _compare_tensor(
                    cached=_load_cached_tensor(
                        out_path,
                        ("prompt_embs", "prompt_embeddings", "text_embs"),
                    ),
                    current=sample_prompt_embs,
                    path=out_path,
                    feature="prompt_embs",
                    atol=compare_atol,
                )
            )
        compared = 1
    else:
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"{out_path} exists; pass --overwrite or --compare-existing"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        if sample_latents is not None:
            payload["video_latents"] = sample_latents
        if sample_first_frame_latents is not None:
            payload["first_frame_latents"] = sample_first_frame_latents
        if sample_prompt_embs is not None:
            payload["prompt_embs"] = sample_prompt_embs
        if not payload:
            raise ValueError("precomputed cache sample has no enabled feature payload")
        tmp_path = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, out_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        saved = 1

    file_bytes = int(out_path.stat().st_size)
    file_sha256 = sha256_file(out_path)

    stats = _tensor_stats(sample_latents) if sample_latents is not None else {}
    first_frame_stats = (
        _tensor_stats(sample_first_frame_latents)
        if sample_first_frame_latents is not None
        else {}
    )
    prompt_stats = (
        _tensor_stats(sample_prompt_embs) if sample_prompt_embs is not None else {}
    )
    relative_path = str(
        out_path.relative_to(output_dir)
        if out_path.is_relative_to(output_dir)
        else out_path
    )
    record = {
        "index": int(dataset_index),
        "trajectory_id": int(trajectory_id),
        "base_index": int(base_index),
        "path": relative_path,
        **compare,
        "first_frame_shape": first_frame_stats.get("shape"),
        "first_frame_dtype": first_frame_stats.get("dtype"),
        "prompt_shape": prompt_stats.get("shape"),
        "prompt_dtype": prompt_stats.get("dtype"),
        "bytes": file_bytes,
        "sha256": file_sha256,
        **stats,
    }

    cache_file = {
        "index": int(dataset_index),
        "trajectory_id": int(trajectory_id),
        "base_index": int(base_index),
        "path": relative_path,
        "bytes": file_bytes,
        "sha256": file_sha256,
    }
    return {
        "record": record,
        "cache_file": cache_file,
        "saved": saved,
        "compared": compared,
        **compare,
    }


class _TensorShardWriter:
    def __init__(
        self,
        *,
        output_dir: Path,
        shard_size: int,
        rank: int,
        overwrite: bool,
        hash_files: bool,
    ) -> None:
        self.output_dir = output_dir
        self.shard_size = int(shard_size)
        if self.shard_size <= 0:
            raise ValueError(f"tensor shard size must be positive, got {shard_size}")
        self.rank = int(rank)
        self.overwrite = bool(overwrite)
        self.hash_files = bool(hash_files)
        self.shard_root = output_dir / "tensor_shards"
        self.shard_root.mkdir(parents=True, exist_ok=True)
        self._local_shard_id = 0
        self._current_shard: dict[str, Any] | None = None
        self.shards: list[dict[str, Any]] = []

    def _new_shard_id(self) -> int:
        return self.rank * 1_000_000_000 + self._local_shard_id

    @staticmethod
    def _payload_tensors(
        sample_latents: torch.Tensor | None,
        sample_first_frame_latents: torch.Tensor | None,
        sample_prompt_embs: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        payload = {}
        if sample_latents is not None:
            payload["video_latents"] = sample_latents
        if sample_first_frame_latents is not None:
            payload["first_frame_latents"] = sample_first_frame_latents
        if sample_prompt_embs is not None:
            payload["prompt_embs"] = sample_prompt_embs
        if not payload:
            raise ValueError("tensor shard sample has no enabled feature payload")
        return payload

    def _ensure_shard(self, payload: dict[str, torch.Tensor]) -> dict[str, Any]:
        if (
            self._current_shard is not None
            and self._current_shard["count"] < self.shard_size
        ):
            return self._current_shard
        self._finalize_current_shard()

        shard_id = self._new_shard_id()
        shard_name = f"rank{self.rank:05d}_shard{self._local_shard_id:06d}"
        self._local_shard_id += 1
        shard_dir = self.shard_root / shard_name
        if shard_dir.exists() and not self.overwrite:
            raise FileExistsError(f"{shard_dir} exists; pass --overwrite")
        shard_dir.mkdir(parents=True, exist_ok=True)
        features: dict[str, Any] = {}
        for payload_key, tensor in payload.items():
            storage_array, storage_dtype = tensor_to_storage_array(tensor)
            path = shard_dir / f"{payload_key}.npy"
            mmap = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=storage_array.dtype,
                shape=(self.shard_size, *storage_array.shape),
            )
            features[payload_key] = {
                "payload_key": payload_key,
                "path": path,
                "mmap": mmap,
                "dtype": str(tensor.dtype),
                "storage_dtype": storage_dtype,
                "shape": list(tensor.shape),
                "storage_shape": list(mmap.shape),
            }
        self._current_shard = {
            "id": shard_id,
            "name": shard_name,
            "dir": shard_dir,
            "count": 0,
            "features": features,
        }
        return self._current_shard

    def _finalize_current_shard(self) -> None:
        shard = self._current_shard
        if shard is None:
            return
        count = int(shard["count"])
        features_meta: dict[str, Any] = {}
        for payload_key, feature in shard["features"].items():
            mmap = feature.pop("mmap")
            mmap.flush()
            path = feature["path"]
            relative_path = path.relative_to(self.output_dir).as_posix()
            features_meta[payload_key] = {
                "payload_key": payload_key,
                "path": relative_path,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path) if self.hash_files else "",
                "dtype": feature["dtype"],
                "storage_dtype": feature["storage_dtype"],
                "shape": feature["shape"],
                "storage_shape": feature["storage_shape"],
                "count": count,
            }
        self.shards.append(
            {
                "id": int(shard["id"]),
                "name": str(shard["name"]),
                "path": shard["dir"].relative_to(self.output_dir).as_posix(),
                "count": count,
                "features": features_meta,
            }
        )
        self._current_shard = None

    def close(self) -> dict[str, Any]:
        self._finalize_current_shard()
        return {
            "format": TENSOR_SHARDS_FORMAT,
            "shard_size": int(self.shard_size),
            "rank": int(self.rank),
            "hash_files": bool(self.hash_files),
            "shards": self.shards,
        }

    def write_sample(
        self,
        *,
        dataset_index: int,
        trajectory_id: int,
        base_index: int,
        sample_latents: torch.Tensor | None,
        sample_first_frame_latents: torch.Tensor | None,
        sample_prompt_embs: torch.Tensor | None,
    ) -> dict[str, Any]:
        payload = self._payload_tensors(
            sample_latents,
            sample_first_frame_latents,
            sample_prompt_embs,
        )
        shard = self._ensure_shard(payload)
        row_offset = int(shard["count"])
        for payload_key, tensor in payload.items():
            if payload_key not in shard["features"]:
                raise ValueError(
                    f"tensor shard feature set changed inside shard: missing {payload_key}"
                )
            feature = shard["features"][payload_key]
            if list(tensor.shape) != feature["shape"]:
                raise ValueError(
                    f"tensor shard {payload_key} shape changed: "
                    f"{list(tensor.shape)} vs {feature['shape']}"
                )
            storage_array, storage_dtype = tensor_to_storage_array(tensor)
            if storage_dtype != feature["storage_dtype"]:
                raise ValueError(
                    f"tensor shard {payload_key} storage dtype changed: "
                    f"{storage_dtype} vs {feature['storage_dtype']}"
                )
            feature["mmap"][row_offset] = storage_array
        shard["count"] = row_offset + 1

        stats = _tensor_stats(sample_latents) if sample_latents is not None else {}
        first_frame_stats = (
            _tensor_stats(sample_first_frame_latents)
            if sample_first_frame_latents is not None
            else {}
        )
        prompt_stats = (
            _tensor_stats(sample_prompt_embs) if sample_prompt_embs is not None else {}
        )
        sample_bytes = int(
            sum(tensor.numel() * tensor.element_size() for tensor in payload.values())
        )
        record = {
            "index": int(dataset_index),
            "trajectory_id": int(trajectory_id),
            "base_index": int(base_index),
            "path": f"tensor_shards/{shard['name']}:{row_offset}",
            "storage_format": TENSOR_SHARDS_FORMAT,
            "shard_id": int(shard["id"]),
            "row_offset": row_offset,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "first_frame_max_abs_diff": 0.0,
            "first_frame_mean_abs_diff": 0.0,
            "prompt_max_abs_diff": 0.0,
            "prompt_mean_abs_diff": 0.0,
            "first_frame_shape": first_frame_stats.get("shape"),
            "first_frame_dtype": first_frame_stats.get("dtype"),
            "prompt_shape": prompt_stats.get("shape"),
            "prompt_dtype": prompt_stats.get("dtype"),
            "bytes": sample_bytes,
            "sha256": "",
            **stats,
        }
        return {
            "record": record,
            "cache_file": {
                "index": int(dataset_index),
                "trajectory_id": int(trajectory_id),
                "base_index": int(base_index),
                "path": record["path"],
                "bytes": sample_bytes,
                "sha256": "",
            },
            "saved": 1,
            "compared": 0,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "first_frame_max_abs_diff": 0.0,
            "first_frame_mean_abs_diff": 0.0,
            "prompt_max_abs_diff": 0.0,
            "prompt_mean_abs_diff": 0.0,
        }


def _merge_tensor_shard_storage(
    rank_summaries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    shards: list[dict[str, Any]] = []
    shard_size = None
    hash_files = False
    for summary in rank_summaries:
        storage = summary.get("storage")
        if (
            not isinstance(storage, dict)
            or storage.get("format") != TENSOR_SHARDS_FORMAT
        ):
            continue
        shard_size = storage.get("shard_size", shard_size)
        hash_files = hash_files or bool(storage.get("hash_files", False))
        shards.extend(storage.get("shards", []) or [])
    if not shards:
        return None
    return {
        "format": TENSOR_SHARDS_FORMAT,
        "shard_size": int(shard_size or 0),
        "hash_files": bool(hash_files),
        "shards": sorted(shards, key=lambda item: int(item["id"])),
    }


def _merge_compare_summaries(
    rank_summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": bool(args.compare_existing),
        "compare_atol": float(args.compare_atol),
    }
    for feature in _COMPARE_RESULT_KEYS:
        summary[feature] = {
            key: max(float(item["compare"][feature][key]) for item in rank_summaries)
            for key in ("max_abs_diff", "max_mean_abs_diff")
        }
    summary["first_frame_latents"]["enabled"] = bool(args.include_first_frame_latents)
    summary["prompt_embs"]["enabled"] = bool(args.include_prompt_embs)
    return summary


def _write_tensor_shard_index(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    storage: dict[str, Any],
) -> dict[str, Any]:
    index_dir = output_dir / "tensor_shards" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "dataset_indices": np.asarray(
            [int(row["index"]) for row in records], dtype=np.int64
        ),
        "trajectory_ids": np.asarray(
            [int(row["trajectory_id"]) for row in records], dtype=np.int64
        ),
        "base_indices": np.asarray(
            [int(row["base_index"]) for row in records], dtype=np.int64
        ),
        "shard_ids": np.asarray(
            [int(row["shard_id"]) for row in records], dtype=np.int64
        ),
        "row_offsets": np.asarray(
            [int(row["row_offset"]) for row in records], dtype=np.int64
        ),
    }
    index_meta: dict[str, Any] = {}
    for name, array in arrays.items():
        path = index_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        index_meta[name] = {
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    storage = dict(storage)
    storage["index"] = index_meta
    return storage


def _tensor_storage_files(storage: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(storage, dict) or storage.get("format") != TENSOR_SHARDS_FORMAT:
        return []
    files: list[dict[str, Any]] = []
    for label, file_info in sorted((storage.get("index", {}) or {}).items()):
        if not isinstance(file_info, dict):
            continue
        files.append(
            {
                "kind": "index",
                "name": str(label),
                "path": str(file_info.get("path", "")),
                "bytes": int(file_info.get("bytes", 0)),
                "sha256": str(file_info.get("sha256", "")),
            }
        )
    for shard in sorted(
        storage.get("shards", []) or [], key=lambda item: int(item["id"])
    ):
        if not isinstance(shard, dict):
            continue
        shard_id = int(shard["id"])
        for feature_name, file_info in sorted(
            (shard.get("features", {}) or {}).items()
        ):
            if not isinstance(file_info, dict):
                continue
            files.append(
                {
                    "kind": "shard",
                    "shard_id": shard_id,
                    "name": str(feature_name),
                    "path": str(file_info.get("path", "")),
                    "bytes": int(file_info.get("bytes", 0)),
                    "sha256": str(file_info.get("sha256", "")),
                }
            )
    return files
