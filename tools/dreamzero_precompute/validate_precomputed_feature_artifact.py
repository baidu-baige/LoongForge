#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Smoke validation for DreamZero precomputed feature artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loongforge.embodied.model.dreamzero.precomputed_cache.artifact import (  # noqa: E402
    TENSOR_SHARDS_FORMAT,
    DreamZeroPrecomputedFeatureArtifact,
    artifact_file_path,
    read_manifest_jsonl_sample,
    validate_file_info,
)

_FEATURE_CHECKS = {
    "video_latents": {
        "fallback_keys": ("video_latents", "latents"),
        "shape_key": "shape",
        "dtype_key": "dtype",
    },
    "first_frame_latents": {
        "fallback_keys": ("first_frame_latents", "image_latents", "y_latents"),
        "shape_key": "first_frame_shape",
        "dtype_key": "first_frame_dtype",
    },
    "prompt_embs": {
        "fallback_keys": ("prompt_embs", "prompt_embeddings", "text_embs"),
        "shape_key": "prompt_shape",
        "dtype_key": "prompt_dtype",
    },
}


def _bool_from_choice(value: str) -> bool | None:
    if value == "any":
        return None
    return value == "true"


def _feature_payload_keys(
    feature_meta: dict, fallback_keys: tuple[str, ...]
) -> tuple[str, ...]:
    raw_keys = feature_meta.get("payload_keys", ())
    if isinstance(raw_keys, str):
        payload_keys = tuple(
            item.strip() for item in raw_keys.split(",") if item.strip()
        )
    elif isinstance(raw_keys, (list, tuple)):
        payload_keys = tuple(str(item) for item in raw_keys if str(item))
    else:
        payload_keys = ()
    keys = (str(feature_meta.get("batch_key", "") or ""), *payload_keys, *fallback_keys)
    deduped = []
    for key in keys:
        if key and key not in deduped:
            deduped.append(key)
    return tuple(deduped)


def _validate_feature_metadata(artifact: DreamZeroPrecomputedFeatureArtifact) -> None:
    prompt_meta = artifact.features_meta.get("prompt_embs", {})
    if not isinstance(prompt_meta, dict) or not bool(prompt_meta.get("enabled", False)):
        return
    layout = str(prompt_meta.get("layout", "blc") or "blc").strip().lower()
    if layout != "blc":
        raise ValueError(
            f"prompt_embs layout must be 'blc', got {layout!r}; regenerate the cache"
        )


def _validate_tensor_shard_payload(
    artifact: DreamZeroPrecomputedFeatureArtifact,
    payload: dict,
    entry: dict,
    *,
    lineno: int,
) -> None:
    checked = 0
    for feature_name, spec in _FEATURE_CHECKS.items():
        feature_meta = artifact.features_meta.get(feature_name, {})
        if not isinstance(feature_meta, dict) or not bool(
            feature_meta.get("enabled", False)
        ):
            continue
        keys = _feature_payload_keys(feature_meta, spec["fallback_keys"])
        tensor = next((payload[key] for key in keys if key in payload), None)
        if tensor is None:
            raise ValueError(
                f"tensor_shards payload misses {feature_name} for manifest.jsonl "
                f"line {lineno}; expected one of {keys}"
            )
        shape_key = spec["shape_key"]
        dtype_key = spec["dtype_key"]
        if entry.get(shape_key) and list(tensor.shape) != list(entry[shape_key]):
            raise ValueError(
                f"tensor_shards {feature_name} shape mismatch on manifest.jsonl "
                f"line {lineno}: payload={list(tensor.shape)} manifest={entry[shape_key]}"
            )
        if entry.get(dtype_key) and str(tensor.dtype) != str(entry[dtype_key]):
            raise ValueError(
                f"tensor_shards {feature_name} dtype mismatch on manifest.jsonl "
                f"line {lineno}: payload={tensor.dtype} manifest={entry[dtype_key]}"
            )
        checked += 1
    if checked == 0:
        raise ValueError(
            "tensor_shards artifact has no enabled feature metadata to validate"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Path to artifact manifest.json"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Override cache tensor directory"
    )
    parser.add_argument(
        "--expect-use-sample-transform-seed",
        choices=("true", "false", "any"),
        default="true",
        help="Expected sample transform seed metadata",
    )
    parser.add_argument("--expect-sample-transform-seed", type=int, default=0)
    parser.add_argument("--require-full-coverage", action="store_true")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--check-manifest-file-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check-sample-hash", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    artifact = DreamZeroPrecomputedFeatureArtifact.load(
        cache_dir=args.cache_dir,
        manifest=args.manifest,
    )
    _validate_feature_metadata(artifact)
    artifact.validate_sample_transform_seed(
        expected=_bool_from_choice(args.expect_use_sample_transform_seed),
        sample_transform_seed=args.expect_sample_transform_seed,
        allow_nondeterministic=args.expect_use_sample_transform_seed
        in {"false", "any"},
    )
    artifact.validate_coverage(require_full_coverage=args.require_full_coverage)
    artifact.validate_success(check_manifest_hash=True)
    artifact.validate_artifact_files(check_hash=args.check_manifest_file_hash)

    manifest_jsonl_path = artifact.manifest_jsonl_path()
    if manifest_jsonl_path is None:
        raise ValueError("files.manifest_jsonl.path is missing")
    validate_file_info(
        manifest_jsonl_path,
        artifact.files_meta.get("manifest_jsonl", {}),
        label="manifest_jsonl",
        check_hash=args.check_manifest_file_hash,
    )

    row_count, samples = read_manifest_jsonl_sample(
        manifest_jsonl_path,
        sample_count=max(args.sample_count, 0),
        seed=args.sample_seed,
    )
    expected_count = artifact.files_meta.get("cache_files_count")
    if expected_count is not None and row_count != int(expected_count):
        raise ValueError(
            f"manifest.jsonl row count mismatch: rows={row_count}, manifest={expected_count}"
        )
    processed = int(artifact.coverage.get("processed", -1))
    if row_count != processed:
        raise ValueError(
            f"coverage processed count mismatch: rows={row_count}, processed={processed}"
        )

    hash_checked = 0
    for lineno, entry in samples:
        if artifact.storage_format == TENSOR_SHARDS_FORMAT:
            payload = artifact.load_tensor_shard_payload(
                index=int(entry["index"]),
                trajectory_id=int(entry["trajectory_id"]),
                base_index=int(entry["base_index"]),
                strict=True,
            )
            if not payload:
                raise ValueError(
                    f"tensor_shards payload is empty for manifest.jsonl line {lineno}"
                )
            _validate_tensor_shard_payload(artifact, payload, entry, lineno=lineno)
            continue
        sample_path = artifact_file_path(
            artifact.cache_dir,
            entry,
            None,
        )
        if sample_path is None:
            raise ValueError(f"sample path is missing in manifest.jsonl line {lineno}")
        validate_file_info(
            sample_path,
            entry,
            label=f"sample line {lineno}",
            check_hash=args.check_sample_hash,
        )
        if args.check_sample_hash and entry.get("sha256"):
            hash_checked += 1

    summary = {
        "manifest": str(artifact.manifest_path),
        "cache_dir": str(artifact.cache_dir),
        "storage_format": artifact.storage_format,
        "coverage": f"{artifact.coverage.get('processed')}/{artifact.coverage.get('selected')}",
        "rows": row_count,
        "sampled": len(samples),
        "sample_hash_checked": hash_checked,
        "storage_files": artifact.files_meta.get("storage_files_count"),
        "use_sample_transform_seed": artifact.cache_meta.get(
            "use_sample_transform_seed"
        ),
        "sample_transform_seed": artifact.cache_meta.get("sample_transform_seed"),
    }
    print(
        "[dreamzero-precomputed-cache-smoke] ok "
        + " ".join(f"{k}={v}" for k, v in summary.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
