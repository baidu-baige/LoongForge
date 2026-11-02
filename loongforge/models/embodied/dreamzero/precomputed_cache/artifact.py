# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero precomputed feature artifact helpers.

The artifact format is scoped to sample-level DreamZero frozen-condition
features: main video latents, optional first-frame latents, and optional
raw frozen-text prompt embeddings. Training config remains in
``precomputed_cache.*``; this module owns manifest parsing, file integrity
checks, and tensor payload extraction so the dataset and command-line validator
share one implementation. Prompt embeddings are raw by default because the
DreamZero prompt postprocess depends on the current batch attention lengths.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

DREAMZERO_PRECOMPUTED_FEATURES_KIND = "dreamzero_precomputed_features"
DEFAULT_CACHE_TEMPLATE = "index_{index:08d}.pt"
TENSOR_SHARDS_FORMAT = "tensor_shards"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 hex digest of a file, reading it in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    """Compute a deterministic SHA-256 hex digest of a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_file_if_present(path: Path) -> dict[str, Any] | None:
    """Return a path/size/sha256 record for an existing file, or None if missing."""
    if not path.exists() or not path.is_file():
        return None
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def dataset_content_sha256(fingerprint: dict[str, Any]) -> str:
    """Hash a fingerprint's file entries (path, bytes, sha256) into one content digest."""
    files = []
    for item in fingerprint.get("files", []):
        files.append(
            {
                "relative_path": item.get("relative_path"),
                "bytes": item.get("bytes"),
                "sha256": item.get("sha256"),
            }
        )
    return json_sha256(files)


def dataset_fingerprint(data_path: Path) -> dict[str, Any]:
    """Hash small metadata files that define sample order and coverage."""
    meta_files = [
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/step_filter.jsonl",
        "meta/modality.json",
        "meta/relative_stats_dreamzero.json",
        "meta/relative_horizon_stats_dreamzero.json",
    ]
    files = []
    for relative in meta_files:
        item = hash_file_if_present(data_path / relative)
        if item is not None:
            item["relative_path"] = relative
            files.append(item)
    path_sha256 = json_sha256(files)
    content_sha256 = dataset_content_sha256({"files": files})
    return {
        "data_path": str(data_path),
        "files": files,
        "sha256": content_sha256,
        "content_sha256": content_sha256,
        "path_sha256": path_sha256,
    }


def normalize_manifest_path(path: str | Path) -> str:
    """Normalize a manifest path to a posix-style string for stable dict lookups."""
    rendered = Path(path)
    return rendered.as_posix() if not rendered.is_absolute() else str(rendered)


def artifact_file_path(
    cache_dir: Path,
    file_info: dict[str, Any] | None,
    manifest_dir: Path | None = None,
) -> Path | None:
    """Resolve a manifest file_info entry to an absolute path under cache_dir/manifest_dir."""
    if not isinstance(file_info, dict):
        return None
    raw_path = file_info.get("path")
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if manifest_dir is not None and (manifest_dir / path).exists():
        return manifest_dir / path
    return cache_dir / path


def validate_file_info(
    path: Path,
    file_info: dict[str, Any],
    *,
    label: str,
    check_hash: bool = True,
) -> None:
    """Validate that a file exists and matches the manifest's recorded size/hash."""
    if not path.exists():
        raise FileNotFoundError(f"DreamZero precomputed cache {label} is missing: {path}")
    expected_bytes = file_info.get("bytes")
    if expected_bytes is not None and int(path.stat().st_size) != int(expected_bytes):
        raise ValueError(
            f"DreamZero precomputed cache {label} bytes mismatch: "
            f"{path} has {path.stat().st_size}, manifest expects {expected_bytes}"
        )
    expected_sha256 = file_info.get("sha256")
    if check_hash and expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError(f"DreamZero precomputed cache {label} sha256 mismatch: {path}")


def cache_sample_path(
    *,
    cache_dir: str | Path,
    cache_template: str,
    index: int,
    trajectory_id: int,
    base_index: int,
) -> Path:
    """Render the per-sample cache file path from the configured cache_template."""
    relative = Path(
        (cache_template or DEFAULT_CACHE_TEMPLATE).format(
            index=int(index),
            trajectory_id=int(trajectory_id),
            base_index=int(base_index),
        )
    )
    if relative.is_absolute():
        return relative
    return Path(cache_dir) / relative


def load_cache_payload(path: Path) -> dict[str, Any]:
    """Load a precomputed cache file (.npy/.pt) into a dict of tensors."""
    if path.suffix == ".npy":
        return {"video_latents": torch.from_numpy(np.load(path, allow_pickle=False))}
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        return {"video_latents": payload.detach().cpu()}
    if isinstance(payload, np.ndarray):
        return {"video_latents": torch.from_numpy(payload)}
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"unsupported DreamZero precomputed cache payload in {path}: {type(payload)!r}")


def extract_tensor_from_payload(
    payload: dict[str, Any],
    path: Path,
    keys: tuple[str, ...],
) -> torch.Tensor:
    """Extract the first matching tensor from payload among the given candidate keys."""
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        raise TypeError(f"{path} key {key!r} is {type(value)!r}, expected tensor or ndarray")
    raise KeyError(f"{path} does not contain any of {keys!r}")


def tensor_to_storage_array(tensor: torch.Tensor) -> tuple[np.ndarray, str]:
    """Convert a tensor to a numpy array suitable for on-disk shard storage."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy(), "uint16"
    if tensor.dtype == torch.float16:
        return tensor.numpy(), "float16"
    if tensor.dtype == torch.float32:
        return tensor.numpy(), "float32"
    raise TypeError(f"unsupported tensor shard dtype: {tensor.dtype}")


def tensor_from_storage_array(
    array: np.ndarray,
    *,
    tensor_dtype: str,
    storage_dtype: str,
) -> torch.Tensor:
    """Reconstruct the original tensor dtype from a stored numpy shard array."""
    if tensor_dtype == "torch.bfloat16":
        if storage_dtype != "uint16":
            raise TypeError(
                "bfloat16 tensor shard expects uint16 storage, "
                f"got {storage_dtype!r}"
            )
        return torch.from_numpy(np.asarray(array, dtype=np.uint16).copy()).view(torch.bfloat16)
    if tensor_dtype == "torch.float16":
        return torch.from_numpy(np.asarray(array, dtype=np.float16).copy())
    if tensor_dtype == "torch.float32":
        return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
    raise TypeError(f"unsupported tensor shard tensor dtype: {tensor_dtype!r}")


def read_manifest_jsonl_sample(
    path: Path,
    *,
    sample_count: int,
    seed: int,
) -> tuple[int, list[tuple[int, dict[str, Any]]]]:
    """Reservoir-sample up to sample_count JSONL rows, returning the total row count."""
    rng = random.Random(seed)
    samples: list[tuple[int, dict[str, Any]]] = []
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            count += 1
            if sample_count <= 0:
                continue
            if len(samples) < sample_count:
                samples.append((lineno, entry))
            else:
                replace_at = rng.randrange(count)
                if replace_at < sample_count:
                    samples[replace_at] = (lineno, entry)
    return count, samples


@dataclass
class DreamZeroPrecomputedFeatureArtifact:
    """Parsed DreamZero precomputed-feature manifest with validation helpers."""

    manifest_path: Path
    cache_dir: Path
    artifact: dict[str, Any]
    _manifest_rows_by_path: dict[str, dict[str, Any]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _tensor_shard_index: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _tensor_shard_arrays: dict[tuple[int, str], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def load(
        cls,
        *,
        cache_dir: str | Path | None = None,
        manifest: str | Path | None = None,
    ) -> "DreamZeroPrecomputedFeatureArtifact":
        """Load and parse the manifest JSON from cache_dir and/or an explicit manifest path."""
        if cache_dir is None and manifest is None:
            raise ValueError("precomputed cache artifact load requires cache_dir or manifest")
        if manifest:
            manifest_path = Path(manifest)
            if not manifest_path.is_absolute() and cache_dir is not None:
                manifest_path = Path(cache_dir) / manifest_path
        else:
            manifest_path = Path(cache_dir) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"DreamZero precomputed feature manifest is missing: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            artifact = json.load(f)
        if artifact.get("kind") != DREAMZERO_PRECOMPUTED_FEATURES_KIND:
            raise ValueError(
                "unsupported DreamZero precomputed artifact kind: "
                f"{artifact.get('kind')!r}; expected {DREAMZERO_PRECOMPUTED_FEATURES_KIND!r}"
            )
        if cache_dir is None:
            output_dir = str(artifact.get("cache", {}).get("output_dir", "")).strip()
            if not output_dir:
                raise ValueError(
                    "DreamZero precomputed artifact cache.output_dir is missing and "
                    "cache_dir was not provided"
                )
            cache_dir = output_dir
        cache_dir = Path(cache_dir)
        return cls(manifest_path=manifest_path, cache_dir=cache_dir, artifact=artifact)

    @property
    def manifest_dir(self) -> Path:
        """Directory containing the manifest file."""
        return self.manifest_path.parent

    @property
    def cache_meta(self) -> dict[str, Any]:
        """The artifact's ``cache`` metadata block, or {} if absent."""
        value = self.artifact.get("cache", {})
        return value if isinstance(value, dict) else {}

    @property
    def features_meta(self) -> dict[str, Any]:
        """The artifact's ``features`` metadata block, or {} if absent."""
        value = self.artifact.get("features", {})
        return value if isinstance(value, dict) else {}

    @property
    def files_meta(self) -> dict[str, Any]:
        """The artifact's ``files`` metadata block, or {} if absent."""
        value = self.artifact.get("files", {})
        return value if isinstance(value, dict) else {}

    @property
    def storage_meta(self) -> dict[str, Any]:
        """The artifact's ``storage`` metadata block, or {} if absent."""
        value = self.artifact.get("storage", {})
        return value if isinstance(value, dict) else {}

    @property
    def storage_format(self) -> str:
        """The artifact's storage format, defaulting to ``sample_files``."""
        return str(self.storage_meta.get("format", "sample_files") or "sample_files")

    @property
    def coverage(self) -> dict[str, Any]:
        """The artifact's ``coverage`` metadata block, or {} if absent."""
        value = self.artifact.get("coverage", {})
        return value if isinstance(value, dict) else {}

    def _feature_meta(self, name: str) -> dict[str, Any]:
        """Return the metadata block for a single named feature, or {} if absent."""
        value = self.features_meta.get(name, {})
        return value if isinstance(value, dict) else {}

    def validate_cache_template(self, expected_template: str) -> None:
        """Raise ValueError if the manifest's cache_template disagrees with the config."""
        if self.storage_format == TENSOR_SHARDS_FORMAT:
            return
        artifact_template = self.cache_meta.get("cache_template")
        if artifact_template and artifact_template != expected_template:
            raise ValueError(
                "DreamZero precomputed cache template mismatch: "
                f"config={expected_template!r}, manifest={artifact_template!r}"
            )

    def validate_feature_requirements(self, cache_config: Any) -> None:
        """Raise ValueError if a config-enabled feature is missing from the artifact."""
        for name, feature in (
            ("video_latents", cache_config.video_latents),
            ("first_frame_latents", cache_config.first_frame_latents),
            ("prompt_embs", cache_config.prompt_embs),
        ):
            if not feature.enabled:
                continue
            feature_meta = self._feature_meta(name)
            if not bool(feature_meta.get("enabled", False)):
                raise ValueError(
                    f"precomputed_cache.features.{name}.enabled=true but artifact "
                    f"does not provide {name}"
                )

        if cache_config.video_latents.enabled:
            expected_layout = str(cache_config.video_latents.layout or "bcthw").strip().lower()
            artifact_layout = str(
                self._feature_meta("video_latents").get("layout", "bcthw")
            ).lower()
            if artifact_layout != expected_layout:
                raise ValueError(
                    "DreamZero precomputed video_latents layout mismatch: "
                    f"config={expected_layout!r}, manifest={artifact_layout!r}"
                )
        if cache_config.prompt_embs.enabled:
            expected_layout = str(cache_config.prompt_embs.layout or "blc").strip().lower()
            prompt_meta = self._feature_meta("prompt_embs")
            artifact_layout = str(prompt_meta.get("layout", "blc")).strip().lower()
            if artifact_layout != expected_layout:
                raise ValueError(
                    "DreamZero precomputed prompt_embs layout mismatch: "
                    f"config={expected_layout!r}, manifest={artifact_layout!r}"
                )

    def validate_sample_transform_seed(
        self,
        *,
        expected: bool | None,
        sample_transform_seed: int | None,
        allow_nondeterministic: bool,
    ) -> None:
        """Raise ValueError if the sample-transform-seed policy mismatches the manifest."""
        if "use_sample_transform_seed" not in self.cache_meta:
            raise ValueError(
                "DreamZero precomputed artifact manifest is missing "
                "cache.use_sample_transform_seed; regenerate the cache with "
                "the current precompute tool"
            )
        use_sample_transform_seed = self.cache_meta["use_sample_transform_seed"]
        if use_sample_transform_seed is False and not allow_nondeterministic:
            raise ValueError(
                "DreamZero precomputed artifact was generated without per-sample "
                "transform RNG seeding; set "
                "precomputed_cache.validation.allow_nondeterministic_artifact=true "
                "only for exploratory runs"
            )
        if expected is None:
            return
        if bool(use_sample_transform_seed) != bool(expected):
            raise ValueError(
                "DreamZero use_sample_transform_seed mismatch: "
                f"config={bool(expected)}, manifest={bool(use_sample_transform_seed)}"
            )
        if bool(expected) and sample_transform_seed is not None:
            if "sample_transform_seed" not in self.cache_meta:
                raise ValueError(
                    "DreamZero precomputed artifact manifest is missing "
                    "cache.sample_transform_seed; regenerate the cache with "
                    "the current precompute tool"
                )
            artifact_seed = int(self.cache_meta["sample_transform_seed"])
            if int(sample_transform_seed) != artifact_seed:
                raise ValueError(
                    "DreamZero sample transform seed mismatch: "
                    f"config={int(sample_transform_seed)}, manifest={artifact_seed}"
                )

    def validate_transform_config(
        self,
        current: dict[str, Any] | None,
        *,
        require_transform_config: bool,
    ) -> None:
        """Raise ValueError if the current transform config disagrees with the artifact."""
        artifact_config = self.artifact.get("config")
        required_keys = (
            "image_height",
            "image_width",
            "crop_scale",
            "enable_color_jitter",
        )
        if not isinstance(artifact_config, dict):
            message = (
                "DreamZero precomputed artifact has no config block; cannot validate "
                "image/crop/color-jitter transform compatibility"
            )
            if require_transform_config:
                raise ValueError(message)
            return
        if current is None:
            return

        missing = [key for key in required_keys if key not in artifact_config]
        if missing and require_transform_config:
            raise ValueError(
                "DreamZero precomputed artifact config misses transform keys "
                f"{missing}; current={current}"
            )

        for key in (
            "image_height",
            "image_width",
            "enable_color_jitter",
            "max_chunk_size",
            "language_chunk_sampling",
            "resize_interpolation",
        ):
            if key not in artifact_config or key not in current:
                continue
            artifact_value = artifact_config[key]
            current_value = current[key]
            if key in {"image_height", "image_width", "max_chunk_size"}:
                artifact_value = int(artifact_value)
                current_value = int(current_value)
            elif key in {"enable_color_jitter", "language_chunk_sampling"}:
                artifact_value = bool(artifact_value)
                current_value = bool(current_value)
            else:
                artifact_value = str(artifact_value)
                current_value = str(current_value)
            if artifact_value != current_value:
                raise ValueError(
                    "DreamZero precomputed artifact config mismatch: "
                    f"{key} current={current_value!r}, manifest={artifact_value!r}"
                )

        if "crop_scale" in artifact_config and "crop_scale" in current:
            artifact_crop = float(artifact_config["crop_scale"])
            current_crop = float(current["crop_scale"])
            if abs(artifact_crop - current_crop) > 1e-8:
                raise ValueError(
                    "DreamZero precomputed artifact config mismatch: "
                    f"crop_scale current={current_crop!r}, manifest={artifact_crop!r}"
                )

        artifact_color_jitter = artifact_config.get("color_jitter")
        current_color_jitter = current.get("color_jitter")
        if isinstance(artifact_color_jitter, dict) and isinstance(current_color_jitter, dict):
            for key in ("brightness", "contrast", "saturation", "hue"):
                if key not in artifact_color_jitter or key not in current_color_jitter:
                    continue
                artifact_value = artifact_color_jitter[key]
                current_value = current_color_jitter[key]
                if artifact_value is None or current_value is None:
                    if artifact_value != current_value:
                        raise ValueError(
                            "DreamZero precomputed artifact color-jitter mismatch: "
                            f"{key} current={current_value!r}, manifest={artifact_value!r}"
                        )
                    continue
                if abs(float(artifact_value) - float(current_value)) > 1e-8:
                    raise ValueError(
                        "DreamZero precomputed artifact color-jitter mismatch: "
                        f"{key} current={current_value!r}, manifest={artifact_value!r}"
                    )

    def validate_coverage(
        self,
        *,
        dataset_len: int | None = None,
        require_full_coverage: bool = False,
    ) -> None:
        """Raise ValueError if the artifact's coverage is invalid or incomplete."""
        selected = int(self.coverage.get("selected", -1))
        processed = int(self.coverage.get("processed", -1))
        if selected <= 0 or processed <= 0 or processed > selected:
            raise ValueError(f"invalid DreamZero precomputed cache coverage: {self.coverage}")
        manifest_dataset_len = self.coverage.get("dataset_len")
        if (
            dataset_len is not None
            and manifest_dataset_len is not None
            and int(manifest_dataset_len) != int(dataset_len)
        ):
            raise ValueError(
                "DreamZero precomputed cache dataset length mismatch: "
                f"current={dataset_len}, manifest={manifest_dataset_len}"
            )
        if require_full_coverage:
            if processed != selected or (
                manifest_dataset_len is not None and selected != int(manifest_dataset_len)
            ):
                raise ValueError(
                    "DreamZero precomputed cache does not cover the full dataset: "
                    f"processed={processed}, selected={selected}, "
                    f"dataset_len={manifest_dataset_len}"
                )

    def validate_dataset_fingerprint(self, dataset_path: Path | None) -> None:
        """Raise ValueError if the dataset's content hash disagrees with the artifact."""
        if dataset_path is None:
            return
        artifact_dataset = self.artifact.get("dataset", {})
        if not isinstance(artifact_dataset, dict) or not artifact_dataset.get("files"):
            return
        artifact_sha = artifact_dataset.get("content_sha256") or dataset_content_sha256(
            artifact_dataset
        )
        current_dataset = dataset_fingerprint(dataset_path)
        current_sha = current_dataset.get("content_sha256") or current_dataset.get("sha256")
        if artifact_sha and current_sha and artifact_sha != current_sha:
            raise ValueError(
                "DreamZero precomputed cache dataset metadata mismatch: "
                f"current={current_sha}, manifest={artifact_sha}"
            )

    def validate_success(self, *, check_manifest_hash: bool = True) -> None:
        """Raise if the artifact's _SUCCESS marker is missing, not ok, or hash-mismatched."""
        success_candidates = [self.manifest_dir / "_SUCCESS", self.cache_dir / "_SUCCESS"]
        success_path = next(
            (candidate for candidate in success_candidates if candidate.exists()),
            success_candidates[0],
        )
        if not success_path.exists():
            raise FileNotFoundError(
                "DreamZero precomputed artifact _SUCCESS is missing: "
                f"checked {success_candidates}"
            )
        with success_path.open("r", encoding="utf-8") as f:
            success = json.load(f)
        if not success.get("ok", False):
            raise ValueError(f"DreamZero precomputed artifact _SUCCESS is not ok: {success_path}")
        manifest_sha256 = success.get("manifest_sha256")
        if (
            check_manifest_hash
            and manifest_sha256
            and sha256_file(self.manifest_path) != manifest_sha256
        ):
            raise ValueError(
                "DreamZero precomputed artifact _SUCCESS manifest_sha256 mismatch: "
                f"{success_path}"
            )

    def validate_artifact_files(self, *, check_hash: bool = True) -> None:
        """Validate manifest_jsonl/manifest_csv (and tensor shards, if applicable)."""
        for label in ("manifest_jsonl", "manifest_csv"):
            file_info = self.files_meta.get(label)
            file_path = artifact_file_path(self.cache_dir, file_info, self.manifest_dir)
            if file_info and file_path is not None:
                validate_file_info(file_path, file_info, label=label, check_hash=check_hash)
        if self.storage_format == TENSOR_SHARDS_FORMAT:
            self.validate_tensor_shards(check_hash=check_hash)

    def validate_tensor_shards(self, *, check_hash: bool = True) -> None:
        """Validate tensor_shards index files, shard files, and index consistency."""
        index_meta = self.storage_meta.get("index", {})
        if not isinstance(index_meta, dict):
            raise ValueError("DreamZero tensor_shards artifact storage.index is missing")
        required_index_files = (
            "dataset_indices",
            "trajectory_ids",
            "base_indices",
            "shard_ids",
            "row_offsets",
        )
        missing_index_files = [name for name in required_index_files if name not in index_meta]
        if missing_index_files:
            raise ValueError(
                "DreamZero tensor_shards artifact is missing index files: "
                f"{missing_index_files}"
            )
        for label, file_info in index_meta.items():
            if not isinstance(file_info, dict):
                continue
            file_path = artifact_file_path(self.cache_dir, file_info, self.manifest_dir)
            if file_path is not None:
                validate_file_info(
                    file_path,
                    file_info,
                    label=f"tensor_shards.index.{label}",
                    check_hash=check_hash,
                )
        shards = self.storage_meta.get("shards", []) or []
        if not shards:
            raise ValueError("DreamZero tensor_shards artifact storage.shards is empty")
        for shard in shards:
            if not isinstance(shard, dict):
                continue
            shard_id = shard.get("id", "?")
            for feature_name, file_info in (shard.get("features", {}) or {}).items():
                if not isinstance(file_info, dict):
                    continue
                file_path = artifact_file_path(self.cache_dir, file_info, self.manifest_dir)
                if file_path is not None:
                    validate_file_info(
                        file_path,
                        file_info,
                        label=f"tensor_shards.shard{shard_id}.{feature_name}",
                        check_hash=check_hash,
                    )
        shard_index = self._load_tensor_shard_index()
        lengths = {
            name: len(shard_index[name])
            for name in required_index_files
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"DreamZero tensor_shards index length mismatch: {lengths}")
        dataset_indices = shard_index["dataset_indices"]
        if len(dataset_indices) > 1 and bool(np.any(np.diff(dataset_indices) <= 0)):
            raise ValueError(
                "DreamZero tensor_shards dataset_indices index is not strictly sorted"
            )

    def manifest_jsonl_path(self) -> Path | None:
        """Resolve the manifest.jsonl file path, or None if not declared."""
        return artifact_file_path(
            self.cache_dir,
            self.files_meta.get("manifest_jsonl"),
            self.manifest_dir,
        )

    def load_manifest_rows(self, *, strict: bool = False) -> dict[str, dict[str, Any]]:
        """Load and cache manifest.jsonl rows keyed by normalized sample path."""
        if self._manifest_rows_by_path is not None:
            return self._manifest_rows_by_path
        manifest_jsonl = self.manifest_jsonl_path()
        if manifest_jsonl is None or not manifest_jsonl.exists():
            if strict:
                raise FileNotFoundError(
                    f"DreamZero precomputed manifest.jsonl is missing: {manifest_jsonl}"
                )
            self._manifest_rows_by_path = {}
            return self._manifest_rows_by_path

        rows: dict[str, dict[str, Any]] = {}
        with manifest_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_path = row.get("path")
                if row_path:
                    rows[normalize_manifest_path(row_path)] = row
        self._manifest_rows_by_path = rows
        return rows

    def validate_sample_file(
        self,
        path: Path,
        *,
        strict: bool = False,
        check_hash: bool = True,
    ) -> bool:
        """Validate a sample file's size/hash against its manifest row, if found."""
        if self.storage_format == TENSOR_SHARDS_FORMAT:
            return True
        rows = self.load_manifest_rows(strict=strict)
        keys = [normalize_manifest_path(path)]
        try:
            keys.insert(0, normalize_manifest_path(path.relative_to(self.cache_dir)))
        except ValueError:
            pass
        row = None
        for key in keys:
            row = rows.get(key)
            if row is not None:
                break
        if row is None:
            if strict:
                raise FileNotFoundError(
                    f"DreamZero precomputed sample is absent from manifest.jsonl: {path}"
                )
            return False
        validate_file_info(path, row, label="sample", check_hash=check_hash)
        return True

    def validate_for_training(
        self,
        cache_config: Any,
        *,
        dataset_len: int | None,
        dataset_path: Path | None,
        current_transform_config: dict[str, Any] | None,
        use_sample_transform_seed: bool | None,
        sample_transform_seed: int | None,
    ) -> None:
        """Run the full set of validations required before using this artifact for training."""
        self.validate_cache_template(cache_config.cache_template or DEFAULT_CACHE_TEMPLATE)
        self.validate_feature_requirements(cache_config)
        self.validate_sample_transform_seed(
            expected=use_sample_transform_seed,
            sample_transform_seed=sample_transform_seed,
            allow_nondeterministic=(
                cache_config.validation.allow_nondeterministic_artifact
            ),
        )
        self.validate_transform_config(
            current_transform_config,
            require_transform_config=cache_config.validation.require_transform_config,
        )
        self.validate_coverage(
            dataset_len=dataset_len,
            require_full_coverage=cache_config.validation.require_full_coverage,
        )
        self.validate_dataset_fingerprint(dataset_path)
        if cache_config.validation.require_success or cache_config.strict:
            self.validate_success()
        if self.storage_format == TENSOR_SHARDS_FORMAT and cache_config.validation.validate_sample_hash:
            raise ValueError(
                "precomputed_cache.validation.validate_sample_hash is not supported "
                "for tensor_shards artifacts; use validate_file_hash for shard files"
            )
        if cache_config.validation.validate_file_hash:
            self.validate_artifact_files(check_hash=True)

    def _tensor_shard_index_path(self, name: str) -> Path:
        """Resolve the path of a named tensor_shards index file from the manifest."""
        index_meta = self.storage_meta.get("index", {})
        file_info = index_meta.get(name)
        path = artifact_file_path(self.cache_dir, file_info, self.manifest_dir)
        if path is None:
            raise ValueError(f"tensor_shards index {name!r} is missing from manifest")
        return path

    def _load_tensor_shard_index(self) -> dict[str, Any]:
        """Load and cache the tensor_shards index arrays (memory-mapped) plus shard map."""
        if self._tensor_shard_index is not None:
            return self._tensor_shard_index
        if self.storage_format != TENSOR_SHARDS_FORMAT:
            raise ValueError(f"artifact storage format is {self.storage_format!r}, not tensor_shards")
        dataset_indices = np.load(
            self._tensor_shard_index_path("dataset_indices"),
            mmap_mode="r",
            allow_pickle=False,
        )
        trajectory_ids = np.load(
            self._tensor_shard_index_path("trajectory_ids"),
            mmap_mode="r",
            allow_pickle=False,
        )
        base_indices = np.load(
            self._tensor_shard_index_path("base_indices"),
            mmap_mode="r",
            allow_pickle=False,
        )
        shard_ids = np.load(
            self._tensor_shard_index_path("shard_ids"),
            mmap_mode="r",
            allow_pickle=False,
        )
        row_offsets = np.load(
            self._tensor_shard_index_path("row_offsets"),
            mmap_mode="r",
            allow_pickle=False,
        )
        shards = {
            int(shard["id"]): shard
            for shard in self.storage_meta.get("shards", []) or []
            if isinstance(shard, dict) and "id" in shard
        }
        self._tensor_shard_index = {
            "dataset_indices": dataset_indices,
            "trajectory_ids": trajectory_ids,
            "base_indices": base_indices,
            "shard_ids": shard_ids,
            "row_offsets": row_offsets,
            "shards": shards,
        }
        return self._tensor_shard_index

    def _tensor_shard_array(
        self,
        *,
        shard_id: int,
        feature_name: str,
        feature_info: dict[str, Any],
    ) -> np.ndarray:
        """Load and cache a single tensor shard feature array (memory-mapped)."""
        key = (int(shard_id), str(feature_name))
        cached = self._tensor_shard_arrays.get(key)
        if cached is not None:
            return cached
        path = artifact_file_path(self.cache_dir, feature_info, self.manifest_dir)
        if path is None:
            raise ValueError(
                f"tensor shard {shard_id} feature {feature_name!r} has no path"
            )
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        self._tensor_shard_arrays[key] = array
        return array

    def load_tensor_shard_payload(
        self,
        *,
        index: int,
        trajectory_id: int,
        base_index: int,
        strict: bool,
    ) -> dict[str, Any] | None:
        """Look up and load a sample's feature payload from tensor_shards storage."""
        shard_index = self._load_tensor_shard_index()
        dataset_indices = shard_index["dataset_indices"]
        position = int(np.searchsorted(dataset_indices, int(index), side="left"))
        if position >= len(dataset_indices) or int(dataset_indices[position]) != int(index):
            if strict:
                raise FileNotFoundError(
                    f"missing DreamZero tensor_shards cache row for index={index}"
                )
            return None
        if (
            int(shard_index["trajectory_ids"][position]) != int(trajectory_id)
            or int(shard_index["base_indices"][position]) != int(base_index)
        ):
            raise ValueError(
                "DreamZero tensor_shards index identity mismatch: "
                f"index={index} expected trajectory/base={trajectory_id}/{base_index}, "
                f"manifest has "
                f"{int(shard_index['trajectory_ids'][position])}/"
                f"{int(shard_index['base_indices'][position])}"
            )
        shard_id = int(shard_index["shard_ids"][position])
        row_offset = int(shard_index["row_offsets"][position])
        shard = shard_index["shards"].get(shard_id)
        if shard is None:
            raise ValueError(f"tensor_shards manifest has no shard id {shard_id}")

        payload: dict[str, Any] = {}
        for feature_name, feature_info in (shard.get("features", {}) or {}).items():
            if not isinstance(feature_info, dict):
                continue
            count = int(feature_info.get("count", shard.get("count", 0)))
            if row_offset < 0 or row_offset >= count:
                raise IndexError(
                    f"tensor shard row_offset {row_offset} out of range for "
                    f"shard={shard_id} feature={feature_name} count={count}"
                )
            array = self._tensor_shard_array(
                shard_id=shard_id,
                feature_name=str(feature_name),
                feature_info=feature_info,
            )
            tensor = tensor_from_storage_array(
                array[row_offset],
                tensor_dtype=str(feature_info.get("dtype", "")),
                storage_dtype=str(feature_info.get("storage_dtype", "")),
            )
            payload_key = str(feature_info.get("payload_key", feature_name))
            payload[payload_key] = tensor
        return payload
