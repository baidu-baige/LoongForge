# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.

"""Precomputed VAE-latent cache mixin for ``DreamZeroLeRobotDataset``.

Handles precomputed-cache strictness/path resolution, cache config
validation against the on-disk artifact, and attaching cached
video/first-frame latent tensors onto a raw sample dict in ``__getitem__``.
Mixed into ``DreamZeroLeRobotDataset``; relies on attributes assigned in that
class's ``__init__`` (``self.precomputed_cache_config``, ``self.dataset_path``,
etc.) via normal Python attribute lookup at call time.
"""

from pathlib import Path
import logging
from typing import Any

from loongforge.embodied.model.dreamzero.precomputed_cache.artifact import (
    DreamZeroPrecomputedFeatureArtifact,
    cache_sample_path,
    extract_tensor_from_payload,
    load_cache_payload,
)

from ..transforms.video import VideoColorJitter, VideoCrop, VideoResize

logger = logging.getLogger(__name__)

_DREAMZERO_DATA_LOGGED = set()


def _dreamzero_data_log_once(key: str, message: str) -> None:
    """Log ``message`` at info level the first time ``key`` is seen, then suppress."""
    if key in _DREAMZERO_DATA_LOGGED:
        return
    _DREAMZERO_DATA_LOGGED.add(key)
    logger.info(message)


class _DreamZeroPrecomputedCacheMixin:
    """Precomputed VAE-latent cache methods for ``DreamZeroLeRobotDataset``."""

    def _precomputed_cache_strict(self) -> bool:
        """Return whether missing/invalid precomputed cache features should raise."""
        cfg = self.precomputed_cache_config
        return bool(
            cfg.strict
            or cfg.video_latents.required
            or cfg.first_frame_latents.required
            or cfg.prompt_embs.required
        )

    def _precomputed_cache_path(self, index: int, trajectory_id: int, base_index: int) -> Path | None:
        """Return the precomputed cache file path for a sample, or None if unset."""
        cfg = self.precomputed_cache_config
        cache_dir = str(cfg.cache_dir or "").strip()
        if not cache_dir:
            return None
        return cache_sample_path(
            cache_dir=cache_dir,
            cache_template=cfg.cache_template,
            index=index,
            trajectory_id=trajectory_id,
            base_index=base_index,
        )

    def _dreamzero_current_precomputed_transform_config(self) -> dict[str, Any]:
        """Summarize the active transform pipeline for precomputed-artifact validation."""
        current: dict[str, Any] = {}
        color_jitter_seen = False
        for transform in self.transforms.transforms:
            if isinstance(transform, VideoCrop):
                current["crop_scale"] = float(transform.scale)
            elif isinstance(transform, VideoResize):
                current["image_height"] = int(transform.height)
                current["image_width"] = int(transform.width)
                current["resize_interpolation"] = str(transform.interpolation)
            elif isinstance(transform, VideoColorJitter):
                color_jitter_seen = True
                current["color_jitter"] = {
                    "brightness": transform.brightness,
                    "contrast": transform.contrast,
                    "saturation": transform.saturation,
                    "hue": transform.hue,
                }

        current["enable_color_jitter"] = color_jitter_seen
        if self.max_chunk_size is not None:
            current["max_chunk_size"] = int(self.max_chunk_size)
        current["language_chunk_sampling"] = bool(self.language_chunk_sampling)
        return current

    def _maybe_validate_precomputed_artifact(self) -> DreamZeroPrecomputedFeatureArtifact | None:
        """Load and validate the precomputed feature artifact once, caching the result."""
        if self._dreamzero_precomputed_artifact_checked:
            return self._dreamzero_precomputed_artifact
        self._dreamzero_precomputed_artifact_checked = True
        cfg = self.precomputed_cache_config
        should_validate = bool(
            cfg.manifest
            or cfg.strict
            or cfg.validation.validate_artifact
            or cfg.validation.validate_file_hash
            or cfg.validation.validate_sample_hash
            or cfg.validation.require_success
        )
        if not should_validate:
            return None

        cache_dir_raw = str(cfg.cache_dir or "").strip()
        if not cache_dir_raw:
            raise ValueError(
                "precomputed cache artifact validation requires "
                "precomputed_cache.cache_dir"
            )
        artifact = DreamZeroPrecomputedFeatureArtifact.load(
            cache_dir=cache_dir_raw,
            manifest=str(cfg.manifest or "").strip() or None,
        )
        artifact.validate_for_training(
            cfg,
            dataset_len=len(self),
            dataset_path=self.dataset_path,
            current_transform_config=self._dreamzero_current_precomputed_transform_config(),
            use_sample_transform_seed=self.use_sample_transform_seed,
            sample_transform_seed=self.sample_transform_seed,
        )
        self._dreamzero_precomputed_artifact = artifact
        coverage = artifact.coverage
        _dreamzero_data_log_once(
            "precomputed_artifact_validated",
            "[dreamzero-data] validated precomputed feature artifact "
            f"{artifact.manifest_path}; coverage={coverage.get('processed')}/{coverage.get('selected')}",
        )
        return artifact

    def _maybe_validate_precomputed_sample_file(self, path: Path) -> None:
        """Validate a single precomputed sample file's hash against the manifest, if enabled."""
        cfg = self.precomputed_cache_config
        if not cfg.validation.validate_sample_hash:
            return
        artifact = self._maybe_validate_precomputed_artifact()
        if artifact is None:
            return
        found = artifact.validate_sample_file(
            path,
            strict=self._precomputed_cache_strict(),
            check_hash=True,
        )
        if not found:
            _dreamzero_data_log_once(
                "precomputed_manifest_row_missing",
                f"[dreamzero-data] DreamZero precomputed sample is absent from manifest.jsonl: {path}",
            )

    def _maybe_attach_precomputed_features(
        self,
        data: dict,
        *,
        index: int,
        trajectory_id: int,
        base_index: int,
    ) -> dict:
        """Attach precomputed video/first-frame/prompt features to ``data`` if enabled."""
        cfg = self.precomputed_cache_config
        enabled_features = (
            ("video_latents", cfg.video_latents, ("video_latents", "latents")),
            (
                "first_frame_latents",
                cfg.first_frame_latents,
                ("first_frame_latents", "image_latents", "y_latents"),
            ),
            (
                "prompt_embs",
                cfg.prompt_embs,
                ("prompt_embs", "prompt_embeddings", "text_embs"),
            ),
        )
        if not any(feature.enabled for _, feature, _ in enabled_features):
            return data
        if all((not feature.enabled) or feature.batch_key in data for _, feature, _ in enabled_features):
            return data

        artifact = self._maybe_validate_precomputed_artifact()
        if artifact is not None and artifact.storage_format == "tensor_shards":
            payload = artifact.load_tensor_shard_payload(
                index=index,
                trajectory_id=trajectory_id,
                base_index=base_index,
                strict=self._precomputed_cache_strict(),
            )
            if payload is None:
                return data
            cache_source = f"{artifact.manifest_path}#tensor_shards"
        else:
            path = self._precomputed_cache_path(index, trajectory_id, base_index)
            if path is None:
                if self._precomputed_cache_strict():
                    raise ValueError(
                        "precomputed_cache.enabled=true requires precomputed_cache.cache_dir"
                    )
                return data
            if not path.exists():
                if self._precomputed_cache_strict():
                    raise FileNotFoundError(f"missing DreamZero precomputed cache file: {path}")
                _dreamzero_data_log_once(
                    "precomputed_cache_missing",
                    f"[dreamzero-data] no precomputed cache at {path}; fallback to online features",
                )
                return data
            self._maybe_validate_precomputed_sample_file(path)
            payload = load_cache_payload(path)
            cache_source = str(path.parent)
        for feature_name, feature, fallback_keys in enabled_features:
            if not feature.enabled:
                continue
            data[feature.batch_key] = extract_tensor_from_payload(
                payload,
                Path(cache_source),
                cfg.payload_keys(feature_name, (feature.batch_key, *fallback_keys)),
            )
        _dreamzero_data_log_once(
            "precomputed_cache_enabled",
            f"[dreamzero-data] load precomputed features from {cache_source}",
        )
        return data
