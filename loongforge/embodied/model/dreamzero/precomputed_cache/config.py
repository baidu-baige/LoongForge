# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero precomputed cache artifact configuration helpers.

The training-facing schema is ``precomputed_cache``: one sample-level artifact
may provide video latents, first-frame latents, and raw frozen-text prompt
embeddings. Training applies DreamZero's batch-dependent prompt postprocess
when prompt embeddings are consumed. Flat ``precomputed_*`` YAML/dotlist keys
are intentionally not accepted; wrappers and configs should use the structured
schema directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .artifact import DEFAULT_CACHE_TEMPLATE


@dataclass(frozen=True)
class DreamZeroPrecomputedFeatureConfig:
    """Configuration for a single precomputed cache feature (video/frame/prompt)."""

    enabled: bool = False
    required: bool = False
    batch_key: str = ""
    payload_keys: tuple[str, ...] = field(default_factory=tuple)
    layout: str = ""


@dataclass(frozen=True)
class DreamZeroPrecomputedValidationConfig:
    """Configuration controlling precomputed cache artifact validation."""

    validate_artifact: bool = False
    validate_file_hash: bool = False
    validate_sample_hash: bool = False
    require_success: bool = False
    require_full_coverage: bool = False
    require_transform_config: bool = False
    allow_nondeterministic_artifact: bool = False


@dataclass(frozen=True)
class DreamZeroPrecomputedCacheConfig:
    """Top-level configuration for DreamZero's precomputed sample-artifact cache."""

    enabled: bool = False
    cache_dir: str = ""
    manifest: str = ""
    cache_template: str = DEFAULT_CACHE_TEMPLATE
    strict: bool = False
    video_latents: DreamZeroPrecomputedFeatureConfig = field(
        default_factory=DreamZeroPrecomputedFeatureConfig
    )
    first_frame_latents: DreamZeroPrecomputedFeatureConfig = field(
        default_factory=DreamZeroPrecomputedFeatureConfig
    )
    prompt_embs: DreamZeroPrecomputedFeatureConfig = field(
        default_factory=DreamZeroPrecomputedFeatureConfig
    )
    validation: DreamZeroPrecomputedValidationConfig = field(
        default_factory=DreamZeroPrecomputedValidationConfig
    )
    skip_pixel_preprocess: bool = False
    first_frame_only: bool = False

    def feature(self, name: str) -> DreamZeroPrecomputedFeatureConfig:
        """Return the feature config for ``name``, raising if unknown."""
        if name == "video_latents":
            return self.video_latents
        if name == "first_frame_latents":
            return self.first_frame_latents
        if name == "prompt_embs":
            return self.prompt_embs
        raise KeyError(f"unknown DreamZero precomputed cache feature: {name!r}")

    def payload_keys(self, name: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
        """Return deduped candidate payload keys for feature ``name``."""
        feature = self.feature(name)
        keys = [feature.batch_key, *feature.payload_keys, *fallback]
        deduped: list[str] = []
        for key in keys:
            if key and key not in deduped:
                deduped.append(str(key))
        return tuple(deduped)

    def to_config_dict(self) -> dict[str, Any]:
        """Serialize this config into a plain dict for storing on model_cfg."""
        features: dict[str, Any] = {}
        for name, feature in (
            ("video_latents", self.video_latents),
            ("first_frame_latents", self.first_frame_latents),
            ("prompt_embs", self.prompt_embs),
        ):
            if not feature.enabled:
                continue
            row: dict[str, Any] = {
                "enabled": feature.enabled,
                "required": feature.required,
                "batch_key": feature.batch_key,
            }
            if feature.payload_keys:
                row["payload_keys"] = list(feature.payload_keys)
            if feature.layout:
                row["layout"] = feature.layout
            features[name] = row
        return {
            "enabled": self.enabled,
            "cache_dir": self.cache_dir,
            "manifest": self.manifest,
            "cache_template": self.cache_template,
            "strict": self.strict,
            "skip_pixel_preprocess": self.skip_pixel_preprocess,
            "first_frame_only": self.first_frame_only,
            "features": features,
            "validation": {
                "validate_artifact": self.validation.validate_artifact,
                "validate_file_hash": self.validation.validate_file_hash,
                "validate_sample_hash": self.validation.validate_sample_hash,
                "require_success": self.validation.require_success,
                "require_full_coverage": self.validation.require_full_coverage,
                "require_transform_config": self.validation.require_transform_config,
                "allow_nondeterministic_artifact": (
                    self.validation.allow_nondeterministic_artifact
                ),
            },
        }


def _to_container(value: Any) -> Any:
    if value is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    cfg = _to_container(cfg)
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    if cfg is None:
        return default
    try:
        return vars(cfg).get(name, default)
    except TypeError:
        return default


def _mapping(value: Any) -> dict[str, Any]:
    value = _to_container(value)
    return dict(value) if isinstance(value, Mapping) else {}


def _pick(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _as_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    value = _to_container(value)
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return default


def _feature_mapping(features: Mapping[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        feature = features.get(name)
        if isinstance(_to_container(feature), Mapping):
            return _mapping(feature)
    return {}


def _build_feature(
    *,
    structured_enabled: bool,
    structured_feature: Mapping[str, Any],
    default_batch_key: str,
    default_payload_keys: tuple[str, ...],
    default_layout: str = "",
    required_default: bool = False,
    default_to_enabled_when_omitted: bool = False,
) -> DreamZeroPrecomputedFeatureConfig:
    feature_present = bool(structured_feature)
    if "enabled" in structured_feature:
        enabled = _as_bool(structured_feature.get("enabled"), False)
    elif "required" in structured_feature:
        enabled = _as_bool(structured_feature.get("required"), False)
    elif feature_present:
        enabled = structured_enabled
    elif default_to_enabled_when_omitted:
        enabled = structured_enabled
    else:
        enabled = False

    required = _as_bool(structured_feature.get("required"), required_default)
    batch_key = _as_str(
        _pick(structured_feature, "batch_key", "key", default=default_batch_key),
        default_batch_key,
    )
    payload_keys = _as_tuple(
        _pick(structured_feature, "payload_keys", "keys", default=None),
        default_payload_keys,
    )
    layout = _as_str(
        _pick(structured_feature, "layout", default=default_layout),
        default_layout,
    )
    return DreamZeroPrecomputedFeatureConfig(
        enabled=enabled,
        required=enabled and required,
        batch_key=batch_key or default_batch_key,
        payload_keys=payload_keys,
        layout=layout,
    )


def _default_first_frame_cache_enabled(model_cfg: Any) -> bool:
    """Infer whether first-frame latent caching should default to enabled."""
    concat_first = _cfg_get(model_cfg, "backbone_concat_first_frame_latent", None)
    if concat_first is not None:
        return _as_bool(concat_first, False)
    model_type = _as_str(_cfg_get(model_cfg, "backbone_model_type", ""), "").strip().lower()
    if model_type:
        return model_type == "i2v"
    backbone_variant = _as_str(_cfg_get(model_cfg, "backbone_variant", ""), "").strip().lower()
    return "wan21_14b" in backbone_variant


def _default_prompt_cache_enabled(model_cfg: Any) -> bool:
    """Infer whether prompt-embedding caching should default to enabled."""
    backbone_variant = _as_str(_cfg_get(model_cfg, "backbone_variant", ""), "").strip().lower()
    return "14b" not in backbone_variant


def build_precomputed_cache_config(model_cfg: Any) -> DreamZeroPrecomputedCacheConfig:
    """Build the effective DreamZero precomputed-cache config from model settings."""
    raw_cache = _mapping(_cfg_get(model_cfg, "precomputed_cache", None))
    structured_artifact_requested = any(
        key in raw_cache
        for key in (
            "cache_dir",
            "dir",
            "output_dir",
            "manifest",
            "manifest_path",
            "cache_template",
            "template",
            "features",
        )
    )
    structured_enabled = _as_bool(
        raw_cache.get("enabled"),
        default=structured_artifact_requested,
    )

    cache_dir = _as_str(
        _pick(
            raw_cache,
            "cache_dir",
            "dir",
            "output_dir",
            default="",
        )
    )
    manifest = _as_str(
        _pick(
            raw_cache,
            "manifest",
            "manifest_path",
            default="",
        )
    )
    cache_template = _as_str(
        _pick(
            raw_cache,
            "cache_template",
            "template",
            default=DEFAULT_CACHE_TEMPLATE,
        ),
        DEFAULT_CACHE_TEMPLATE,
    ) or DEFAULT_CACHE_TEMPLATE
    strict = _as_bool(raw_cache.get("strict"), False)

    validation_raw = _mapping(raw_cache.get("validation"))
    validation = DreamZeroPrecomputedValidationConfig(
        validate_artifact=_as_bool(
            _pick(
                validation_raw,
                "validate_artifact",
                default=False,
            )
        ),
        validate_file_hash=_as_bool(
            _pick(
                validation_raw,
                "validate_file_hash",
                default=False,
            )
        ),
        validate_sample_hash=_as_bool(
            _pick(
                validation_raw,
                "validate_sample_hash",
                default=False,
            )
        ),
        require_success=_as_bool(
            _pick(
                validation_raw,
                "require_success",
                "validate_success",
                default=False,
            )
        ),
        require_full_coverage=_as_bool(
            _pick(
                validation_raw,
                "require_full_coverage",
                default=False,
            )
        ),
        require_transform_config=_as_bool(
            _pick(
                validation_raw,
                "require_transform_config",
                default=False,
            )
        ),
        allow_nondeterministic_artifact=_as_bool(
            _pick(
                validation_raw,
                "allow_nondeterministic_artifact",
                default=False,
            )
        ),
    )

    skip_pixel_preprocess = _as_bool(raw_cache.get("skip_pixel_preprocess"), False)
    first_frame_only = _as_bool(raw_cache.get("first_frame_only"), False)
    if "enabled" in raw_cache and not structured_enabled:
        return DreamZeroPrecomputedCacheConfig(
            enabled=False,
            cache_dir=cache_dir,
            manifest=manifest,
            cache_template=cache_template,
            strict=strict,
            validation=validation,
            skip_pixel_preprocess=skip_pixel_preprocess,
            first_frame_only=first_frame_only,
        )

    features = _mapping(raw_cache.get("features"))
    unsupported_features = {
        key
        for key in features
        if str(key) in {"clip_features", "clip_feas", "clip"}
    }
    if unsupported_features:
        raise ValueError(
            "DreamZero precomputed CLIP cache has been removed; "
            f"unsupported precomputed_cache.features keys: {sorted(unsupported_features)}"
        )
    video_feature = _build_feature(
        structured_enabled=structured_enabled,
        structured_feature=_feature_mapping(features, "video_latents", "video", "latents"),
        default_batch_key="video_latents",
        default_payload_keys=("video_latents", "latents"),
        default_layout="bcthw",
        required_default=strict,
        default_to_enabled_when_omitted=structured_enabled and not features,
    )
    first_frame_feature = _build_feature(
        structured_enabled=structured_enabled,
        structured_feature=_feature_mapping(
            features,
            "first_frame_latents",
            "first_frame",
            "image_latents",
        ),
        default_batch_key="first_frame_latents",
        default_payload_keys=("first_frame_latents", "image_latents", "y_latents"),
        required_default=strict,
        default_to_enabled_when_omitted=(
            structured_enabled
            and not features
            and _default_first_frame_cache_enabled(model_cfg)
        ),
    )
    prompt_feature = _build_feature(
        structured_enabled=structured_enabled,
        structured_feature=_feature_mapping(
            features,
            "prompt_embs",
            "prompt_embeddings",
            "text_embs",
            "text_embeddings",
        ),
        default_batch_key="prompt_embs",
        default_payload_keys=("prompt_embs", "prompt_embeddings", "text_embs"),
        default_layout="blc",
        required_default=strict,
        default_to_enabled_when_omitted=(
            structured_enabled
            and not features
            and _default_prompt_cache_enabled(model_cfg)
        ),
    )

    enabled = (
        structured_enabled
        or video_feature.enabled
        or first_frame_feature.enabled
        or prompt_feature.enabled
    )
    if enabled and not (
        video_feature.enabled
        or first_frame_feature.enabled
        or prompt_feature.enabled
    ):
        raise ValueError(
            "precomputed_cache.enabled=true requires at least one enabled feature "
            "under precomputed_cache.features; omit the features block to use "
            "the default model-relevant feature set"
        )
    return DreamZeroPrecomputedCacheConfig(
        enabled=enabled,
        cache_dir=cache_dir,
        manifest=manifest,
        cache_template=cache_template,
        strict=strict,
        video_latents=video_feature,
        first_frame_latents=first_frame_feature,
        prompt_embs=prompt_feature,
        validation=validation,
        skip_pixel_preprocess=skip_pixel_preprocess,
        first_frame_only=first_frame_only,
    )


def apply_precomputed_cache_config(model_cfg: Any) -> DreamZeroPrecomputedCacheConfig:
    """Normalize and write the effective precomputed-cache config back to ``model_cfg``."""
    cache_cfg = build_precomputed_cache_config(model_cfg)
    try:
        setattr(model_cfg, "precomputed_cache", cache_cfg.to_config_dict())
    except Exception:
        pass
    return cache_cfg
