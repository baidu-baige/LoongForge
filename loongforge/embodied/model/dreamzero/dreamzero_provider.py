# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero model provider.

This provider builds the four DreamZero submodules directly from
``DreamZeroConfig`` backbone presets and explicit local checkpoint paths.

Build order:
1. ``WanTextEncoder`` (T5; frozen).
2. ``WanImageEncoder`` (CLIP-XLM-Roberta-ViT-H/14; frozen).
3. ``WanVideoVAE`` or ``WanVideoVAE38`` (chosen by ``vae_class`` in preset; frozen).
4. ``CausalWanModel`` (the trainable backbone).

Pretrained weights are loaded only from explicit local paths; there is no
``huggingface_hub.hf_hub_download`` fallback. This keeps provider behaviour
deterministic and offline-safe.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
import glob
import json
import logging
import os

import torch

from .action_state_checkpoint import (
    ACTION_STATE_TARGETS as _ACTION_STATE_TARGETS,
    candidate_action_state_files,
)
from .model_configuration_dreamzero import (
    DreamZeroConfig,
    _BACKBONE_PRESETS,
    dreamzero_dit_performance_options,
)
from .modeling_dreamzero import DreamZeroPolicy
from .modules.causal_wan_model import CausalWanModel
from .modules.wan_video_image_encoder import WanImageEncoder
from .modules.wan_video_text_encoder import WanTextEncoder
from .modules.wan_video_vae import WanVideoVAE, WanVideoVAE38

logger = logging.getLogger(__name__)


def _rank0() -> bool:
    return int(os.environ.get("RANK", "0") or 0) == 0


@contextmanager
def _skip_default_reset_parameters(enabled: bool):
    """Skip expensive default init while constructing checkpoint-covered modules."""
    if not enabled:
        yield
        return

    classes = (
        torch.nn.Linear,
        torch.nn.Embedding,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
    )
    originals = {
        cls: cls.reset_parameters
        for cls in classes
        if hasattr(cls, "reset_parameters")
    }

    def _noop_reset(self):
        return None

    try:
        for cls in originals:
            cls.reset_parameters = _noop_reset
        yield
    finally:
        for cls, reset_parameters in originals.items():
            cls.reset_parameters = reset_parameters


def _dit_checkpoint_coverage_is_complete(
    model_config: DreamZeroConfig,
    full_init_path: str | None = None,
    action_state_path: str | None = None,
) -> bool:
    if (full_init_path or model_config.dit_init_checkpoint_path or "").strip():
        return True
    action_state_path = action_state_path or model_config.action_state_init_checkpoint_path
    return bool(
        model_config.backbone_variant == "wan21_14b"
        and (action_state_path or "").strip()
    )


def _should_skip_default_reset(
    model_config: DreamZeroConfig,
    checkpoint_path: str | None,
) -> bool:
    return bool(
        model_config.skip_init_weights
        and (checkpoint_path or "").strip()
    )


def _should_skip_dit_default_init(
    model_config: DreamZeroConfig,
    full_init_path: str | None,
    action_state_path: str | None,
) -> bool:
    """Return whether DiT construction can skip DreamZero's explicit init."""
    return bool(
        model_config.skip_init_weights
        and _dit_checkpoint_coverage_is_complete(
            model_config,
            full_init_path=full_init_path,
            action_state_path=action_state_path,
        )
    )


def _raise_if_missing_after_skipped_reset(
    *,
    component: str,
    checkpoint_path: str,
    missing: list[str] | tuple[str, ...] | set[str],
) -> None:
    if not missing:
        return
    missing = sorted(str(item) for item in missing)
    raise RuntimeError(
        f"DreamZero skipped default reset while building {component}, but "
        f"checkpoint {checkpoint_path} did not cover all tensors: "
        f"missing_count={len(missing)} first_missing={missing[:10]}"
    )


def _coerce_dreamzero_config(config) -> DreamZeroConfig:
    """Coerce registry/provider inputs into a concrete DreamZeroConfig."""
    if isinstance(config, DreamZeroConfig):
        return config

    valid_fields = {field.name for field in fields(DreamZeroConfig)}
    raw = config
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            try:
                raw = OmegaConf.to_container(config, resolve=True)
            except Exception:
                raw = OmegaConf.to_container(config, resolve=False)
    except Exception:
        pass

    if isinstance(raw, dict):
        values = {key: value for key, value in raw.items() if key in valid_fields}
    else:
        raw_values = vars(raw)
        values = {
            key: raw_values[key]
            for key in valid_fields
            if key in raw_values
        }
    return DreamZeroConfig(**values)


def _loadable_model_state_keys(model: torch.nn.Module) -> set[str]:
    """Return checkpoint-backed keys, excluding empty TE compatibility state."""
    return {
        key
        for key in model.state_dict().keys()
        if not key.endswith("._extra_state")
    }


def _maybe_load_full_dit_from_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str | None = None,
) -> set[str]:
    """Optionally load the full DiT from a DreamZero checkpoint."""
    ckpt_path = (ckpt_path or "").strip()
    if not ckpt_path:
        return set()

    from safetensors import safe_open  # local import: optional dep

    target_keys = _loadable_model_state_keys(model)
    source_to_target = {f"action_head.model.{target_name}": target_name for target_name in target_keys}
    source_keys = set(source_to_target)
    strict = True

    if os.path.isfile(ckpt_path):
        file_to_sources = {ckpt_path: sorted(source_keys)}
    elif os.path.isdir(ckpt_path):
        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                weight_map = json.load(f).get("weight_map", {})
            file_to_sources: dict[str, list[str]] = {}
            for source_key in sorted(source_keys):
                shard_name = weight_map.get(source_key)
                if shard_name is None:
                    continue
                file_to_sources.setdefault(os.path.join(ckpt_path, shard_name), []).append(source_key)
        else:
            files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
            file_to_sources = {fpath: sorted(source_keys) for fpath in files}
        if not file_to_sources:
            raise FileNotFoundError(f"no DiT safetensors shards found under dit_init_checkpoint_path={ckpt_path}")
    else:
        raise FileNotFoundError(f"dit_init_checkpoint_path does not exist: {ckpt_path}")

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded_targets: set[str] = set()
    missing_source_keys = set(source_keys)
    shape_mismatches: list[str] = []
    missing_targets: list[str] = []

    def _copy_tensor(target_name: str, tensor: torch.Tensor, source_key: str) -> bool:
        param = params.get(target_name)
        buf = buffers.get(target_name)
        target = param if param is not None else buf
        if target is None:
            missing_targets.append(target_name)
            return False
        if target.shape != tensor.shape:
            shape_mismatches.append(
                f"{source_key} -> {target_name}: {tuple(tensor.shape)} vs {tuple(target.shape)}"
            )
            return False
        with torch.no_grad():
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        loaded_targets.add(target_name)
        missing_source_keys.discard(source_key)
        return True

    for fpath, expected_sources in sorted(file_to_sources.items()):
        if not os.path.exists(fpath):
            continue
        with safe_open(fpath, framework="pt", device="cpu") as f:
            available = set(f.keys())
            for source_key in expected_sources:
                if source_key not in available:
                    continue
                target_name = source_to_target[source_key]
                if target_name in loaded_targets:
                    continue
                _copy_tensor(target_name, f.get_tensor(source_key), source_key)

    unloaded_targets = sorted(target_keys - loaded_targets)
    problems = []
    if missing_source_keys:
        problems.append(f"missing source keys={sorted(missing_source_keys)[:10]} count={len(missing_source_keys)}")
    if missing_targets:
        problems.append(f"missing target attrs={sorted(set(missing_targets))[:10]} count={len(set(missing_targets))}")
    if shape_mismatches:
        problems.append(f"shape mismatches={shape_mismatches[:10]} count={len(shape_mismatches)}")

    if problems and strict:
        raise RuntimeError(
            "failed to load complete DreamZero DiT init from "
            f"{ckpt_path}: {'; '.join(problems)}"
        )

    if _rank0():
        logger.info(
            "[dreamzero] full DiT init loaded %s/%s tensors from %s; unloaded_count=%s",
            len(loaded_targets),
            len(target_keys),
            ckpt_path,
            len(unloaded_targets),
        )
        if unloaded_targets:
            groups: dict[str, int] = {}
            for key in unloaded_targets:
                group = key.split(".", 1)[0]
                groups[group] = groups.get(group, 0) + 1
            logger.warning(
                "[dreamzero] full DiT init unloaded groups=%s; first 10: %s",
                groups,
                unloaded_targets[:10],
            )
        if problems:
            logger.warning("[dreamzero] full DiT init warnings: %s", "; ".join(problems))
    return loaded_targets


def _maybe_load_action_state_from_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str | None = None,
) -> set[str]:
    """Optionally seed action/state modules from a DreamZero checkpoint.

    The Wan DiT release does not include the action/state encoder/decoder
    parameters, while DreamZero training checkpoints store them under
    ``action_head.model.*``.
    """
    ckpt_path = (ckpt_path or "").strip()
    if not ckpt_path:
        return set()

    from safetensors import safe_open  # local import: optional dep

    source_to_target = {
        f"action_head.model.{target_name}": target_name for target_name in _ACTION_STATE_TARGETS
    }
    source_keys = set(source_to_target)
    strict = True

    try:
        candidates = candidate_action_state_files(ckpt_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"action_state_init_checkpoint_path does not exist: {ckpt_path}"
        ) from None

    file_to_sources: dict[str, list[str]] = {}
    for source_key, files in candidates.items():
        for fpath in files:
            file_to_sources.setdefault(str(fpath), []).append(source_key)
    if not file_to_sources:
        raise FileNotFoundError(
            "no action/state safetensors shards found under "
            f"action_state_init_checkpoint_path={ckpt_path}"
        )

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded_targets: set[str] = set()
    missing_source_keys = set(source_keys)
    shape_mismatches: list[str] = []
    missing_targets: list[str] = []

    def _copy_tensor(target_name: str, tensor: torch.Tensor, source_key: str) -> bool:
        param = params.get(target_name)
        buf = buffers.get(target_name)
        target = param if param is not None else buf
        if target is None:
            missing_targets.append(target_name)
            return False
        if target.shape != tensor.shape:
            shape_mismatches.append(
                f"{source_key} -> {target_name}: {tuple(tensor.shape)} vs {tuple(target.shape)}"
            )
            return False
        with torch.no_grad():
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        loaded_targets.add(target_name)
        missing_source_keys.discard(source_key)
        return True

    for fpath, expected_sources in sorted(file_to_sources.items()):
        if not os.path.exists(fpath):
            continue
        with safe_open(fpath, framework="pt", device="cpu") as f:
            available = set(f.keys())
            for source_key in expected_sources:
                if source_key not in available:
                    continue
                target_name = source_to_target[source_key]
                if target_name in loaded_targets:
                    continue
                _copy_tensor(target_name, f.get_tensor(source_key), source_key)

    unloaded_targets = sorted(set(_ACTION_STATE_TARGETS) - loaded_targets)
    problems = []
    if missing_source_keys:
        problems.append(f"missing source keys={sorted(missing_source_keys)}")
    if missing_targets:
        problems.append(f"missing target attrs={sorted(set(missing_targets))}")
    if shape_mismatches:
        problems.append(f"shape mismatches={shape_mismatches}")

    if problems and strict:
        raise RuntimeError(
            "failed to load complete DreamZero action/state init from "
            f"{ckpt_path}: {'; '.join(problems)}"
        )

    if _rank0():
        logger.info(
            "[dreamzero] action/state init loaded %s/%s tensors from %s; unloaded=%s",
            len(loaded_targets),
            len(_ACTION_STATE_TARGETS),
            ckpt_path,
            unloaded_targets,
        )
        if problems:
            logger.warning("[dreamzero] action/state init warnings: %s", "; ".join(problems))
    return loaded_targets

# DiT key rename table: diffusers (HF Wan release) -> CausalWanModel naming.
# Kept in sync with
# tools/convert_checkpoint/dreamzero/convert_hf_to_torch.py::_DIT_RENAME_TEMPLATE.
_DIT_RENAME_TEMPLATE = {
    "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
    "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
    "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
    "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
    "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
    "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
    "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
    "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
    "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
    "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
    "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
    "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
    "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
    "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
    "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
    "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
    "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
    "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
    "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
    "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
    "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
    "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
    "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
    "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
    "blocks.0.norm2.bias": "blocks.0.norm3.bias",
    "blocks.0.norm2.weight": "blocks.0.norm3.weight",
    "blocks.0.scale_shift_table": "blocks.0.modulation",
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "patch_embedding.weight": "patch_embedding.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
}


def _diffusers_to_civitai(name: str) -> str | None:
    """Return civitai-naming form of a diffusers DiT key, or None if no match."""
    if name in _DIT_RENAME_TEMPLATE:
        return _DIT_RENAME_TEMPLATE[name]
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "blocks":
        templated = ".".join(parts[:1] + ["0"] + parts[2:])
        if templated in _DIT_RENAME_TEMPLATE:
            renamed = _DIT_RENAME_TEMPLATE[templated]
            renamed_parts = renamed.split(".")
            return ".".join(renamed_parts[:1] + [parts[1]] + renamed_parts[2:])
    return None


def _load_dit_pretrained(model: torch.nn.Module, path: str) -> set[str]:
    """Load DiT weights into ``CausalWanModel`` from a sharded HF Wan release.

    Streams each shard with ``safe_open`` and immediately materializes tensors
    into the target model, never holding all shards in memory at once. Supports
    both diffusers and CausalWanModel naming, detected per key.
    """
    from safetensors import safe_open  # local import: optional dep

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        index_path = os.path.join(path, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            shard_set = sorted(set(index["weight_map"].values()))
            files = [os.path.join(path, s) for s in shard_set]
        else:
            files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no .safetensors shards under {path}")
    else:
        raise FileNotFoundError(f"dit_pretrained_path does not exist: {path}")

    target_keys = _loadable_model_state_keys(model)
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded_targets: set[str] = set()
    loaded = 0
    skipped: list[str] = []
    rank = int(os.environ.get("RANK", "0"))

    def _copy_tensor(target_name: str, tensor: torch.Tensor, source_key: str) -> bool:
        nonlocal loaded
        with torch.no_grad():
            param = params.get(target_name)
            buf = buffers.get(target_name)
            target = param if param is not None else buf
            if target is None:
                skipped.append(source_key)
                return False
            if target.shape != tensor.shape:
                skipped.append(
                    f"{source_key} (shape mismatch {tuple(tensor.shape)} vs {tuple(target.shape)})"
                )
                return False
            target.copy_(tensor.to(target.dtype))
            loaded_targets.add(target_name)
            loaded += 1
            return True

    for fpath in files:
        with safe_open(fpath, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                # Try direct civitai naming first; fall back to diffusers->civitai.
                if key in target_keys:
                    target_name = key
                else:
                    target_name = _diffusers_to_civitai(key)
                if target_name is not None and target_name in target_keys:
                    _copy_tensor(target_name, f.get_tensor(key), key)
                    continue
                skipped.append(key)

    if rank == 0:
        n_target = len(target_keys)
        missing_targets = sorted(target_keys - loaded_targets)
        logger.info("[dreamzero] DiT loaded %s/%s tensors from %s", loaded, n_target, path)
        if missing_targets:
            groups: dict[str, int] = {}
            for key in missing_targets:
                group = key.split(".", 1)[0]
                groups[group] = groups.get(group, 0) + 1
            logger.warning(
                "[dreamzero] DiT missing target tensors=%s groups=%s; first 10: %s",
                len(missing_targets),
                groups,
                missing_targets[:10],
            )
        if skipped:
            logger.warning("[dreamzero] DiT skipped %s keys; first 5: %s", len(skipped), skipped[:5])
    return loaded_targets


def _load_vae_pretrained(
    vae: torch.nn.Module,
    path: str,
    *,
    require_complete: bool = False,
) -> None:
    """Load Wan VAE weights into the wrapper's inner ``.model`` module."""
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"vae_pretrained_path does not exist: {path}")
    state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    target = getattr(vae, "model", vae)
    missing, unexpected = target.load_state_dict(state_dict, strict=False)
    if require_complete:
        _raise_if_missing_after_skipped_reset(
            component="VAE",
            checkpoint_path=path,
            missing=missing,
        )
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        logger.info(
            "[dreamzero] VAE loaded from %s; missing=%s unexpected=%s",
            path,
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.warning("[dreamzero] VAE missing first 5: %s", list(missing)[:5])
        if unexpected:
            logger.warning("[dreamzero] VAE unexpected first 5: %s", list(unexpected)[:5])


def _resolve_weight_path(path: str, default_filename: str, field_name: str) -> str:
    """Resolve file-or-directory pretrained paths."""
    if not path:
        return ""
    resolved = os.path.join(path, default_filename) if os.path.isdir(path) else path
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"{field_name} does not exist: {resolved}")
    return resolved


def _load_text_encoder_pretrained(
    text_encoder: torch.nn.Module,
    path: str,
    *,
    require_complete: bool = False,
) -> None:
    """Load frozen Wan T5 weights."""
    if not path:
        return
    resolved = _resolve_weight_path(
        path,
        "models_t5_umt5-xxl-enc-bf16.pth",
        "text_encoder_pretrained_path",
    )
    state_dict = torch.load(resolved, map_location="cpu")
    missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
    if require_complete:
        _raise_if_missing_after_skipped_reset(
            component="text encoder",
            checkpoint_path=resolved,
            missing=missing,
        )
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        logger.info(
            "[dreamzero] Text encoder loaded from %s; missing=%s unexpected=%s",
            resolved,
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.warning("[dreamzero] Text encoder missing first 5: %s", list(missing)[:5])
        if unexpected:
            logger.warning("[dreamzero] Text encoder unexpected first 5: %s", list(unexpected)[:5])


def _load_image_encoder_pretrained(
    image_encoder: torch.nn.Module,
    path: str,
    *,
    require_complete: bool = False,
) -> None:
    """Load frozen Wan CLIP weights into WanImageEncoder.model."""
    if not path:
        return
    resolved = _resolve_weight_path(
        path,
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "image_encoder_pretrained_path",
    )
    state_dict = torch.load(resolved, map_location="cpu")
    target = getattr(image_encoder, "model", image_encoder)
    missing, unexpected = target.load_state_dict(state_dict, strict=False)
    if require_complete:
        _raise_if_missing_after_skipped_reset(
            component="image encoder",
            checkpoint_path=resolved,
            missing=missing,
        )
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        logger.info(
            "[dreamzero] Image encoder loaded from %s; missing=%s unexpected=%s",
            resolved,
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.warning("[dreamzero] Image encoder missing first 5: %s", list(missing)[:5])
        if unexpected:
            logger.warning("[dreamzero] Image encoder unexpected first 5: %s", list(unexpected)[:5])


def _build_text_encoder(model_config: DreamZeroConfig) -> torch.nn.Module:
    """Construct ``WanTextEncoder`` (UMT5-XXL by default)."""
    pretrained_path = model_config.text_encoder_pretrained_path or ""
    skip_default_reset = _should_skip_default_reset(model_config, pretrained_path)
    with _skip_default_reset_parameters(skip_default_reset):
        text_encoder = WanTextEncoder(
            text_encoder_pretrained_path=pretrained_path or None,
            skip_init_weights=skip_default_reset,
        )
    _load_text_encoder_pretrained(
        text_encoder,
        pretrained_path,
        require_complete=skip_default_reset,
    )
    return text_encoder


def _build_image_encoder(model_config: DreamZeroConfig) -> torch.nn.Module:
    """Construct ``WanImageEncoder`` (CLIP-XLM-Roberta-ViT-H/14)."""
    pretrained_path = model_config.image_encoder_pretrained_path or ""
    skip_default_reset = _should_skip_default_reset(model_config, pretrained_path)
    with _skip_default_reset_parameters(skip_default_reset):
        image_encoder = WanImageEncoder(
            image_encoder_pretrained_path=pretrained_path or None,
        )
    _load_image_encoder_pretrained(
        image_encoder,
        pretrained_path,
        require_complete=skip_default_reset,
    )
    return image_encoder


def _build_vae(model_config: DreamZeroConfig) -> torch.nn.Module:
    """Construct VAE matching the backbone preset.

    - ``wan21_14b`` → ``WanVideoVAE`` (z_dim=16).
    - ``wan22_5b``  → ``WanVideoVAE38`` (z_dim=48, dim=160).
    """
    preset = _BACKBONE_PRESETS[model_config.backbone_variant]
    pretrained_path = model_config.vae_pretrained_path or ""
    skip_default_reset = _should_skip_default_reset(model_config, pretrained_path)
    with _skip_default_reset_parameters(skip_default_reset):
        if preset["vae_class"] == "WanVideoVAE38":
            vae = WanVideoVAE38(
                z_dim=preset["vae_z_dim"],
                dim=preset["vae_dim"],
                vae_pretrained_path=pretrained_path or None,
            )
        else:
            vae = WanVideoVAE(
                z_dim=preset["vae_z_dim"],
                vae_pretrained_path=pretrained_path or None,
            )
    setattr(vae, "batch_encode", bool(model_config.batch_vae_encode))
    _load_vae_pretrained(
        vae,
        pretrained_path,
        require_complete=skip_default_reset,
    )
    return vae


def _build_diffusion_model(model_config: DreamZeroConfig) -> torch.nn.Module:
    """Construct the ``CausalWanModel`` backbone from preset hyperparameters.

    Loads pretrained DiT weights from ``model_config.dit_pretrained_path`` if set
    (sharded safetensors directory or single file). Without it, the DiT keeps
    normal random initialization.
    """
    preset = dict(_BACKBONE_PRESETS[model_config.backbone_variant])
    dit_pretrained_path = model_config.dit_pretrained_path or ""
    full_init_path = model_config.dit_init_checkpoint_path or ""
    action_state_path = model_config.action_state_init_checkpoint_path or ""
    skip_default_reset = _should_skip_dit_default_init(
        model_config,
        full_init_path,
        action_state_path,
    )
    with _skip_default_reset_parameters(skip_default_reset):
        model = CausalWanModel(
            model_type=preset["model_type"],
            in_dim=preset["in_dim"],
            dim=preset["dim"],
            ffn_dim=preset["ffn_dim"],
            freq_dim=preset["freq_dim"],
            out_dim=preset["out_dim"],
            num_heads=preset["num_heads"],
            num_layers=preset["num_layers"],
            frame_seqlen=preset["frame_seqlen"],
            text_len=int(model_config.text_len),
            concat_first_frame_latent=preset["concat_first_frame_latent"],
            max_chunk_size=model_config.max_chunk_size,
            num_frame_per_block=model_config.num_frame_per_block,
            num_action_per_block=model_config.num_action_per_block,
            num_state_per_block=model_config.num_state_per_block,
            action_dim=model_config.max_action_dim,
            max_state_dim=model_config.max_state_dim,
            max_num_embodiments=model_config.max_num_embodiments,
            hidden_size=model_config.dit_action_state_hidden_size,
            attention_backend=model_config.attention_backend,
            performance_options=dreamzero_dit_performance_options(model_config),
            context_parallel_size=model_config.context_parallel_size,
            skip_init_weights=skip_default_reset,
        )
    loaded_targets: set[str] = set()
    if dit_pretrained_path:
        loaded_targets.update(_load_dit_pretrained(model, dit_pretrained_path))
    loaded_targets.update(_maybe_load_full_dit_from_checkpoint(model, full_init_path))
    loaded_targets.update(
        _maybe_load_action_state_from_checkpoint(model, action_state_path)
    )
    if skip_default_reset:
        _raise_if_missing_after_skipped_reset(
            component="DiT",
            checkpoint_path=(
                f"dit_pretrained_path={dit_pretrained_path!r}, "
                f"dit_init_checkpoint_path={full_init_path!r}, "
                f"action_state_init_checkpoint_path={action_state_path!r}"
            ),
            missing=_loadable_model_state_keys(model) - loaded_targets,
        )
    return model


def dreamzero_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: int | None = None,
    config=None,
):
    """Build the DreamZero policy.

    Takes the standard pre/post/parallel args expected by the model registry
    (unused here — DreamZero is a self-contained policy with no pipeline-stage
    split for P0) plus the model config.
    """
    if config is None:
        raise ValueError("DreamZero requires an explicit config in the new embodied registry")
    model_config = _coerce_dreamzero_config(config)

    if not hasattr(model_config, "device") or model_config.device is None:
        model_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder = _build_text_encoder(model_config)
    image_encoder = _build_image_encoder(model_config)
    vae = _build_vae(model_config)
    model = _build_diffusion_model(model_config)

    policy = DreamZeroPolicy(
        config=model_config,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        vae=vae,
        model=model,
    )
    return policy
