# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dataset construction and frozen-feature encoding for DreamZero caches."""

from __future__ import annotations

import importlib.util
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from loongforge.embodied.data.datasets.dreamzero.dataset.datasets import (
    DreamZeroLeRobotDataset,
)
from loongforge.embodied.data.datasets.dreamzero.dataset.modality_configs import (
    EMBODIMENT_BUILDERS,
    EMBODIMENT_TAG_TO_ID,
)
from loongforge.embodied.data.datasets.dreamzero.transforms.base import (
    ComposedModalityTransform,
)
from loongforge.embodied.data.datasets.dreamzero.transforms.concat import (
    ConcatTransform,
)
from loongforge.embodied.data.datasets.dreamzero.transforms.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from loongforge.embodied.data.datasets.dreamzero.transforms.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from loongforge.embodied.data.datasets.dreamzero.transforms.dreamzero_collator import (
    DreamTransform,
    HuggingfaceTokenizer,
)
from loongforge.embodied.data.datasets.dreamzero.transforms.dreamzero_collator import (
    collate as dreamzero_collate,
)
from loongforge.embodied.model.dreamzero.dreamzero_provider import _build_text_encoder

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _load_vae_module():
    path = _REPO_ROOT / "loongforge/embodied/model/dreamzero/modules/wan_video_vae.py"
    spec = importlib.util.spec_from_file_location("_dreamzero_wan_video_vae", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load VAE module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_vae_pretrained(vae: torch.nn.Module, path: str) -> None:
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
    print(
        f"[dreamzero-cache] VAE loaded from {path}; "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


def _build_vae_for_cache(config: SimpleNamespace) -> torch.nn.Module:
    module = _load_vae_module()
    if config.vae_class == "WanVideoVAE38":
        vae = module.WanVideoVAE38(
            z_dim=int(config.vae_z_dim),
            dim=int(config.vae_dim),
            vae_pretrained_path=config.vae_pretrained_path or None,
        )
    else:
        vae = module.WanVideoVAE(
            z_dim=int(config.vae_z_dim),
            vae_pretrained_path=config.vae_pretrained_path or None,
        )
    _load_vae_pretrained(vae, config.vae_pretrained_path or "")
    return vae


def _build_text_encoder_for_cache(config: SimpleNamespace) -> torch.nn.Module:
    if not config.text_encoder_pretrained_path:
        raise ValueError(
            "DreamZero prompt precompute requires config.text_encoder_pretrained_path"
        )
    return _build_text_encoder(config)


def _default_include_first_frame_latents(config: SimpleNamespace) -> bool:
    if bool(config.backbone_concat_first_frame_latent):
        return True
    return str(config.backbone_model_type or "").strip().lower() == "i2v"


def _default_include_prompt_embs(config: SimpleNamespace) -> bool:
    return "14b" not in str(config.backbone_variant or "").strip().lower()


def _encode_prompt_embs_for_cache(
    text_encoder: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    input_ids = input_ids.to(device=device, non_blocking=True)
    attention_mask = attention_mask.to(device=device, non_blocking=True)
    with torch.no_grad():
        prompt_embs = text_encoder(input_ids, attention_mask)
    return prompt_embs.to(dtype=torch.bfloat16)


def _build_transforms(
    config: SimpleNamespace, modality_configs
) -> ComposedModalityTransform:
    video_keys = modality_configs["video"].modality_keys
    state_keys = modality_configs["state"].modality_keys
    action_keys = modality_configs["action"].modality_keys

    pipeline = [
        VideoToTensor(apply_to=video_keys),
        VideoCrop(apply_to=video_keys, scale=float(config.crop_scale)),
        VideoResize(
            apply_to=video_keys,
            height=int(config.image_height),
            width=int(config.image_width),
            interpolation="linear",
        ),
    ]
    if bool(config.enable_color_jitter):
        pipeline.append(
            VideoColorJitter(
                apply_to=video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            )
        )
    pipeline += [
        VideoToNumpy(apply_to=video_keys),
        StateActionToTensor(apply_to=state_keys + action_keys),
        StateActionTransform(
            apply_to=state_keys + action_keys,
            normalization_modes={key: "q99" for key in state_keys + action_keys},
        ),
        ConcatTransform(
            video_concat_order=video_keys,
            state_concat_order=state_keys,
            action_concat_order=action_keys,
        ),
        DreamTransform(
            default_instruction="Perform the default behavior.",
            max_state_dim=int(config.max_state_dim),
            max_action_dim=int(config.max_action_dim),
            max_length=int(config.max_text_length),
            state_horizon=int(config.state_horizon),
            action_horizon=int(config.action_horizon),
            embodiment_tag_mapping=EMBODIMENT_TAG_TO_ID,
            tokenizer_path=config.tokenizer_path,
        ),
    ]
    return ComposedModalityTransform(transforms=pipeline)


def _build_dataset(
    config: SimpleNamespace,
    data_path: Path,
    use_sample_transform_seed: bool,
    sample_transform_seed: int,
) -> DreamZeroLeRobotDataset:
    embodiment_tag = config.embodiment_tag
    language_chunk_sampling = bool(config.language_chunk_sampling)
    modality_video_frames = 25 if language_chunk_sampling else int(config.num_frames)
    modality_chunk_size = 1 if language_chunk_sampling else int(config.max_chunk_size)
    modality_configs = EMBODIMENT_BUILDERS[embodiment_tag](
        num_video_frames=modality_video_frames,
        action_horizon=int(config.action_horizon),
        state_horizon=int(config.state_horizon),
        max_chunk_size=modality_chunk_size,
    )
    transforms = _build_transforms(config, modality_configs)
    return DreamZeroLeRobotDataset(
        dataset_path=data_path,
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
        use_global_metadata=bool(config.use_global_metadata),
        metadata_version=config.metadata_version,
        video_backend=config.video_backend,
        transforms=transforms,
        max_chunk_size=int(config.max_chunk_size),
        relative_action=bool(config.relative_action),
        relative_action_keys=config.relative_action_keys,
        relative_action_per_horizon=bool(config.relative_action_per_horizon),
        language_chunk_sampling=language_chunk_sampling,
        use_sample_transform_seed=use_sample_transform_seed,
        sample_transform_seed=sample_transform_seed,
    )


def _prepare_video_tensor(
    images: torch.Tensor,
    config: SimpleNamespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    videos = images.to(device=device, non_blocking=True)
    videos = videos.permute(0, 4, 1, 2, 3).contiguous()
    if videos.dtype == torch.uint8:
        videos = videos.float() / 255.0
        bsz, channels, frames, height, width = videos.shape
        videos = videos.permute(0, 2, 1, 3, 4).reshape(
            bsz * frames, channels, height, width
        )
        videos = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(videos)
        videos = videos.reshape(bsz, frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        )
    videos = videos.to(dtype=dtype)

    target_h = config.target_video_height
    target_w = config.target_video_width
    if target_h is not None and target_w is not None:
        _, _, frames, height, width = videos.shape
        if (height, width) != (target_h, target_w):
            bsz, channels, _, _, _ = videos.shape
            videos = torch.nn.functional.interpolate(
                videos.reshape(bsz * frames, channels, height, width),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(bsz, channels, frames, target_h, target_w)
    return videos


def _prepare_videos(
    images: torch.Tensor | list[Any],
    config: SimpleNamespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(images):
        return _prepare_video_tensor(images, config, device, dtype)

    prepared = []
    for item in images:
        tensor = item if torch.is_tensor(item) else torch.from_numpy(item)
        prepared.append(
            _prepare_video_tensor(tensor.unsqueeze(0), config, device, dtype)
        )
    return torch.cat(prepared, dim=0)


class _IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        dataset_index = int(self.indices[position])
        sample = self.dataset[dataset_index]
        sample["_dataset_index"] = dataset_index
        return sample


def _rank_indices_for_sampler_order(
    indices: list[int],
    *,
    rank: int,
    world_size: int,
) -> list[int]:
    return [int(index) for index in indices[int(rank) :: int(world_size)]]


def _collate_precompute_features(
    features: list[dict[str, Any]],
    *,
    include_text: bool = False,
    tokenizer: HuggingfaceTokenizer | None = None,
    num_views: int = 3,
    embodiment_tag_mapping: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Collate tensors needed by enabled offline feature encoders."""
    batch: dict[str, Any] = {
        "images": [feature["images"] for feature in features],
        "dataset_indices": [int(feature["_dataset_index"]) for feature in features],
    }

    if not include_text or not features or "text" not in features[0]:
        return batch

    if all(torch.is_tensor(feature["text"]) for feature in features):
        batch["text"] = torch.stack([feature["text"] for feature in features], dim=0)
    else:
        if tokenizer is None:
            raise ValueError(
                "prompt precompute requires a tokenizer for string text features"
            )
        text_features = [
            {
                "text": feature["text"],
                "embodiment_id": feature["embodiment_id"],
            }
            for feature in features
        ]
        text_batch = dreamzero_collate(
            text_features,
            tokenizer,
            num_views=num_views,
            embodiment_tag_mapping=embodiment_tag_mapping,
        )
        batch["text"] = text_batch["text"]
        batch["text_attention_mask"] = text_batch["text_attention_mask"]

    if "text_attention_mask" not in batch and "text_attention_mask" in features[0]:
        batch["text_attention_mask"] = torch.stack(
            [feature["text_attention_mask"] for feature in features],
            dim=0,
        )
    return batch


def _build_precompute_loader(
    *,
    dataset: Dataset,
    indices: list[int],
    config: SimpleNamespace,
    args: Any,
    device: torch.device,
) -> DataLoader:
    tokenizer = None
    if args.include_prompt_embs:
        tokenizer = HuggingfaceTokenizer(
            name=config.tokenizer_path,
            seq_len=int(config.max_text_length),
            clean="whitespace",
        )
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": partial(
            _collate_precompute_features,
            include_text=bool(args.include_prompt_embs),
            tokenizer=tokenizer,
            num_views=3,
            embodiment_tag_mapping=EMBODIMENT_TAG_TO_ID,
        ),
        "pin_memory": bool(args.pin_memory) and device.type == "cuda",
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        if args.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return DataLoader(_IndexedSubset(dataset, indices), **loader_kwargs)
