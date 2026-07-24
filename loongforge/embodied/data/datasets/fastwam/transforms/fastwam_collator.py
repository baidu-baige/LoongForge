# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM collator for LoongForge embodied dataloaders."""

import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from loongforge.embodied.data.datasets.transforms.collator import BasePreprocessor, PreparedBatch, register_preprocessor


def _ensure_fastwam_on_path() -> None:
    """Insert the FastWAM ``src`` directory into ``sys.path`` if present."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "FastWAM" / "src"
        if candidate.is_dir():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return


def _model_id_to_enc_id(model_id: str) -> str:
    """Convert a HuggingFace model ID to a short alphanumeric encoder identifier."""
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


@dataclass
class FastWAMPreparedBatch(PreparedBatch):
    """CPU batch consumed by ``FastWAMPolicy``."""

    video: torch.Tensor = None
    action: torch.Tensor = None
    context: torch.Tensor = None
    context_mask: torch.Tensor = None
    proprio: Optional[torch.Tensor] = None
    action_is_pad: Optional[torch.Tensor] = None
    image_is_pad: Optional[torch.Tensor] = None

    def to_sample(self) -> Dict[str, torch.Tensor]:
        """Convert the prepared batch to a plain dict suitable for the model."""
        sample = {
            "video": self.video,
            "action": self.action,
            "context": self.context,
            "context_mask": self.context_mask,
        }
        if self.proprio is not None:
            sample["proprio"] = self.proprio
        if self.action_is_pad is not None:
            sample["action_is_pad"] = self.action_is_pad
        if self.image_is_pad is not None:
            sample["image_is_pad"] = self.image_is_pad
        return sample


@register_preprocessor("fastwam")
class FastWAMPreprocessor(BasePreprocessor):
    """Collate LeRobot-style samples into upstream FastWAM sample dicts."""

    def __init__(self, model_cfg: Any, data_cfg: Any):
        """Initialize preprocessor from typed FastWAMConfig and FastWAMDataConfig."""
        self.cfg = model_cfg
        self.data_cfg = data_cfg
        self.context_len = model_cfg.tokenizer_max_len
        self.text_dim = model_cfg.video_dit_config["text_dim"]
        self.num_video_frames = data_cfg.num_video_frames
        self.model_id = model_cfg.model_id

    @classmethod
    def from_config(
        cls,
        model_cfg,
        data_cfg,
        training_args=None,
        dataset_stats=None,
        dataset=None,
    ) -> "FastWAMPreprocessor":
        """Construct a ``FastWAMPreprocessor`` from typed configs."""
        return cls(model_cfg, data_cfg)

    def __call__(self, examples: List[Dict[str, Any]]) -> FastWAMPreparedBatch:
        """Collate a list of per-sample dicts into a ``FastWAMPreparedBatch``."""
        videos = [self._build_video(ex) for ex in examples]
        actions = [self._require_action(ex) for ex in examples]
        proprio = [self._build_proprio(ex, action.shape[0]) for ex, action in zip(examples, actions)]
        prompts = [str(ex.get("prompt", "")) for ex in examples]

        # Use pre-computed context from RobotVideoDataset if available
        if "context" in examples[0] and "context_mask" in examples[0]:
            context = torch.stack([ex["context"].float() for ex in examples], dim=0)
            context_mask = torch.stack([ex["context_mask"].bool() for ex in examples], dim=0)
        else:
            context, context_mask = self._build_context(prompts)
        video = torch.stack(videos, dim=0)
        action = torch.stack(actions, dim=0)
        proprio_tensor = None
        if not all(p is None for p in proprio):
            proprio_dim = next(p.shape[1] for p in proprio if p is not None)
            proprio_tensor = torch.stack([
                p if p is not None else torch.zeros((action.shape[1], proprio_dim), dtype=torch.float32)
                for p in proprio
            ], dim=0)

        return FastWAMPreparedBatch(
            video=video,
            action=action,
            context=context,
            context_mask=context_mask,
            proprio=proprio_tensor,
            action_is_pad=self._build_action_is_pad(examples, action.shape),
            image_is_pad=torch.zeros(video.shape[0], video.shape[2], dtype=torch.bool),
        )

    def _build_video(self, example: Dict[str, Any]) -> torch.Tensor:
        """Build a ``[C, T, H, W]`` video tensor in ``[-1, 1]`` from a sample."""
        # RobotVideoDataset already returns [C, T, H, W] in [-1, 1]
        if "video" in example and example["video"] is not None:
            return example["video"].float()
        images = example.get("images") or []
        if not images:
            raise ValueError("FastWAM sample is missing images")
        camera_frames = []
        for image in images:
            if image.ndim != 3:
                raise ValueError(f"FastWAM image must be [C,H,W], got {tuple(image.shape)}")
            if image.shape[0] != 3:
                raise ValueError(f"FastWAM image channel dimension must be 3, got {image.shape[0]}")
            image = image.float()
            if image.max() > 2.0:
                image = image / 255.0
            camera_frames.append(image * 2.0 - 1.0)
        first_frame = torch.cat(camera_frames, dim=-1) if len(camera_frames) > 1 else camera_frames[0]
        frames = [first_frame.clone() for _ in range(self.num_video_frames)]
        return torch.stack(frames, dim=1)

    @staticmethod
    def _require_action(example: Dict[str, Any]) -> torch.Tensor:
        """Extract and validate the action tensor from a sample."""
        action = example.get("action")
        if action is None:
            raise ValueError("FastWAM sample is missing action")
        action = action.float()
        if action.ndim != 2:
            raise ValueError(f"FastWAM action must be [T,D], got {tuple(action.shape)}")
        return action

    @staticmethod
    def _build_proprio(example: Dict[str, Any], action_horizon: int) -> Optional[torch.Tensor]:
        """Build a ``[T, D]`` proprioception tensor aligned to the action horizon."""
        proprio = example.get("proprio", None)
        if proprio is None:
            return None
        proprio = proprio.float()
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0).expand(action_horizon, -1).clone()
        elif proprio.ndim == 2 and proprio.shape[0] != action_horizon:
            proprio = proprio[:1].expand(action_horizon, -1).clone()
        elif proprio.ndim != 2:
            raise ValueError(f"FastWAM proprio must be [D] or [T,D], got {tuple(proprio.shape)}")
        return proprio

    @staticmethod
    def _build_action_is_pad(examples: List[Dict[str, Any]], action_shape: tuple) -> torch.Tensor:
        """Build the ``action_is_pad`` boolean mask tensor for the batch."""
        pads = [ex.get("action_is_pad", None) for ex in examples]
        if all(p is not None for p in pads):
            return torch.stack(
                [p.bool() if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.bool) for p in pads],
                dim=0,
            )
        return torch.zeros(action_shape[:2], dtype=torch.bool)

    def _build_context(self, prompts: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build text context tensors from prompt strings or cache files."""
        cache_paths = self._context_cache_paths(prompts)
        if cache_paths:
            contexts, masks = [], []
            for path in cache_paths:
                payload = torch.load(path, map_location="cpu")
                context = payload.get("context", payload.get("prompt_emb", payload.get("embedding")))
                mask = payload.get("context_mask", payload.get("mask", payload.get("attention_mask")))
                if context is None or mask is None:
                    raise KeyError(f"FastWAM text cache missing context/mask tensors: {path}")
                contexts.append(context)
                masks.append(mask)
            context = torch.stack(contexts, dim=0).float()
            mask = torch.stack(masks, dim=0).bool()
            context[~mask] = 0.0
            return context, torch.ones_like(mask, dtype=torch.bool)

        context = torch.zeros((len(prompts), self.context_len, self.text_dim), dtype=torch.float32)
        context_mask = torch.ones((len(prompts), self.context_len), dtype=torch.bool)
        return context, context_mask

    def _context_cache_paths(self, prompts: List[str]) -> Optional[List[Path]]:
        """Resolve pre-computed text embedding cache file paths for each prompt."""
        cache_dir = self.data_cfg.text_embedding_cache_dir
        if not cache_dir:
            return None
        import hashlib

        paths = []
        cache_root = Path(str(cache_dir))
        enc_id = _model_id_to_enc_id(self.model_id)
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            path = cache_root / f"{digest}.t5_len{self.context_len}.{enc_id}.pt"
            if not path.exists():
                legacy_path = cache_root / f"{hashlib.sha1(prompt.encode('utf-8')).hexdigest()}.pt"
                if legacy_path.exists():
                    path = legacy_path
                else:
                    raise FileNotFoundError(f"Missing FastWAM text embedding cache for prompt '{prompt}': {path}")
            paths.append(path)
        return paths
