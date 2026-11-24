# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VLA v2 inference model for the LoongForge eval server.

Eval serves the upstream ``lingbotvla`` policy directly (the training adapter in
``modeling_lingbot_vla_v2.py`` wraps the same ``LingbotVlaV2Policy``), and mirrors
the community RoboTwin policy server ``deploy/lingbot_vla_v2_policy.py`` step by
step so the release protocol is reproduced:

- config from ``<ckpt>/../../../lingbotvla_cli.yaml`` (the training dump next to
  ``checkpoints/global_step_N/hf_ckpt``), eager joint attention, ``use_cache``;
- build -> ``load_state_dict(strict=True)`` -> ``.float()`` in that order, so the
  bf16-constructed submodules (action-expert attention, Qwen3-VL text model) keep
  the same load-time rounding as the community "FP32" server;
- ``torch.compile`` of ``qwenvl_with_expert`` / ``sample_actions`` and the global
  seed 42 of the community server;
- images resized to ``data.img_size`` on float32 tensors, then
  ``FeatureTransform.apply(policy_eval=True)`` / ``sample_actions`` /
  ``FeatureTransform.unapply``; the first ``use_length`` (50) absolute 14D joint
  actions are returned and executed open-loop by the chunk-cached eval policy.

``noise_seed`` (off by default) draws the initial flow-matching noise from a
fixed generator on every call; it exists only for parity checks against the
community server. ``dump_dir`` records the first ``dump_max`` calls' inputs and
outputs for the same purpose.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

# Community RoboTwin client keys, in the camera order of configs/robot_configs/robotwin.yaml.
IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
STATE_KEY = "observation.state"
STATE_DIM = 14


@dataclass
class LingBotVLAV2InferenceConfig:
    """Eval-side knobs; YAML ``model:`` keys map 1:1 onto these fields."""

    # Training config dumped by the trainer; empty -> <ckpt>/../../../lingbotvla_cli.yaml.
    training_config_path: str = ""
    # Qwen3-VL config / tokenizer / processor dir; empty -> $QWEN3VL_PATH, then the yaml.
    qwen3vl_path: str = ""
    # Empty -> <lingbotvla repo>/configs/robot_configs/<robo_name>.yaml.
    robot_config_path: str = ""
    robo_name: str = "robotwin"
    # Empty -> the robot config's ``norm_stats`` entry resolved against the lingbotvla repo.
    norm_stats_path: str = ""
    use_length: int = 50
    use_compile: bool = True
    attention_implementation: str = "eager"
    seed: int = 42
    noise_seed: Optional[int] = None
    dump_dir: str = ""
    dump_max: int = 0


def set_seed_everywhere(seed: int) -> None:
    """Same global seeding as the community policy server (applied at its import)."""
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def fixed_noise(seed: int, batch: int, config: Any):
    """Initial flow-matching noise for parity runs: CPU generator, so every env draws the same tensor."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    return torch.randn((batch, config.n_action_steps, config.max_action_dim), generator=generator, dtype=torch.float32)


def _transformers_major() -> int:
    import transformers

    return int(transformers.__version__.split(".")[0])


def _upstream(symbol: str, module: str):
    """Import an upstream symbol through the training adapter's guarded path."""
    from loongforge.embodied.model.lingbot_vla_v2.upstream_adapter import _resolve_symbol

    return _resolve_symbol(symbol, module)


def _apply_upstream_patches() -> None:
    """Upstream Qwen3-VL patch, plus the adapter's Transformers-5 fixups when needed."""
    _upstream("apply_lingbot_qwen3_vl_patch", "lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla")()
    if _transformers_major() >= 5:
        from loongforge.embodied.model.lingbot_vla_v2 import upstream_adapter

        upstream_adapter._apply_qwen3_vl_structure_fixups()
        upstream_adapter._apply_qwen2_action_expert_fixups()


def _lingbotvla_repo_root() -> Path:
    import lingbotvla

    return Path(lingbotvla.__file__).resolve().parents[1]


# Text keys copied from the Qwen3-VL config onto the policy config (deploy ``merge_qwen_config``).
_QWEN_TEXT_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_theta",
    "vocab_size",
    "max_position_embeddings",
    "hidden_act",
    "tie_word_embeddings",
    "tokenizer_path",
)


def _merge_qwen_config(config: Any, qwen_config: Any) -> None:
    config_dict = qwen_config.to_dict() if hasattr(qwen_config, "to_dict") else qwen_config
    text_config = config_dict.get("text_config", {})
    for key in _QWEN_TEXT_KEYS:
        if key in text_config:
            setattr(config, key, text_config[key])
        elif key in config_dict:
            setattr(config, key, config_dict[key])
    if "vision_config" in config_dict:
        config.vision_config = qwen_config.vision_config


def _pad_and_stack(values: List[Any]):
    """Batch per-sample tensors (deploy ``_pad_and_stack_tensors``)."""
    import torch

    shapes = [tuple(v.shape) for v in values]
    if len(set(shapes)) == 1:
        return torch.stack(values, dim=0)
    if all(v.ndim == 1 for v in values):
        max_len = max(v.shape[0] for v in values)
        fill = False if values[0].dtype == torch.bool else 0
        padded = []
        for v in values:
            out = torch.full((max_len,), fill, dtype=v.dtype, device=v.device)
            out[: v.shape[0]] = v
            padded.append(out)
        return torch.stack(padded, dim=0)
    raise ValueError(f"Cannot batch tensors with different shapes: {shapes}")


class LingBotVLAV2PredictActionModel:
    """Upstream LingBot-VLA v2 policy behind the shared ``predict_action`` contract."""

    def __init__(self, vla: Any, sample_actions_fn: Any, feature_transform: Any, data_config: Any,
                 config: Any, cfg: LingBotVLAV2InferenceConfig, use_bf16: bool, device: str) -> None:
        self.vla = vla
        self.sample_actions_fn = sample_actions_fn
        self.feature_transform = feature_transform
        self.data_config = data_config
        self.config = config
        self.cfg = cfg
        self.use_bf16 = bool(use_bf16)
        self.device = device
        self.action_keys = list(feature_transform.org_features["actions"])
        self.image_keys = list(feature_transform.org_features["images"])
        self._num_calls = 0

    @classmethod
    def from_pretrained(cls, ckpt_path: str, cfg: LingBotVLAV2InferenceConfig, use_bf16: bool = False,
                        device: str = "cuda") -> "LingBotVLAV2PredictActionModel":
        """Build the policy exactly as the community server does and load ``hf_ckpt``."""
        import torch
        import yaml

        config_cls = _upstream("LingbotVLAV2Config", "lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla")
        policy_cls = _upstream("LingbotVlaV2Policy", "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2")
        feature_transform_cls = _upstream("FeatureTransform", "lingbotvla.data.vla_data.utils")
        build_processor = _upstream("build_processor", "lingbotvla.models")
        from safetensors import safe_open
        from transformers import AutoConfig

        set_seed_everywhere(cfg.seed)
        _apply_upstream_patches()

        ckpt = Path(ckpt_path).expanduser()
        training_config_path = Path(cfg.training_config_path or ckpt.parent.parent.parent / "lingbotvla_cli.yaml")
        with open(training_config_path, "r") as f:
            training_config = yaml.safe_load(f)
        model_config = dict(training_config["model"])
        model_config.update(training_config["train"])
        config = config_cls(**model_config)
        for key, value in model_config.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        config.attention_implementation = cfg.attention_implementation
        base_model = training_config["model"]["tokenizer_path"]
        if not ("qwen3" in base_model.lower() and "vl" in base_model.lower()):
            raise ValueError(f"Unsupported base model {base_model!r} in {training_config_path}")
        qwen3vl_path = cfg.qwen3vl_path or os.environ.get("QWEN3VL_PATH", "") or base_model
        config.tokenizer_path = qwen3vl_path
        _merge_qwen_config(config, AutoConfig.from_pretrained(qwen3vl_path))
        if training_config["model"].get("vocab_size", 0):
            config.vocab_size = training_config["model"]["vocab_size"]
        config.use_cache = True
        processor = build_processor(qwen3vl_path)
        data_config = SimpleNamespace(**training_config["data"])

        vla = policy_cls(config, eval=True)
        merged = {}
        for file_path in sorted(glob(str(ckpt / "*.safetensors"))):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged[key] = f.get_tensor(key)
        if not merged:
            raise FileNotFoundError(f"no *.safetensors under {ckpt}")
        vla.load_state_dict(merged, strict=True)
        del merged

        vla.model._use_compile_predict_velocity = bool(cfg.use_compile)
        vla.model._compiled_predict_velocity = None
        sample_actions_fn = vla.model.sample_actions
        if cfg.use_compile:
            vla.model.qwenvl_with_expert = torch.compile(vla.model.qwenvl_with_expert)
            sample_actions_fn = torch.compile(vla.model.sample_actions)
        if use_bf16:
            vla = vla.to(torch.bfloat16).to(device).eval()
        else:
            vla.model.float()
            vla = vla.to(device).eval()

        repo_root = _lingbotvla_repo_root()
        robot_config_path = Path(cfg.robot_config_path or repo_root / "configs/robot_configs" / f"{cfg.robo_name}.yaml")
        norm_stats_path = cfg.norm_stats_path or getattr(data_config, "norm_stats_file", None)
        if not norm_stats_path:
            with open(robot_config_path, "r") as f:
                norm_stats_path = yaml.safe_load(f)["norm_stats"]
        norm_stats_path = Path(norm_stats_path)
        if not norm_stats_path.is_absolute():
            norm_stats_path = repo_root / norm_stats_path
        feature_transform = feature_transform_cls(
            str(robot_config_path), data_config, config, processor,
            chunk_size=config.chunk_size, norm_stats_path=str(norm_stats_path),
        )
        vla.feature_transform = feature_transform
        return cls(vla, sample_actions_fn, feature_transform, data_config, config, cfg, use_bf16, device)

    # ------------------------------------------------------------------
    # Inference (deploy ``_prepare_model_input`` / ``sample_actions_batch`` /
    # ``_unapply_batched_actions`` for a batch of one)
    # ------------------------------------------------------------------
    def _prepare_model_input(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from torchvision.transforms.v2 import Resize

        observation = dict(observation)
        image_size = getattr(self.data_config, "img_size", 256)
        resize = Resize((image_size, image_size))
        for key in self.image_keys:
            image = observation[key]
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{key} must be HWC RGB, got shape {image.shape}")
            observation[key] = resize(torch.as_tensor(image).permute(2, 0, 1).contiguous().to(dtype=torch.float32))
        for key, value in list(observation.items()):
            if isinstance(value, np.ndarray):
                observation[key] = torch.from_numpy(value)
        observation = self.feature_transform.apply(observation, policy_eval=True)
        if self.use_bf16:
            observation["state"] = observation["state"].to(torch.bfloat16)
        return observation

    def _sample(self, batch: Dict[str, Any]):
        import torch

        dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        device = self.device
        images, img_masks = batch["images"], batch["img_masks"]
        lang_tokens, lang_masks, state = batch["lang_tokens"], batch["lang_masks"], batch["state"]
        if img_masks.ndim < 2:
            images, img_masks = images.unsqueeze(0), img_masks.unsqueeze(0)
        if lang_tokens.ndim == 1:
            lang_tokens, lang_masks = lang_tokens.unsqueeze(0), lang_masks.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        grid = batch.get("image_grid_thw")
        kwargs: Dict[str, Any] = {
            "image_grid_thw": None if grid is None else grid.to(device=device, dtype=torch.long),
        }
        if self.cfg.noise_seed is not None:
            kwargs["noise"] = fixed_noise(int(self.cfg.noise_seed), state.shape[0], self.config).to(device=device, dtype=dtype)
        with torch.no_grad():
            actions = self.sample_actions_fn(
                images.to(dtype=dtype, device=device),
                img_masks.to(device=device),
                lang_tokens.to(device=device),
                lang_masks.to(device=device),
                state.to(dtype=dtype, device=device),
                **kwargs,
            )
        return actions.to(dtype=torch.float32, device="cpu")

    def predict_action(self, images: Any, instructions: Any, state: Optional[Any] = None,
                       dataset_stats: Optional[Dict[str, Any]] = None, episode_id: Optional[str] = None,
                       episode_step: int = 0) -> np.ndarray:
        """Return the (use_length, 14) absolute joint chunk for one RoboTwin observation.

        ``images`` is the eval batch-of-one ``[[head, left, right]]`` (HWC uint8);
        ``state`` the raw 14D ``joint_action.vector``. Normalization lives in the
        upstream ``FeatureTransform`` (the robot config's norm stats), so
        ``dataset_stats`` is unused.
        """
        import torch

        views = images[0] if len(images) == 1 and isinstance(images[0], (list, tuple)) else images
        if len(views) != len(IMAGE_KEYS):
            raise ValueError(f"lingbot_vla_v2 expects {len(IMAGE_KEYS)} views, got {len(views)}")
        instruction = instructions[0] if isinstance(instructions, (list, tuple)) else instructions
        if state is None:
            raise ValueError("lingbot_vla_v2 requires the 14D joint state")
        observation: Dict[str, Any] = {key: np.array(view, dtype=np.uint8, copy=True) for key, view in zip(IMAGE_KEYS, views)}
        observation[STATE_KEY] = np.array(state, dtype=np.float64, copy=True).reshape(-1)
        if observation[STATE_KEY].shape[0] != STATE_DIM:
            raise ValueError(f"lingbot_vla_v2 expects a {STATE_DIM}D state, got {observation[STATE_KEY].shape}")
        observation["task"] = str(instruction)

        applied = self._prepare_model_input(observation)
        batch = {k: (_pad_and_stack([v]) if isinstance(v, torch.Tensor) else [v]) for k, v in applied.items()}
        actions = self._sample(batch)

        single = dict(applied)
        single["actions"] = actions[0]
        if self.use_bf16 and "state" in single:
            single["state"] = single["state"].to(dtype=torch.float32)
        data = self.feature_transform.unapply(single)
        if len(self.action_keys) != 1:
            raise ValueError(f"expected one action feature, got {self.action_keys}")
        value = data[self.action_keys[0]]
        chunk = value.float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.float32)
        if chunk.shape[0] < self.cfg.use_length:
            raise ValueError(f"model chunk {chunk.shape[0]} shorter than use_length {self.cfg.use_length}")
        chunk = chunk[: self.cfg.use_length].astype(np.float32)
        self._maybe_dump(observation, applied, actions[0], chunk, episode_id, episode_step)
        self._num_calls += 1
        return chunk

    def _maybe_dump(self, observation, applied, normalized, chunk, episode_id, episode_step) -> None:
        if not self.cfg.dump_dir or self._num_calls >= self.cfg.dump_max:
            return
        import torch

        out = Path(self.cfg.dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        record = {f"obs/{k}": v for k, v in observation.items() if isinstance(v, np.ndarray)}
        record["obs/task"] = np.asarray(observation["task"])
        for key, value in applied.items():
            if isinstance(value, torch.Tensor):
                record[f"in/{key}"] = value.detach().float().cpu().numpy() if value.is_floating_point() else value.cpu().numpy()
        record["out/normalized"] = normalized.numpy()
        record["out/actions"] = chunk
        record["meta/episode"] = np.asarray(f"{episode_id}|{episode_step}")
        np.savez_compressed(out / f"call_{self._num_calls:05d}.npz", **record)
