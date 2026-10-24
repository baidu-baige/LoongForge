# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""LoongForge registry wrapper around the pinned VPP2 numerical modules."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import init_on_device

from loongforge.embodied.model.registry import register_model

from .modeling_configuration_vpp2 import VPP2ModelConfig
from .modules.action_dit import ActionDiT
from .modules.helpers.io import load_state_dict
from .modules.helpers.loader import _convert_vae_state_dict, _load_state_dict_assign
from .modules.libero_action_model import LiberoActionModel
from .modules.mot import MoT
from .modules.wan_video_clip import WanImageEncoderCLIP
from .modules.wan_video_dit import WanVideoDiT
from .modules.wan_video_vae import WanVideoVAE

logger = logging.getLogger(__name__)


def _dtype(value: str) -> torch.dtype:
    name = str(value).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported VPP2 dtype: {value!r}")


@register_model("vpp2")
class VPP2Model(LiberoActionModel):
    """Action-stage VPP2 model implementing the embodied trainer contract."""

    _no_split_modules = ["DiTBlock", "WanVideoDiT", "ActionDiT"]
    config_class = VPP2ModelConfig

    @classmethod
    def from_pretrained(cls, cfg: VPP2ModelConfig) -> "VPP2Model":
        if not isinstance(cfg, VPP2ModelConfig):
            raise TypeError(f"VPP2Model.from_pretrained expects VPP2ModelConfig, got {type(cfg).__name__}")
        compute_dtype = _dtype(cfg.dtype)
        # Always construct the large tower on meta.  The eager training mode
        # below materializes it in the configured dtype before loading; direct
        # CUDA construction would allocate the default fp32 parameters first
        # and OOM while the video checkpoint is staged.
        device = torch.device("meta")
        video_cfg = dict(cfg.video_dit_config)
        action_cfg = dict(cfg.action_dit_config)
        video_cfg["action_dim"] = cfg.action_dim
        action_cfg["action_dim"] = cfg.action_dim
        # P2-1 deliberately builds a storage-free shell.  P2-2 owns loading and
        # materialization; no 32 GB checkpoint is touched here.
        with init_on_device(device, include_buffers=False):
            video_expert = WanVideoDiT(**video_cfg)
            action_expert = ActionDiT(**action_cfg)
            mot = MoT(
                mixtures={"video": video_expert, "action": action_expert},
                mot_checkpoint_mixed_attn=cfg.mot_checkpoint_mixed_attn,
                use_video_tokens=False,
            )
            vae = WanVideoVAE(z_dim=16)
            clip = WanImageEncoderCLIP() if cfg.load_clip_encoder else None
            model = cls(
                video_expert=video_expert,
                action_expert=action_expert,
                mot=mot,
                vae=vae,
                text_encoder=None,
                tokenizer=None,
                clip_encoder=clip,
                text_dim=int(video_cfg["text_dim"]),
                proprio_dim=cfg.proprio_dim,
                device=str(device),
                torch_dtype=compute_dtype,
                video_train_shift=float(cfg.video_scheduler["train_shift"]),
                video_infer_shift=float(cfg.video_scheduler["infer_shift"]),
                video_num_train_timesteps=int(cfg.video_scheduler["num_train_timesteps"]),
                video_train_sampling_strategy=str(cfg.video_scheduler["train_sampling_strategy"]),
                video_train_beta_alpha=float(cfg.video_scheduler["beta_alpha"]),
                video_train_beta_beta=float(cfg.video_scheduler["beta_beta"]),
                action_train_shift=float(cfg.action_scheduler["train_shift"]),
                action_infer_shift=float(cfg.action_scheduler["infer_shift"]),
                action_num_train_timesteps=int(cfg.action_scheduler["num_train_timesteps"]),
                loss_lambda_video=float(cfg.loss["lambda_video"]),
                loss_lambda_action=float(cfg.loss["lambda_action"]),
                action_chunk_size=cfg.action_chunk_size,
                action_visible_video_frames=cfg.action_visible_video_frames,
                action_only_first_frame_fast_path=cfg.action_only_first_frame_fast_path,
            )
        model.config = cfg
        model.model_paths = {
            "video_dit": cfg.video_checkpoint_path,
            "action_dit_backbone": cfg.action_dit_pretrained_path,
        }
        if not cfg.init_on_meta:
            # The generic embodied trainer wraps a regular DDP model before its
            # deferred-meta materialization hook.  Eager VPP2 construction must
            # therefore load the complete strict bundle before that wrapper.
            target = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            model.materialize(target, cfg.dtype)
            model.load_pretrained(cfg.video_checkpoint_path, device=target)
            logger.info(
                "VPP2 eager strict bundle loaded: video=%s action=%s",
                cfg.video_checkpoint_path,
                cfg.action_dit_pretrained_path,
            )
        model.freeze_modules()
        return model

    @staticmethod
    def default_fp8_targets() -> dict[str, Any]:
        return {
            "module_patterns": ["video_expert.blocks", "action_expert.blocks"],
            "skip_modules": [],
        }

    @staticmethod
    def default_lora_targets() -> dict[str, Any]:
        return {"target_modules": [], "modules_to_save": []}

    def set_frozen_modules_to_eval_mode(self) -> None:
        for module in (self.video_expert, self.vae, self.clip_encoder, self.text_encoder):
            if module is not None:
                module.eval()
        # BASE enters action stage with the complete model in eval mode and
        # re-enables only ActionDiT/proprio.  ``MoT`` owns the mixed-attention
        # checkpoint switch, so keep its wrapper in eval without recursively
        # changing the action expert back to eval.
        self.mot.training = False

    def _assert_action_stage_freeze_contract(self) -> None:
        if any(parameter.requires_grad for parameter in self.video_expert.parameters()):
            raise RuntimeError("Action stage must freeze the complete Video expert")

    def freeze_modules(self) -> None:
        # The action-stage BASE freezes the entire video pathway and trains only
        # ActionDiT plus the proprio projection.
        for module in (self.video_expert, self.vae, self.clip_encoder, self.text_encoder):
            if module is not None:
                module.requires_grad_(False)
        self.action_expert.requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.requires_grad_(True)
        self.set_frozen_modules_to_eval_mode()
        self._assert_action_stage_freeze_contract()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.set_frozen_modules_to_eval_mode()
            self._assert_action_stage_freeze_contract()
        return self

    def materialize(self, device: torch.device | str, dtype: str = "bfloat16") -> None:
        """Materialize the P2-1 meta shell for the later checkpoint-loading task."""
        target = torch.device(device)
        self.to_empty(device=target)
        compute_dtype = _dtype(dtype)
        # BASE keeps the trainable DiT path in the configured compute dtype,
        # while its frozen image tower and VAE are deliberately pinned to fp16
        # and fp32 respectively.  Do not call ``self.to(dtype=...)`` here: that
        # would silently violate the auxiliary precision contract.
        self.video_expert.to(device=target, dtype=compute_dtype)
        self.action_expert.to(device=target, dtype=compute_dtype)
        if self.proprio_encoder is not None:
            self.proprio_encoder.to(device=target, dtype=compute_dtype)
        self.vae.to(device=target, dtype=torch.float32)
        if self.clip_encoder is not None:
            self.clip_encoder.to(device=target, dtype=torch.float16)
        # ``to_empty`` intentionally does not initialize storage.  These four
        # action projection tensors are excluded from action_init.pt by BASE's
        # documented ``projection_initialization=random`` policy, so initialize
        # them exactly once before merging the strict backbone state.
        # ActionDiT.from_pretrained in BASE constructs these two projections
        # with ordinary fp32 ``nn.Linear`` initialization and only then casts
        # the complete expert to bf16.  Reproduce that order rather than
        # initializing a bf16 CPU tensor directly.
        self.action_expert.action_encoder.to(device=target, dtype=torch.float32)
        self.action_expert.head.to(device=target, dtype=torch.float32)
        self.action_expert.action_encoder.reset_parameters()
        self.action_expert.head.reset_parameters()
        self.action_expert.action_encoder.to(device=target, dtype=compute_dtype)
        self.action_expert.head.to(device=target, dtype=compute_dtype)
        if self.proprio_encoder is not None:
            self.proprio_encoder.reset_parameters()
        self.device = target
        self.torch_dtype = compute_dtype
        self.freeze_modules()

    def load_pretrained(self, path: str, device=None):
        """Load the BASE video/action bundle with strict key and shape checks.

        ``path`` identifies the video checkpoint (the trainer's generic
        pretrained-checkpoint slot).  The action initializer and Wan auxiliary
        assets come from the typed model config.  The action initializer carries
        only the 1010 backbone tensors by design; the four projection tensors
        remain the freshly initialized tensors described by its policy metadata,
        then the *merged* 1014-key state is loaded with ``strict=True``.
        """
        if any(parameter.is_meta for parameter in self.parameters()):
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.materialize(device, self.config.dtype)
        video_path = Path(path) if path else Path(self.config.video_checkpoint_path)
        if not video_path.is_file():
            video_path = Path(self.config.video_checkpoint_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"VPP2 video checkpoint not found: {video_path}")

        video_payload = torch.load(
            video_path, map_location="cpu", mmap=True, weights_only=True
        )
        if not isinstance(video_payload, dict):
            raise ValueError(f"Invalid VPP2 video checkpoint payload: {type(video_payload)}")
        if video_payload.get("format") not in {None, "wan21_video_v1"}:
            raise ValueError(
                f"Unexpected VPP2 video checkpoint format: {video_payload.get('format')!r}"
            )
        video_state = video_payload.get("dit")
        if video_state is None and isinstance(video_payload.get("mot"), dict):
            video_state = {
                key.removeprefix("mixtures.video."): value
                for key, value in video_payload["mot"].items()
                if key.startswith("mixtures.video.")
            }
        if not isinstance(video_state, dict):
            raise ValueError("VPP2 video checkpoint has no `dit` state dictionary")
        video_device = next(self.video_expert.parameters()).device
        video_state = {
            key: value.to(device=video_device, dtype=next(self.video_expert.parameters()).dtype)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in video_state.items()
        }
        video_result = _load_state_dict_assign(self.video_expert, video_state, strict=True)
        if video_result.missing_keys or video_result.unexpected_keys:
            raise RuntimeError(
                "Strict VPP2 video load returned incompatible keys: "
                f"missing={video_result.missing_keys}, unexpected={video_result.unexpected_keys}"
            )

        action_path = Path(self.config.action_dit_pretrained_path)
        if not action_path.is_file():
            raise FileNotFoundError(f"VPP2 action initializer not found: {action_path}")
        action_payload = torch.load(
            action_path, map_location="cpu", mmap=True, weights_only=True
        )
        if not isinstance(action_payload, dict):
            raise ValueError(f"Invalid VPP2 action checkpoint payload: {type(action_payload)}")
        if action_payload.get("format") not in {"wan21_action_v1", "vpp2_wan21_action_v1"}:
            raise ValueError(
                f"Unexpected VPP2 action checkpoint format: {action_payload.get('format')!r}"
            )
        action_meta = action_payload.get("meta")
        if not isinstance(action_meta, dict):
            raise ValueError("VPP2 action initializer is missing its `meta` dictionary")
        for key in ActionDiT.ACTION_BACKBONE_META_KEYS:
            if key not in action_meta:
                raise ValueError(f"VPP2 action initializer is missing meta.{key}")
            if key == "eps":
                if abs(float(action_meta[key]) - float(self.config.action_dit_config[key])) > 1e-12:
                    raise ValueError(f"Action initializer meta.{key} does not match model config")
            elif int(action_meta[key]) != int(self.config.action_dit_config[key]):
                raise ValueError(
                    f"Action initializer meta.{key}={action_meta[key]!r} does not match "
                    f"configured value {self.config.action_dit_config[key]!r}"
                )
        source_backbone = action_payload.get("backbone_state_dict")
        if not isinstance(source_backbone, dict):
            raise ValueError("VPP2 action initializer has no backbone_state_dict")
        target_state = self.action_expert.state_dict()
        backbone_keys = ActionDiT.backbone_key_set(target_state.keys())
        source_keys = set(source_backbone)
        missing_backbone = sorted(backbone_keys - source_keys)
        unexpected_backbone = sorted(source_keys - backbone_keys)
        if missing_backbone or unexpected_backbone:
            raise RuntimeError(
                "Strict VPP2 action-backbone key check failed: "
                f"missing={missing_backbone}, unexpected={unexpected_backbone}"
            )
        merged_action = dict(target_state)
        for key in sorted(backbone_keys):
            value = source_backbone[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Action initializer tensor {key!r} is {type(value)}")
            if tuple(value.shape) != tuple(target_state[key].shape):
                raise RuntimeError(
                    f"Action initializer shape mismatch for {key!r}: "
                    f"source={tuple(value.shape)} target={tuple(target_state[key].shape)}"
                )
            merged_action[key] = value.to(
                device=target_state[key].device, dtype=target_state[key].dtype
            )
        action_result = _load_state_dict_assign(self.action_expert, merged_action, strict=True)
        if action_result.missing_keys or action_result.unexpected_keys:
            raise RuntimeError(
                "Strict VPP2 action load returned incompatible keys: "
                f"missing={action_result.missing_keys}, unexpected={action_result.unexpected_keys}"
            )

        # Load the frozen auxiliary towers exactly as BASE does.  These loads
        # are separate from the two training checkpoints so their precision and
        # eval/freeze invariants can be audited independently.
        model_dir = Path(self.config.model_id)
        vae_state = load_state_dict(
            str(model_dir / "Wan2.1_VAE.pth"), torch_dtype=torch.float32, device="cpu"
        )
        vae_device = next(self.vae.parameters()).device
        vae_state = {
            key: value.to(device=vae_device, dtype=torch.float32)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in vae_state.items()
        }
        vae_result = _load_state_dict_assign(
            self.vae, _convert_vae_state_dict(vae_state), strict=True
        )
        if vae_result.missing_keys or vae_result.unexpected_keys:
            raise RuntimeError(
                f"Strict VPP2 VAE load failed: missing={vae_result.missing_keys}, "
                f"unexpected={vae_result.unexpected_keys}"
            )
        if self.clip_encoder is not None:
            clip_state = load_state_dict(
                str(model_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                torch_dtype=torch.float16,
                device="cpu",
            )
            clip_state = {
                key: value
                for key, value in clip_state.items()
                if key.startswith("visual.") or key == "log_scale"
            }
            clip_device = next(self.clip_encoder.parameters()).device
            clip_state = {
                key: value.to(device=clip_device, dtype=torch.float16)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in clip_state.items()
            }
            clip_result = _load_state_dict_assign(self.clip_encoder, clip_state, strict=True)
            if clip_result.missing_keys or clip_result.unexpected_keys:
                raise RuntimeError(
                    f"Strict VPP2 CLIP load failed: missing={clip_result.missing_keys}, "
                    f"unexpected={clip_result.unexpected_keys}"
                )
        self.freeze_modules()
        return {
            "video": {
                "path": str(video_path),
                "source_keys": len(video_state),
                "missing_keys": list(video_result.missing_keys),
                "unexpected_keys": list(video_result.unexpected_keys),
            },
            "action": {
                "path": str(action_path),
                "source_backbone_keys": len(source_backbone),
                "target_keys": len(target_state),
                "missing_backbone": missing_backbone,
                "unexpected_backbone": unexpected_backbone,
                "missing_keys": list(action_result.missing_keys),
                "unexpected_keys": list(action_result.unexpected_keys),
            },
            "vae": {
                "missing_keys": list(vae_result.missing_keys),
                "unexpected_keys": list(vae_result.unexpected_keys),
            },
            "clip": None
            if self.clip_encoder is None
            else {
                "missing_keys": list(clip_result.missing_keys),
                "unexpected_keys": list(clip_result.unexpected_keys),
            },
        }

    def forward(self, batch, iteration=None, **kwargs):
        """Run BASE's action-only training loss with the trainer's iteration kwarg."""
        if os.environ.get("VPP2_ITERATION_PROBE", "0").strip().lower() in {"1", "true", "yes", "on"}:
            logger.info("VPP2_ITERATION_PROBE iteration=%r", iteration)
        del iteration, kwargs
        sample = batch.to_sample() if hasattr(batch, "to_sample") else batch
        loss, metrics = self.training_loss(sample, training_stage="action")
        video_loss = metrics.get("loss_video", 0.0)
        action_loss = metrics.get("loss_action", loss.detach())
        device = loss.device
        video_loss = video_loss if torch.is_tensor(video_loss) else torch.tensor(float(video_loss), device=device)
        action_loss = action_loss if torch.is_tensor(action_loss) else torch.tensor(float(action_loss), device=device)
        return loss, {"video_loss": video_loss.detach(), "action_loss": action_loss.detach()}
