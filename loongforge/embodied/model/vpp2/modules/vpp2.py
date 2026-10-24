# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE and THIRD_PARTY_NOTICES.md for retained source notices.
from contextlib import contextmanager, nullcontext
from typing import Any, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from .logging_utils import get_logger
from .action_dit import ActionDiT
from .helpers.loader import load_wan21_i2v_14b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)
JOINT_DENOISING_MODES = ("policy", "forward_dynamics", "inverse_dynamics")
JOINT_DENOISING_SAMPLERS = ("euler",)
UNIPC_SIGMA_START = 0.999


def _normalize_joint_denoising_mode_probs(probs):
    if probs:
        raise ValueError("Joint denoising modes are unsupported in this release")
    return {}


def _normalize_joint_denoising_sampler(sampler):
    if sampler != "euler":
        raise ValueError("Only cached-video Euler action inference is supported")
    return "euler"


WAN_OFFICIAL_PEFT_VERSION = "0.14.0"
WAN_OFFICIAL_VIDEO_LORA_TARGET_MODULES = ("q", "k", "v", "o", "ffn.0", "ffn.2")


def _normalize_video_lora_config(config):
    if config and config.get("enabled", False):
        raise ValueError("This release trains the complete Video DiT; LoRA is unsupported")
    return {"enabled": False}


def _apply_wan_official_video_lora(video_expert, config):
    _normalize_video_lora_config(config)
    return video_expert


class VPP2(torch.nn.Module):
    """MoT world model with video/action experts."""

    supports_action_only_fast_path = False

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        clip_encoder=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        video_train_sampling_strategy: str = "shifted_uniform",
        video_train_beta_alpha: float = 7.0,
        video_train_beta_beta: float = 1.0,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        video_lora_joint_video_loss: bool = False,
        train_wan: bool = True,
        action_chunk_size: int = 1,
        action_visible_video_frames: int = 1,
        action_only_first_frame_fast_path: bool = False,
        action_clean_first_frame_start_layer: Optional[int] = None,
        action_clean_first_frame_timestep_cutoff: Optional[float] = None,
        action_first_frame_kv_source: str = "clean_prefill",
        action_video_timestep_cutoff: Optional[float] = None,
        video_lora_config: Optional[dict[str, Any]] = None,
        use_fixed_video_layers: Optional[int] = None,
        use_video_tokens: bool = False,
        video_token_latent_dim: int = 2560,
        joint_denoising: bool = False,
        joint_denoising_mode_probs: Optional[dict[str, float]] = None,
        joint_denoising_sampler: str = "euler",
    ):
        if (
            joint_denoising
            or use_video_tokens
            or use_fixed_video_layers is not None
            or (action_only_first_frame_fast_path and not self.supports_action_only_fast_path)
        ):
            raise ValueError("Unsupported experimental model option")
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.clip_encoder = clip_encoder
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
            train_sampling_strategy=video_train_sampling_strategy,
            beta_alpha=video_train_beta_alpha,
            beta_beta=video_train_beta_beta,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_infer_shift
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_train_shift
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_infer_shift
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.video_lora_joint_video_loss = bool(video_lora_joint_video_loss)
        self.train_mode = "joint"
        self.train_wan = True
        self.action_chunk_size = int(action_chunk_size)
        self.action_visible_video_frames = int(action_visible_video_frames)
        self.action_only_first_frame_fast_path = bool(action_only_first_frame_fast_path)
        self.action_clean_first_frame_start_layer = (
            None
            if action_clean_first_frame_start_layer is None
            else int(action_clean_first_frame_start_layer)
        )
        self.action_clean_first_frame_timestep_cutoff = (
            None
            if action_clean_first_frame_timestep_cutoff is None
            else float(action_clean_first_frame_timestep_cutoff)
        )
        self.action_first_frame_kv_source = str(action_first_frame_kv_source).strip().lower()
        if self.action_first_frame_kv_source not in {"clean_prefill", "noisy_cache_prefix"}:
            raise ValueError(
                f"`action_first_frame_kv_source` must be 'clean_prefill' or 'noisy_cache_prefix', got {action_first_frame_kv_source!r}."
            )
        self.action_video_timestep_cutoff = (
            None if action_video_timestep_cutoff is None else float(action_video_timestep_cutoff)
        )
        self.video_lora_config = _normalize_video_lora_config(video_lora_config)
        self.video_lora_enabled = bool(self.video_lora_config["enabled"])
        if self.video_lora_joint_video_loss and (not False):
            raise ValueError(
                "`video_lora_joint_video_loss=true` requires `video_lora.enabled=true`."
            )
        if (
            self.video_lora_joint_video_loss
            and self.train_video_scheduler.train_sampling_strategy != "pure_noise"
        ):
            raise ValueError(
                "Video-LoRA joint Video loss is defined at the pure-noise endpoint and requires video train_sampling_strategy='pure_noise'."
            )
        if self.video_lora_joint_video_loss and False:
            raise ValueError(
                "Video-LoRA joint Video loss requires the complete target video; set action_only_first_frame_fast_path=false."
            )
        self.use_fixed_video_layers = (
            None if use_fixed_video_layers is None else int(use_fixed_video_layers)
        )
        self.use_video_tokens = bool(False)
        self.video_token_latent_dim = int(video_token_latent_dim)
        if self.action_visible_video_frames <= 0:
            raise ValueError(
                f"`action_visible_video_frames` must be positive, got {self.action_visible_video_frames}"
            )
        if self.action_chunk_size <= 0:
            raise ValueError(f"`action_chunk_size` must be positive, got {self.action_chunk_size}")
        if self.action_expert.action_dim % self.action_chunk_size != 0:
            raise ValueError(
                f"`action_expert.action_dim` must be divisible by action_chunk_size, got {self.action_expert.action_dim} and {self.action_chunk_size}"
            )
        if self.action_clean_first_frame_start_layer is not None:
            if not 0 < self.action_clean_first_frame_start_layer < int(self.mot.num_layers):
                raise ValueError(
                    f"`action_clean_first_frame_start_layer` must be inside the ActionDiT stack [1, {int(self.mot.num_layers) - 1}], got {self.action_clean_first_frame_start_layer}."
                )
            if self.use_fixed_video_layers is not None or False:
                raise ValueError(
                    "Clean-first-frame late ActionDiT layers currently require direct per-layer video K/V (`use_fixed_video_layers=null`, `use_video_tokens=false`)."
                )
        if self.action_video_timestep_cutoff is not None:
            if not 0.0 < self.action_video_timestep_cutoff <= float(action_num_train_timesteps):
                raise ValueError(
                    f"`action_video_timestep_cutoff` must be in (0, {int(action_num_train_timesteps)}], got {self.action_video_timestep_cutoff}."
                )
            if self.action_clean_first_frame_start_layer is not None:
                raise ValueError(
                    "Timestep-gated video conditioning and the clean-first-frame layer split are separate controlled experiments and cannot be enabled together."
                )
        if self.action_clean_first_frame_timestep_cutoff is not None:
            if (
                not 0.0
                < self.action_clean_first_frame_timestep_cutoff
                <= float(action_num_train_timesteps)
            ):
                raise ValueError(
                    f"`action_clean_first_frame_timestep_cutoff` must be in (0, {int(action_num_train_timesteps)}], got {self.action_clean_first_frame_timestep_cutoff}."
                )
            if self.action_clean_first_frame_start_layer is not None:
                raise ValueError(
                    "Clean-first-frame layer and timestep routes are separate controlled experiments and cannot be enabled together."
                )
            if self.action_video_timestep_cutoff is not None:
                raise ValueError(
                    "Clean-first-frame timestep routing and no-video timestep gating cannot be enabled together."
                )
            if self.use_fixed_video_layers is not None or False:
                raise ValueError(
                    "Clean-first-frame timestep routing currently requires direct per-layer video K/V (`use_fixed_video_layers=null`, `use_video_tokens=false`)."
                )
        self.joint_denoising = bool(False)
        self.joint_denoising_mode_probs = _normalize_joint_denoising_mode_probs(
            joint_denoising_mode_probs
        )
        self.joint_denoising_sampler = _normalize_joint_denoising_sampler(joint_denoising_sampler)
        self.raw_action_dim = self.action_expert.action_dim // self.action_chunk_size
        self.set_train_wan(train_wan)
        self.to(self.device)

    def set_train_mode(self, mode: str):
        if mode != "joint":
            raise ValueError("Only joint training is supported")
        self.train_mode = "joint"
        self.train_wan = True

    def set_train_wan(self, train_wan: bool):
        return self.set_train_mode("joint" if train_wan else "action_only")

    def _video_base_model(self):
        return self.video_expert

    def _load_video_base_state_dict(self, state: dict[str, torch.Tensor]):
        base_model = self._video_base_model()
        return base_model.load_state_dict(state, strict=True)
        current_keys = set(base_model.state_dict().keys())
        remapped = {}
        unmatched = []
        for key, value in state.items():
            if key in current_keys:
                remapped[key] = value
                continue
            (prefix, separator, suffix) = key.rpartition(".")
            candidate = f"{prefix}.base_layer.{suffix}" if separator else ""
            if candidate in current_keys:
                remapped[candidate] = value
            else:
                unmatched.append(key)
        if unmatched:
            raise ValueError(
                f"Video base checkpoint has keys that cannot be mapped into the PEFT-wrapped backbone: {unmatched[:8]}"
            )
        load_result = base_model.load_state_dict(remapped, strict=False)
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        missing = [
            key
            for key in getattr(load_result, "missing_keys", [])
            if ".lora_A." not in key and ".lora_B." not in key
        ]
        if unexpected or missing:
            raise ValueError(
                f"Video base checkpoint did not strictly cover the frozen PEFT backbone: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        return load_result

    def _pack_action_tokens(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_chunk_size == 1:
            return action
        if action.ndim != 3:
            raise ValueError(f"`action` must be 3D [B,T,D], got {tuple(action.shape)}")
        (batch_size, horizon, dim) = action.shape
        if dim != self.raw_action_dim:
            raise ValueError(f"Expected raw action dim {self.raw_action_dim}, got {dim}")
        if horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"Action horizon {horizon} must be divisible by action_chunk_size={self.action_chunk_size}"
            )
        return action.reshape(
            batch_size, horizon // self.action_chunk_size, dim * self.action_chunk_size
        )

    def _unpack_action_tokens(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_chunk_size == 1:
            return action
        if action.ndim not in (2, 3):
            raise ValueError(f"`action` must be 2D or 3D, got {tuple(action.shape)}")
        if action.shape[-1] != self.action_expert.action_dim:
            raise ValueError(
                f"Expected packed action dim {self.action_expert.action_dim}, got {action.shape[-1]}"
            )
        leading = action.shape[:-1]
        return action.reshape(
            *leading[:-1], leading[-1] * self.action_chunk_size, self.raw_action_dim
        )

    def _pack_action_is_pad(self, action_is_pad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if action_is_pad is None or self.action_chunk_size == 1:
            return action_is_pad
        if action_is_pad.ndim != 2:
            raise ValueError(f"`action_is_pad` must be 2D [B,T], got {tuple(action_is_pad.shape)}")
        (batch_size, horizon) = action_is_pad.shape
        if horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"Action pad horizon {horizon} must be divisible by action_chunk_size={self.action_chunk_size}"
            )
        return action_is_pad.reshape(
            batch_size, horizon // self.action_chunk_size, self.action_chunk_size
        ).any(dim=2)

    @classmethod
    def from_wan21_14b_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "weights/Wan2.1-I2V-14B-480P",
        dit_checkpoint_path: str | None = None,
        tokenizer_model_id: str | None = None,
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        load_clip_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        action_chunk_size: int = 1,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        video_train_sampling_strategy: str = "shifted_uniform",
        video_train_beta_alpha: float = 7.0,
        video_train_beta_beta: float = 1.0,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        video_lora_joint_video_loss: bool = False,
        action_visible_video_frames: int = 1,
        action_only_first_frame_fast_path: bool = False,
        action_clean_first_frame_start_layer: Optional[int] = None,
        action_clean_first_frame_timestep_cutoff: Optional[float] = None,
        action_first_frame_kv_source: str = "clean_prefill",
        action_video_timestep_cutoff: Optional[float] = None,
        video_lora_config: Optional[dict[str, Any]] = None,
        use_fixed_video_layers: Optional[int] = None,
        use_video_tokens: bool = False,
        video_token_latent_dim: int = 2560,
        joint_denoising: bool = False,
        joint_denoising_mode_probs: Optional[dict[str, float]] = None,
        joint_denoising_sampler: str = "euler",
    ):
        if video_dit_config is None:
            video_dit_config = {}
        if "text_dim" not in video_dit_config:
            video_dit_config = dict(video_dit_config)
            video_dit_config.setdefault("text_dim", 4096)
        components = load_wan21_i2v_14b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            dit_checkpoint_path=dit_checkpoint_path,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
            load_clip_encoder=load_clip_encoder,
        )
        video_expert = components.dit
        normalized_video_lora_config = _normalize_video_lora_config(video_lora_config)
        action_dit_config = dict(action_dit_config or {})
        action_dit_config["use_video_tokens"] = bool(False)
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if not False and int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError(
                "ActionDiT `num_heads` must match video expert for MoT mixed attention."
            )
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError(
                "ActionDiT `attn_head_dim` must match video expert for MoT mixed attention."
            )
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")
        video_expert = _apply_wan_official_video_lora(video_expert, normalized_video_lora_config)
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            use_fixed_video_layers=use_fixed_video_layers,
            use_video_tokens=False,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            clip_encoder=components.clip_encoder,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            video_train_sampling_strategy=video_train_sampling_strategy,
            video_train_beta_alpha=video_train_beta_alpha,
            video_train_beta_beta=video_train_beta_beta,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            video_lora_joint_video_loss=video_lora_joint_video_loss,
            action_chunk_size=action_chunk_size,
            action_visible_video_frames=action_visible_video_frames,
            action_only_first_frame_fast_path=action_only_first_frame_fast_path,
            action_clean_first_frame_start_layer=action_clean_first_frame_start_layer,
            action_clean_first_frame_timestep_cutoff=action_clean_first_frame_timestep_cutoff,
            action_first_frame_kv_source=action_first_frame_kv_source,
            action_video_timestep_cutoff=action_video_timestep_cutoff,
            video_lora_config=normalized_video_lora_config,
            use_fixed_video_layers=use_fixed_video_layers,
            use_video_tokens=False,
            video_token_latent_dim=video_token_latent_dim,
            joint_denoising=False,
            joint_denoising_mode_probs=joint_denoising_mode_probs,
            joint_denoising_sampler=joint_denoising_sampler,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "clip_encoder": components.clip_encoder_path,
            "action_dit_backbone": "SKIPPED_PRETRAIN"
            if skip_dit_load_from_pretrain
            else action_dit_pretrained_path,
        }
        return model

    def _restore_auxiliary_component_dtypes(self):
        """Keep Wan I2V helper modules aligned with the reference generation path."""
        if self.clip_encoder is not None:
            self.clip_encoder.eval().requires_grad_(False).to(
                device=self.device, dtype=torch.float16
            )
        self.vae.eval().requires_grad_(False).to(device=self.device, dtype=torch.float32)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self._restore_auxiliary_component_dtypes()
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return (height, width, num_frames)

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        (ids, mask) = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return (prompt_emb.to(device=self.device), mask)

    def _append_proprio_to_context(
        self, context: torch.Tensor, context_mask: torch.Tensor, proprio: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return (context, context_mask)
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(
        self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        self._restore_auxiliary_component_dtypes()
        videos = [
            video_tensor[i].to(device=self.device, dtype=torch.float32)
            for i in range(video_tensor.shape[0])
        ]
        z = self.vae.encode(
            videos, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        if isinstance(z, list):
            z = torch.stack(z)
        return z.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        self._restore_auxiliary_component_dtypes()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device, dtype=torch.float32)[0].unsqueeze(1)
        z = self.vae.encode(
            [image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _encode_separate_condition_latents(
        self,
        condition_video: torch.Tensor,
        condition_observation_video: Optional[torch.Tensor] = None,
        tiled: bool = False,
    ) -> torch.Tensor:
        """Encode history and the current observation as independent conditions."""
        history_latents = self._encode_video_latents(condition_video, tiled=tiled)
        if condition_observation_video is None:
            return history_latents
        if condition_observation_video.ndim == 4:
            condition_observation_video = condition_observation_video.unsqueeze(0)
        if condition_observation_video.ndim != 5 or condition_observation_video.shape[1] != 3:
            raise ValueError(
                f"`condition_observation_video` must be [B,3,1,H,W] or [3,1,H,W], got {tuple(condition_observation_video.shape)}"
            )
        if condition_observation_video.shape[2] != 1:
            raise ValueError(
                f"`condition_observation_video` must contain exactly one RGB frame, got T={condition_observation_video.shape[2]}"
            )
        observation_latents = self._encode_video_latents(condition_observation_video, tiled=tiled)
        if observation_latents.shape[2] != 1:
            raise ValueError(
                f"One-frame observation VAE encoding must produce one latent frame, got T={observation_latents.shape[2]}"
            )
        if history_latents.shape[0] != observation_latents.shape[0]:
            raise ValueError("History and observation condition batches must match.")
        if history_latents.shape[3:] != observation_latents.shape[3:]:
            raise ValueError(
                f"History and observation condition latent spatial grids must match, got {tuple(history_latents.shape[3:])} vs {tuple(observation_latents.shape[3:])}."
            )
        return torch.cat([history_latents, observation_latents], dim=2)

    @torch.no_grad()
    def _encode_clip_fea_from_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        self._restore_auxiliary_component_dtypes()
        if self.clip_encoder is None:
            raise ValueError(
                "Wan2.1-I2V-14B requires CLIP image features. Load `clip_encoder` or provide `sample['clip_fea']` / `clip_fea` explicitly."
            )
        if video_tensor.ndim != 5:
            raise ValueError(f"`video_tensor` must be [B,C,T,H,W], got {tuple(video_tensor.shape)}")
        videos = [
            video_tensor[i].to(device=self.device, dtype=torch.float16)
            for i in range(video_tensor.shape[0])
        ]
        return self.clip_encoder(videos).to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _encode_clip_fea_from_image(self, input_image: torch.Tensor) -> torch.Tensor:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [B,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        video = input_image.to(device=self.device, dtype=torch.float16).unsqueeze(2)
        return self._encode_clip_fea_from_video(video)

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        self._restore_auxiliary_component_dtypes()
        video_tensor = self.vae.decode(
            latents.to(device=self.device, dtype=torch.float32),
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False, training_stage: Optional[str] = None):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "VPP2 training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(
                f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}"
            )
        if video.shape[1] != 3:
            raise ValueError(
                f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
            )
        (batch_size, _, observed_num_frames, height, width) = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        normalized_stage = None if training_stage is None else str(training_stage).strip().lower()
        use_action_only_fast_path = False and (
            not self.train_wan or ("joint" == "alternating" and normalized_stage == "action")
        )
        declared_num_frames = sample.get("video_num_frames", None)
        if declared_num_frames is None:
            num_frames = observed_num_frames
        elif isinstance(declared_num_frames, torch.Tensor):
            declared_values = declared_num_frames.detach().reshape(-1).cpu().tolist()
            if not declared_values or any(
                (int(value) != int(declared_values[0]) for value in declared_values)
            ):
                raise ValueError(
                    f"`sample['video_num_frames']` must be identical across the batch, got {declared_values}."
                )
            num_frames = int(declared_values[0])
        else:
            num_frames = int(declared_num_frames)
        if num_frames % 4 != 1:
            raise ValueError(f"Logical video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(
                f"Logical video T must be > 1 for action-conditioned training, got T={num_frames}"
            )
        if observed_num_frames <= 0:
            raise ValueError("`sample['video']` must contain at least one decoded frame.")
        if not use_action_only_fast_path and observed_num_frames != num_frames:
            raise ValueError(
                f"A shortened decoded video is only valid for the action-only first-frame fast path: observed T={observed_num_frames}, logical T={num_frames}."
            )
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for VPP2 training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(
                f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
            )
        action_horizon = int(action.shape[1])
        if action_horizon <= 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be positive, got {action_horizon}"
            )
        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    f"`sample['action_is_pad']` shape mismatch: got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )
        action_loss_is_pad = action_is_pad
        action_terminal_mask = sample.get("action_terminal_mask")
        if action_terminal_mask is not None:
            if action_is_pad is None or action_terminal_mask.shape != action_is_pad.shape:
                raise ValueError("action_terminal_mask must match action_is_pad [B,T].")
            terminal = action_terminal_mask.to(device=action_is_pad.device, dtype=torch.bool)
            if bool((terminal & ~action_is_pad.bool()).any()):
                raise ValueError("action_terminal_mask must be a subset of action_is_pad.")
            action_loss_is_pad = action_is_pad.bool() & ~terminal
        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != observed_num_frames:
                raise ValueError(
                    f"`sample['image_is_pad']` shape mismatch: got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {observed_num_frames})"
                )
        input_latents = None
        input_video = None
        if not use_action_only_fast_path:
            input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            input_latents = self._encode_video_latents(input_video, tiled=tiled)
        condition_video = sample.get("condition_video")
        condition_observation_video = sample.get("condition_observation_video")
        if condition_video is None and condition_observation_video is not None:
            raise ValueError("`condition_observation_video` requires `condition_video` history.")
        if condition_video is None:
            if input_video is None:
                input_video = video[:, :, :1].to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                )
                condition_latents = self._encode_video_latents(input_video, tiled=tiled)
            else:
                condition_latents = input_latents[:, :, :1]
            clip_condition_video = input_video
        else:
            if condition_video.ndim != 5 or condition_video.shape[:2] != (batch_size, 3):
                raise ValueError(
                    f"`sample['condition_video']` must be [B,3,T,H,W], got {tuple(condition_video.shape)}."
                )
            if condition_video.shape[-2:] != (height, width):
                raise ValueError(
                    f"Condition and target video spatial sizes must match, got {tuple(condition_video.shape[-2:])} vs {(height, width)}."
                )
            condition_num_frames = int(condition_video.shape[2])
            if condition_num_frames <= 0 or condition_num_frames % 4 != 1:
                raise ValueError(
                    f"Condition temporal length must be positive and satisfy T % 4 == 1, got T={condition_num_frames}."
                )
            condition_input_video = condition_video.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            if condition_observation_video is not None:
                if condition_observation_video.ndim != 5 or condition_observation_video.shape[
                    :2
                ] != (batch_size, 3):
                    raise ValueError(
                        f"`sample['condition_observation_video']` must be [B,3,1,H,W], got {tuple(condition_observation_video.shape)}"
                    )
                if condition_observation_video.shape[2] != 1:
                    raise ValueError(
                        f"`sample['condition_observation_video']` must contain one frame, got T={condition_observation_video.shape[2]}"
                    )
                condition_observation_video = condition_observation_video.to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                )
            condition_latents = self._encode_separate_condition_latents(
                condition_input_video,
                condition_observation_video=condition_observation_video,
                tiled=tiled,
            )
            logical_latent_frames = (num_frames - 1) // int(self.vae.temporal_downsample_factor) + 1
            if condition_latents.shape[2] > logical_latent_frames:
                raise ValueError(
                    f"Condition cannot be longer than the logical video latent sequence, got {condition_latents.shape[2]} > {logical_latent_frames}."
                )
            if input_latents is not None and condition_latents.shape[3:] != input_latents.shape[3:]:
                raise ValueError(
                    f"Condition and target latent spatial grids must match, got {tuple(condition_latents.shape[3:])} vs {tuple(input_latents.shape[3:])}."
                )
            if condition_observation_video is not None and input_latents is not None:
                if condition_latents.shape[2] > input_latents.shape[2]:
                    raise ValueError(
                        f"Separate condition cannot be longer than the target latent sequence, got {condition_latents.shape[2]} > {input_latents.shape[2]}."
                    )
                input_latents = torch.cat(
                    [condition_latents, input_latents[:, :, condition_latents.shape[2] :]], dim=2
                )
            clip_condition_video = condition_input_video[:, :, :1]
        clip_fea = sample.get("clip_fea", None)
        if clip_fea is None:
            clip_fea = self._encode_clip_fea_from_video(clip_condition_video)
        else:
            clip_fea = clip_fea.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        use_proprio = normalized_stage != "video"
        if self.proprio_encoder is not None and use_proprio:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :]
            (context, context_mask) = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if action_loss_is_pad is not None:
            action_loss_is_pad = action_loss_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        action = self._pack_action_tokens(action)
        action_is_pad = self._pack_action_is_pad(action_is_pad)
        action_loss_is_pad = self._pack_action_is_pad(action_loss_is_pad)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "condition_latents": condition_latents,
            "clip_fea": clip_fea,
            "first_frame_latents": None,
            "fuse_vae_embedding_in_latents": False,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_loss_is_pad": action_loss_is_pad,
            "image_is_pad": image_is_pad,
            "num_video_frames": num_frames,
            "action_only_first_frame_fast_path": use_action_only_fast_path,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_visible_video_frames: Optional[int] = None,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        visible_frames = (
            self.action_visible_video_frames
            if action_visible_video_frames is None
            else int(action_visible_video_frames)
        )
        if visible_frames <= 0:
            raise ValueError(
                f"`action_visible_video_frames` must be positive, got {visible_frames}."
            )
        visible_video_tokens = min(video_tokens_per_frame * visible_frames, video_seq_len)
        mask[video_seq_len:, :visible_video_tokens] = True
        return mask

    def _build_action_video_conditioning_mask(
        self, timestep_action: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Select samples whose action queries may read cached video K/V."""
        cutoff = getattr(self, "action_video_timestep_cutoff", None)
        if cutoff is None:
            return None
        if timestep_action.ndim == 0:
            timestep_action = timestep_action.reshape(1)
        if timestep_action.ndim != 1:
            raise ValueError(
                f"`timestep_action` must be scalar or 1D [B] for timestep-gated video conditioning, got shape {tuple(timestep_action.shape)}"
            )
        return timestep_action.to(dtype=torch.float32) >= float(cutoff)

    def _build_action_clean_first_frame_conditioning_mask(
        self, timestep_action: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Select samples that replace noisy-video K/V with clean frame-0 K/V."""
        cutoff = getattr(self, "action_clean_first_frame_timestep_cutoff", None)
        if cutoff is None:
            return None
        if timestep_action.ndim == 0:
            timestep_action = timestep_action.reshape(1)
        if timestep_action.ndim != 1:
            raise ValueError(
                f"`timestep_action` must be scalar or 1D [B] for timestep-gated clean-first-frame conditioning, got shape {tuple(timestep_action.shape)}"
            )
        return timestep_action.to(dtype=torch.float32) < float(cutoff)

    @staticmethod
    def _masked_batch_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Batch mean with inactive samples contributing zero.

        Normalizing by the batch size (not the local active count) keeps every
        active sample equally weighted after DDP averages the per-rank losses;
        with no mask / an all-true mask this is exactly ``values.mean()``.
        """
        if mask is None:
            return values.mean()
        weights = mask.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / values.numel()

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(
                f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}."
            )
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                f"Cannot align `image_is_pad` with video latent steps: num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )
        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad
        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                f"Video-loss mask shape mismatch: mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )
        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    @contextmanager
    def _disable_video_cache_checkpointing(self):
        old_mot_checkpoint = bool(getattr(self.mot, "mot_checkpoint_mixed_attn", False))
        old_video_checkpoint = bool(getattr(self.video_expert, "use_gradient_checkpointing", False))
        self.mot.mot_checkpoint_mixed_attn = False
        if hasattr(self.video_expert, "use_gradient_checkpointing"):
            self.video_expert.use_gradient_checkpointing = False
        try:
            yield
        finally:
            self.mot.mot_checkpoint_mixed_attn = old_mot_checkpoint
            if hasattr(self.video_expert, "use_gradient_checkpointing"):
                self.video_expert.use_gradient_checkpointing = old_video_checkpoint

    def _build_frozen_video_cache(
        self,
        latents: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        condition_latents: torch.Tensor,
        clip_fea: torch.Tensor,
        action_seq_len: int,
        fuse_vae_embedding_in_latents: bool,
        return_video_prediction: bool = False,
        allow_trainable_video: bool = False,
        action_visible_video_frames: Optional[int] = None,
        max_video_layers: Optional[int] = None,
    ):
        if latents.ndim != 5:
            raise ValueError(
                f"Action-only video context must be [B,C,T,H,W], got shape {tuple(latents.shape)}"
            )
        visible_frames = (
            self.action_visible_video_frames
            if action_visible_video_frames is None
            else int(action_visible_video_frames)
        )
        if visible_frames <= 0:
            raise ValueError(
                f"`action_visible_video_frames` must be positive, got {visible_frames}."
            )
        if latents.shape[2] < visible_frames:
            raise ValueError(
                f"Action-only video context requires at least {visible_frames} video latent frames so all action-visible frames exist, got shape {tuple(latents.shape)}"
            )
        if action_seq_len <= 0:
            raise ValueError(f"`action_seq_len` must be positive, got {action_seq_len}")
        lora_graph_requested = allow_trainable_video and bool(False)
        if self.training or lora_graph_requested:
            valid_alternating_cache = "joint" == "alternating" and allow_trainable_video
            valid_lora_cache = "joint" == "action_only" and lora_graph_requested
            if "joint" != "action_only" and (not valid_alternating_cache):
                raise RuntimeError(
                    f"Frozen video K/V cache is only valid for action-only training or an explicit alternating action block; got train_mode={'joint'!r}."
                )
            if valid_lora_cache:
                allowed = {id(parameter) for (_, parameter) in self.video_lora_named_parameters()}
                invalid_trainable = next(
                    (
                        name
                        for (name, parameter) in self.video_expert.named_parameters()
                        if parameter.requires_grad and id(parameter) not in allowed
                    ),
                    None,
                )
                if invalid_trainable is not None:
                    raise RuntimeError(
                        f"Video-LoRA cache permits only PEFT A/B tensors to train; first unexpected trainable parameter is {invalid_trainable!r}."
                    )
                if not any(
                    (
                        parameter.requires_grad
                        for (_, parameter) in self.video_lora_named_parameters()
                    )
                ):
                    raise RuntimeError(
                        "Video LoRA is enabled but no adapter A/B tensor is trainable."
                    )
            elif not valid_alternating_cache:
                trainable_video_parameter = next(
                    (
                        name
                        for (name, parameter) in self.video_expert.named_parameters()
                        if parameter.requires_grad
                    ),
                    None,
                )
                if trainable_video_parameter is not None:
                    raise RuntimeError(
                        f"Frozen video K/V cache requires every video-expert parameter to have requires_grad=False during training; first trainable parameter is {trainable_video_parameter!r}."
                    )
        retain_video_graph = "joint" == "action_only" and lora_graph_requested
        graph_context = nullcontext() if retain_video_graph else torch.no_grad()
        checkpoint_context = (
            nullcontext() if retain_video_graph else self._disable_video_cache_checkpointing()
        )
        with graph_context:
            video_pre = self.video_expert.pre_dit(
                x=latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                clip_fea=clip_fea,
                condition_latents=condition_latents,
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_seq_len,
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
                action_visible_video_frames=visible_frames,
            )
            with checkpoint_context:
                prefill_kwargs = {
                    "video_tokens": video_pre["tokens"],
                    "video_freqs": video_pre["freqs"],
                    "video_t_mod": video_pre["t_mod"],
                    "video_context_payload": {
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    "video_attention_mask": attention_mask[:video_seq_len, :video_seq_len],
                }
                if return_video_prediction:
                    if max_video_layers is not None:
                        raise ValueError(
                            "`return_video_prediction=true` requires all video layers; `max_video_layers` must be null."
                        )
                    (video_kv_cache, final_video_tokens) = self.mot.prefill_video_cache(
                        **prefill_kwargs, return_final_tokens=True
                    )
                    video_prediction = self.video_expert.post_dit(final_video_tokens, video_pre)
                else:
                    video_kv_cache = self.mot.prefill_video_cache(
                        **prefill_kwargs, max_layers=max_video_layers
                    )
                    video_prediction = None
        if not self.training and bool(False):
            with torch.no_grad():
                video_kv_cache = self.mot.materialize_action_video_kv_cache(video_kv_cache)
        if return_video_prediction:
            return (video_kv_cache, attention_mask, video_seq_len, video_prediction)
        return (video_kv_cache, attention_mask, video_seq_len)

    @torch.no_grad()
    def _build_noisy_first_frame_cache_view(
        self,
        *,
        video_kv_cache: list[dict[str, torch.Tensor]],
        video_seq_len: int,
        video_latent_frames: int,
        action_seq_len: int,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, int]:
        """View frame-0 K/V from a complete noisy-video per-layer cache.

        This deliberately does not claim clean-frame equivalence: frame-0 was
        embedded from noisy latent channels concatenated with the I2V condition,
        and deeper frame-0 states were mixed with the remaining noisy video by
        bidirectional self-attention. The returned tensors are prefix views, so
        no second Video Expert prefill or duplicate K/V storage is introduced.
        """
        if getattr(self, "use_fixed_video_layers", None) is not None or bool(False):
            raise ValueError(
                "Noisy first-frame cache reuse requires direct per-layer video K/V (`use_fixed_video_layers=null`, `use_video_tokens=false`)."
            )
        expected_layers = int(self.mot.num_layers)
        if len(video_kv_cache) != expected_layers:
            raise ValueError(
                f"Noisy first-frame cache reuse requires a complete video cache; expected {expected_layers} layers, got {len(video_kv_cache)}."
            )
        if video_latent_frames <= 0:
            raise ValueError(f"`video_latent_frames` must be positive, got {video_latent_frames}.")
        if video_seq_len <= 0 or video_seq_len % video_latent_frames != 0:
            raise ValueError(
                f"Noisy video token count must divide evenly across latent frames; tokens={video_seq_len}, latent_frames={video_latent_frames}."
            )
        first_frame_seq_len = video_seq_len // video_latent_frames
        first_frame_cache: list[dict[str, torch.Tensor]] = []
        cache_device = None
        for layer_idx, layer_cache in enumerate(video_kv_cache):
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"Noisy first-frame cache reuse requires `k` and `v` in video_kv_cache[{layer_idx}]."
                )
            k = layer_cache["k"]
            v = layer_cache["v"]
            if int(k.shape[1]) != video_seq_len or int(v.shape[1]) != video_seq_len:
                raise ValueError(
                    f"Noisy video K/V sequence mismatch at layer {layer_idx}: k={int(k.shape[1])}, v={int(v.shape[1])}, expected={video_seq_len}."
                )
            first_frame_cache.append(
                {"k": k[:, :first_frame_seq_len], "v": v[:, :first_frame_seq_len]}
            )
            if cache_device is None:
                cache_device = k.device
        if cache_device is None:
            raise ValueError("Noisy video cache is empty.")
        first_frame_attention_mask = self._build_mot_attention_mask(
            video_seq_len=first_frame_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=first_frame_seq_len,
            device=cache_device,
            action_visible_video_frames=1,
        )
        return (first_frame_cache, first_frame_attention_mask, first_frame_seq_len)

    def _sample_action_video_latents(
        self,
        *,
        first_frame_latents: torch.Tensor,
        num_video_frames: int,
        generator: Optional[torch.Generator],
        rand_device: str,
    ) -> torch.Tensor:
        if first_frame_latents.ndim != 5 or first_frame_latents.shape[2] < 1:
            raise ValueError(
                f"`first_frame_latents` must be [B,C,T>=1,H,W], got {tuple(first_frame_latents.shape)}."
            )
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(
                f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}."
            )
        if num_video_frames <= 1:
            raise ValueError(
                f"Action-only inference requires more than one RGB frame, got {num_video_frames}."
            )
        if (num_video_frames - 1) % temporal_factor != 0:
            raise ValueError(
                f"Action-only inference RGB transitions must be divisible by `vae.temporal_downsample_factor`: num_video_frames={num_video_frames}, temporal_downsample_factor={temporal_factor}."
            )
        latent_t = (num_video_frames - 1) // temporal_factor + 1
        if latent_t < self.action_visible_video_frames:
            raise ValueError(
                f"Action-only inference requires at least {self.action_visible_video_frames} video latent frames, got {latent_t} from num_video_frames={num_video_frames} and temporal_downsample_factor={temporal_factor}."
            )
        (batch_size, channels, _, latent_h, latent_w) = first_frame_latents.shape
        return torch.randn(
            (batch_size, channels, latent_t, latent_h, latent_w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

    def training_loss(
        self,
        sample,
        tiled: bool = False,
        training_stage: Optional[str] = None,
        video_feature_callback=None,
    ):
        resolved_stage = "joint"
        if training_stage is not None and str(training_stage).strip().lower() != "joint":
            raise ValueError("VPP2 joint training only accepts training_stage='joint'.")
        if video_feature_callback is not None and "joint" != "joint":
            raise ValueError("Video feature collection requires joint training")
        inputs = self.build_inputs(sample, tiled=tiled, training_stage="joint")
        input_latents = inputs["input_latents"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs.get("action_loss_is_pad", inputs["action_is_pad"])
        image_is_pad = inputs["image_is_pad"]
        condition_latents = inputs["condition_latents"]
        clip_fea = inputs["clip_fea"]
        batch_size = int(condition_latents.shape[0])
        if inputs["action_only_first_frame_fast_path"]:
            temporal_factor = int(self.vae.temporal_downsample_factor)
            if temporal_factor <= 0:
                raise ValueError(
                    f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}."
                )
            num_video_frames = int(inputs["num_video_frames"])
            if (num_video_frames - 1) % temporal_factor != 0:
                raise ValueError(
                    f"Action-only first-frame fast path RGB transitions must be divisible by `vae.temporal_downsample_factor`: num_video_frames={num_video_frames}, temporal_downsample_factor={temporal_factor}."
                )
            latent_t = (num_video_frames - 1) // temporal_factor + 1
            if latent_t < self.action_visible_video_frames:
                raise ValueError(
                    f"Action-only first-frame fast path requires at least {self.action_visible_video_frames} video latent frames, got {latent_t}."
                )
            noise_video = torch.randn(
                (
                    batch_size,
                    int(condition_latents.shape[1]),
                    latent_t,
                    int(condition_latents.shape[3]),
                    int(condition_latents.shape[4]),
                ),
                device=condition_latents.device,
                dtype=condition_latents.dtype,
            )
        else:
            if input_latents is None:
                raise RuntimeError("Full video training path requires encoded input latents.")
            noise_video = torch.randn_like(input_latents)
        joint_denoising = bool(False)
        video_loss_mask = None
        action_loss_mask = None
        shared_action_timestep = None
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=condition_latents.dtype
        )
        joint_video_loss = bool(
            "joint" == "action" and getattr(self, "video_lora_joint_video_loss", False)
        )
        if "joint" == "action" and (not joint_video_loss):
            latents = noise_video
            target_video = None
        elif inputs["action_only_first_frame_fast_path"]:
            latents = noise_video
            target_video = None
        else:
            latents = self.train_video_scheduler.add_noise(
                input_latents, noise_video, timestep_video
            )
            target_video = self.train_video_scheduler.training_target(
                input_latents, noise_video, timestep_video
            )
        noise_action = torch.randn_like(action)
        if shared_action_timestep is not None:
            timestep_action = shared_action_timestep.to(dtype=action.dtype)
        else:
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=action.dtype
            )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )
        if target_video is None:
            raise RuntimeError("Joint training requires a video flow-matching target.")
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            clip_fea=clip_fea,
            condition_latents=condition_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            **{"video_feature_callback": video_feature_callback}
            if video_feature_callback is not None
            else {},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = self._masked_batch_mean(loss_video_per_sample * video_weight, video_loss_mask)
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device, dtype=action_loss_token.dtype
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = self._masked_batch_mean(
            action_loss_per_sample * action_weight, action_loss_mask
        )
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return (loss_total, loss_dict)

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        clean_first_frame_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
        clean_first_frame_attention_mask: Optional[torch.Tensor] = None,
        clean_first_frame_video_seq_len: Optional[int] = None,
        clean_first_frame_start_layer: Optional[int] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            clean_first_frame_kv_cache=clean_first_frame_kv_cache,
            clean_first_frame_attention_mask=clean_first_frame_attention_mask,
            clean_first_frame_video_seq_len=clean_first_frame_video_seq_len,
            clean_first_frame_start_layer=clean_first_frame_start_layer,
            action_video_conditioning_mask=self._build_action_video_conditioning_mask(
                timestep_action
            ),
            action_clean_first_frame_conditioning_mask=self._build_action_clean_first_frame_conditioning_mask(
                timestep_action
            ),
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        condition_video: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        video_seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        num_video_frames: int = 17,
        return_one_step_video: bool = False,
        return_full_video: bool = False,
        full_video_num_inference_steps: Optional[int] = None,
        condition_observation_video: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        self.eval()
        allowed_action_modes = {"bidirectional", "first_frame_causal", "three_frame"}
        if (
            str(getattr(self.video_expert, "video_attention_mask_mode", ""))
            not in allowed_action_modes
        ):
            raise ValueError(
                f"`infer_action` requires `video_attention_mask_mode` in {sorted(allowed_action_modes)}."
            )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        (_, _, height, width) = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(
                    f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        action_generator = (
            None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        )
        video_generator = (
            None
            if video_seed is None and seed is None
            else torch.Generator(device=rand_device).manual_seed(
                seed if video_seed is None else video_seed
            )
        )
        if action_horizon % self.action_chunk_size != 0:
            raise ValueError(
                f"`action_horizon` must be divisible by action_chunk_size={self.action_chunk_size}, got {action_horizon}"
            )
        action_token_horizon = action_horizon // self.action_chunk_size
        latents_action = torch.randn(
            (1, action_token_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if condition_video is None:
            if condition_observation_video is not None:
                raise ValueError(
                    "`condition_observation_video` requires `condition_video` history."
                )
            condition_latents = self._encode_input_image_latents_tensor(
                input_image=input_image, tiled=tiled
            )
            clip_fea = self._encode_clip_fea_from_image(input_image)
        else:
            if condition_video.ndim == 4:
                condition_video = condition_video.unsqueeze(0)
            if condition_video.ndim != 5 or condition_video.shape[:2] != (1, 3):
                raise ValueError(
                    f"`condition_video` must be [1,3,T,H,W] or [3,T,H,W], got {tuple(condition_video.shape)}."
                )
            if condition_video.shape[-2:] != (height, width):
                raise ValueError(
                    f"Condition and input image spatial sizes must match, got {tuple(condition_video.shape[-2:])} vs {(height, width)}."
                )
            condition_num_frames = int(condition_video.shape[2])
            if condition_num_frames <= 0 or condition_num_frames % 4 != 1:
                raise ValueError(
                    f"Condition temporal length must be positive and satisfy T % 4 == 1, got T={condition_num_frames}."
                )
            condition_video = condition_video.to(device=self.device, dtype=self.torch_dtype)
            if condition_observation_video is not None:
                if condition_observation_video.ndim == 4:
                    condition_observation_video = condition_observation_video.unsqueeze(0)
                if condition_observation_video.shape != (1, 3, 1, height, width):
                    raise ValueError(
                        f"`condition_observation_video` must be [1,3,1,H,W] with the same spatial size as the input image, got {tuple(condition_observation_video.shape)}"
                    )
                condition_observation_video = condition_observation_video.to(
                    device=self.device, dtype=self.torch_dtype
                )
            condition_latents = self._encode_separate_condition_latents(
                condition_video,
                condition_observation_video=condition_observation_video,
                tiled=tiled,
            )
            clip_fea = self._encode_clip_fea_from_video(condition_video[:, :, :1])
        first_frame_latents = condition_latents[:, :, :1]
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and (not use_context):
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            (context, context_mask) = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            (context, context_mask) = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )
        latents_video = self._sample_action_video_latents(
            first_frame_latents=condition_latents,
            num_video_frames=num_video_frames,
            generator=video_generator,
            rand_device=rand_device,
        )
        timestep_video = torch.full(
            (latents_video.shape[0],),
            float(self.train_video_scheduler.num_train_timesteps),
            dtype=condition_latents.dtype,
            device=self.device,
        )
        capture_video_prediction = return_one_step_video or return_full_video
        clean_start_layer = getattr(self, "action_clean_first_frame_start_layer", None)
        clean_timestep_cutoff = getattr(self, "action_clean_first_frame_timestep_cutoff", None)
        first_frame_kv_source = getattr(self, "action_first_frame_kv_source", "clean_prefill")
        first_frame_route_enabled = (
            clean_start_layer is not None or clean_timestep_cutoff is not None
        )
        video_cache_result = self._build_frozen_video_cache(
            latents=latents_video,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            condition_latents=condition_latents,
            clip_fea=clip_fea,
            action_seq_len=latents_action.shape[1],
            fuse_vae_embedding_in_latents=fuse_flag,
            return_video_prediction=capture_video_prediction,
            max_video_layers=None
            if capture_video_prediction
            or (first_frame_route_enabled and first_frame_kv_source == "noisy_cache_prefix")
            else clean_start_layer,
        )
        video_prediction = None
        if capture_video_prediction:
            (video_kv_cache, attention_mask, video_seq_len, video_prediction) = video_cache_result
        else:
            (video_kv_cache, attention_mask, video_seq_len) = video_cache_result
        del video_cache_result
        clean_cache_kwargs = {}
        if first_frame_route_enabled:
            if first_frame_kv_source == "noisy_cache_prefix":
                (
                    clean_first_frame_kv_cache,
                    clean_first_frame_attention_mask,
                    clean_first_frame_video_seq_len,
                ) = self._build_noisy_first_frame_cache_view(
                    video_kv_cache=video_kv_cache,
                    video_seq_len=video_seq_len,
                    video_latent_frames=int(latents_video.shape[2]),
                    action_seq_len=int(latents_action.shape[1]),
                )
            else:
                (
                    clean_first_frame_kv_cache,
                    clean_first_frame_attention_mask,
                    clean_first_frame_video_seq_len,
                ) = self._build_frozen_video_cache(
                    latents=first_frame_latents,
                    timestep_video=torch.zeros_like(timestep_video),
                    context=context,
                    context_mask=context_mask,
                    condition_latents=first_frame_latents,
                    clip_fea=clip_fea,
                    action_seq_len=latents_action.shape[1],
                    fuse_vae_embedding_in_latents=fuse_flag,
                    action_visible_video_frames=1,
                )
            clean_cache_kwargs = {
                "clean_first_frame_kv_cache": clean_first_frame_kv_cache,
                "clean_first_frame_attention_mask": clean_first_frame_attention_mask,
                "clean_first_frame_video_seq_len": clean_first_frame_video_seq_len,
            }
            if clean_start_layer is not None:
                clean_cache_kwargs["clean_first_frame_start_layer"] = clean_start_layer
        (infer_timesteps_action, infer_deltas_action) = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )
            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                **clean_cache_kwargs,
            )
            pred_action = pred_action_posi
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )
        result = {
            "action": self._unpack_action_tokens(latents_action)[0]
            .detach()
            .to(device="cpu", dtype=torch.float32)
        }
        del video_kv_cache
        if first_frame_route_enabled:
            del clean_first_frame_kv_cache
        if capture_video_prediction:
            if video_prediction is None:
                raise RuntimeError("One-step video capture did not return a video prediction.")
            (one_step_timesteps, one_step_deltas) = (
                self.infer_video_scheduler.build_inference_schedule(
                    num_inference_steps=1,
                    device=self.device,
                    dtype=latents_video.dtype,
                    shift_override=sigma_shift,
                )
            )
            if len(one_step_timesteps) != 1 or len(one_step_deltas) != 1:
                raise RuntimeError(
                    f"One-step video capture returned an unexpected scheduler length: timesteps={len(one_step_timesteps)}, deltas={len(one_step_deltas)}."
                )
            one_step_timestep = one_step_timesteps[0].to(
                device=timestep_video.device, dtype=timestep_video.dtype
            )
            if not torch.allclose(timestep_video, one_step_timestep.expand_as(timestep_video)):
                raise RuntimeError(
                    f"Action video-cache timestep does not match standalone one-step video scheduler: cache={timestep_video.detach().cpu().tolist()} schedule={float(one_step_timestep.detach().cpu())}."
                )
            one_step_video_latents = self.infer_video_scheduler.step(
                video_prediction, one_step_deltas[0], latents_video
            )
            full_video_inference_steps = int(full_video_num_inference_steps or num_inference_steps)
            if return_full_video and full_video_inference_steps <= 0:
                raise ValueError(
                    f"`full_video_num_inference_steps` must be positive, got {full_video_inference_steps}."
                )
            if return_full_video:
                (video_timesteps, video_deltas) = (
                    self.infer_video_scheduler.build_inference_schedule(
                        num_inference_steps=full_video_inference_steps,
                        device=self.device,
                        dtype=latents_video.dtype,
                        shift_override=sigma_shift,
                    )
                )
                if (
                    len(video_timesteps) != full_video_inference_steps
                    or len(video_deltas) != full_video_inference_steps
                ):
                    raise RuntimeError(
                        f"Full video trajectory capture returned an unexpected scheduler length: expected={full_video_inference_steps}, timesteps={len(video_timesteps)}, deltas={len(video_deltas)}."
                    )
                scheduled_timestep = video_timesteps[0].to(
                    device=timestep_video.device, dtype=timestep_video.dtype
                )
                if not torch.allclose(timestep_video, scheduled_timestep.expand_as(timestep_video)):
                    raise RuntimeError(
                        f"Action video-cache timestep does not match full-video scheduler: cache={timestep_video.detach().cpu().tolist()} schedule={float(scheduled_timestep.detach().cpu())}."
                    )
                trajectory_latents = self.infer_video_scheduler.step(
                    video_prediction, video_deltas[0], latents_video
                )
                for step_t_video, step_delta_video in zip(video_timesteps[1:], video_deltas[1:]):
                    timestep_video_next = step_t_video.unsqueeze(0).to(
                        dtype=trajectory_latents.dtype, device=self.device
                    )
                    next_cache_result = self._build_frozen_video_cache(
                        latents=trajectory_latents,
                        timestep_video=timestep_video_next,
                        context=context,
                        context_mask=context_mask,
                        condition_latents=condition_latents,
                        clip_fea=clip_fea,
                        action_seq_len=latents_action.shape[1],
                        fuse_vae_embedding_in_latents=fuse_flag,
                        return_video_prediction=True,
                    )
                    (_, _, _, next_video_prediction) = next_cache_result
                    del next_cache_result
                    trajectory_latents = self.infer_video_scheduler.step(
                        next_video_prediction, step_delta_video, trajectory_latents
                    )
            result["video_one_step"] = self._decode_latents(one_step_video_latents, tiled=tiled)
            result["video"] = result["video_one_step"]
            result["video_inference_steps"] = 1
            result["video_one_step_source"] = "action_context_cache_forward"
            if return_full_video:
                result["video_full"] = self._decode_latents(trajectory_latents, tiled=tiled)
                result["video_full_inference_steps"] = full_video_inference_steps
                result["video_trajectory_num_inference_steps"] = full_video_inference_steps
                result["video_full_shares_action_context_initial_noise"] = True
                result["video_full_shares_action_context_initial_prediction"] = True
        return result

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {"mot": self.mot.state_dict(), "step": step, "torch_dtype": str(self.torch_dtype)}
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @staticmethod
    def _load_video_checkpoint_state(path):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Video checkpoint must contain a state dictionary, got {type(payload)}"
            )
        mot = payload.get("mot")
        if isinstance(mot, dict):
            payload = {
                key[len("mixtures.video.") :]: value
                for (key, value) in mot.items()
                if key.startswith("mixtures.video.")
            }
        for key in ("dit", "state_dict", "module", "model_state"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
        state = {key: value for (key, value) in payload.items() if isinstance(value, torch.Tensor)}
        if not state:
            raise ValueError(f"No video tensors found in checkpoint: {path}")
        return state

    def load_checkpoint(
        self,
        path,
        optimizer=None,
        video_checkpoint_override=None,
        load_video_from_action_checkpoint=True,
    ):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if "mot" not in payload:
            raise ValueError("Resume requires a full training checkpoint containing mot")
        self.mot.load_state_dict(payload["mot"], strict=True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload.get("step")

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
