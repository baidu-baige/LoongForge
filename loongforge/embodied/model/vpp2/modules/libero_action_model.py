# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO frozen-video Action training; numerical blocks shared with VPP2."""

from contextlib import nullcontext
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from .vpp2 import VPP2
from .checkpoint_compat import ACTION_FORMAT, action_config


class LiberoActionModel(VPP2):
    supports_action_only_fast_path = True

    def __init__(self, *args, **kwargs):
        kwargs["train_wan"] = False
        super().__init__(*args, **kwargs)

    def set_train_mode(self, mode):
        if mode != "action_only":
            raise ValueError("LIBERO Action training requires action_only")
        self.train_mode = mode
        self.train_wan = False

    def save_action_checkpoint(
        self, path, video_checkpoint_path, step=None, resolved_cfg=None, **kwargs
    ):
        config = (
            OmegaConf.to_container(resolved_cfg, resolve=True)
            if isinstance(resolved_cfg, DictConfig)
            else resolved_cfg
        )
        video = Path(video_checkpoint_path).resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        payload = dict(
            format=ACTION_FORMAT,
            step=step,
            action_expert=self.action_expert.state_dict(),
            proprio_encoder=self.proprio_encoder.state_dict(),
            video_checkpoint=dict(path=str(video), size_bytes=video.stat().st_size),
            resolved_config=config,
        )
        torch.save(payload, path)
        return payload

    def load_checkpoint(
        self,
        path,
        optimizer=None,
        video_checkpoint_override=None,
        load_video_from_action_checkpoint=True,
    ):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        action_config(payload)
        if load_video_from_action_checkpoint:
            reference = video_checkpoint_override or payload["video_checkpoint"]["path"]
            video = Path(reference)
            if not video.is_absolute():
                video = Path(path).resolve().parent / video
            self._load_video_base_state_dict(self._load_video_checkpoint_state(video))
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        return payload.get("step")

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
        batch_size, _, observed_num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        normalized_stage = None if training_stage is None else str(training_stage).strip().lower()
        use_action_only_fast_path = self.action_only_first_frame_fast_path and (
            not False or ("action_only" == "alternating" and normalized_stage == "action")
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
            context, context_mask = self._append_proprio_to_context(
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
        if self.training or False:
            valid_alternating_cache = "action_only" == "alternating" and allow_trainable_video
            valid_lora_cache = "action_only" == "action_only" and False
            if valid_lora_cache:
                allowed = {id(parameter) for _, parameter in self.video_lora_named_parameters()}
                invalid_trainable = next(
                    (
                        name
                        for name, parameter in self.video_expert.named_parameters()
                        if parameter.requires_grad and id(parameter) not in allowed
                    ),
                    None,
                )
                if invalid_trainable is not None:
                    raise RuntimeError(
                        f"Video-LoRA cache permits only PEFT A/B tensors to train; first unexpected trainable parameter is {invalid_trainable!r}."
                    )
                if not any(
                    (parameter.requires_grad for _, parameter in self.video_lora_named_parameters())
                ):
                    raise RuntimeError(
                        "Video LoRA is enabled but no adapter A/B tensor is trainable."
                    )
            else:
                trainable_video_parameter = next(
                    (
                        name
                        for name, parameter in self.video_expert.named_parameters()
                        if parameter.requires_grad
                    ),
                    None,
                )
                if trainable_video_parameter is not None:
                    raise RuntimeError(
                        f"Frozen video K/V cache requires every video-expert parameter to have requires_grad=False during training; first trainable parameter is {trainable_video_parameter!r}."
                    )
        retain_video_graph = "action_only" == "action_only" and False
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
                    video_kv_cache, final_video_tokens = self.mot.prefill_video_cache(
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

    def _training_loss_action_only(
        self,
        latents: torch.Tensor,
        noisy_action: torch.Tensor,
        target_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        condition_latents: torch.Tensor,
        clip_fea: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        allow_trainable_video: bool = False,
        target_video: Optional[torch.Tensor] = None,
        image_is_pad: Optional[torch.Tensor] = None,
    ):
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        clean_start_layer = getattr(self, "action_clean_first_frame_start_layer", None)
        clean_timestep_cutoff = getattr(self, "action_clean_first_frame_timestep_cutoff", None)
        first_frame_kv_source = getattr(self, "action_first_frame_kv_source", "clean_prefill")
        first_frame_route_enabled = None is not None or None is not None
        joint_video_loss = bool(False)
        video_cache_result = self._build_frozen_video_cache(
            latents=latents,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            condition_latents=condition_latents,
            clip_fea=clip_fea,
            action_seq_len=int(action_pre["tokens"].shape[1]),
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            return_video_prediction=False,
            allow_trainable_video=allow_trainable_video,
            max_video_layers=None,
        )
        pred_video = None
        video_kv_cache, attention_mask, video_seq_len = video_cache_result
        clean_cache_kwargs = {}
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
            action_video_conditioning_mask=self._build_action_video_conditioning_mask(
                timestep_action
            ),
            **clean_cache_kwargs,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)
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
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_action * loss_action
        weighted_video_loss = 0.0
        loss_dict = {
            "loss_video": weighted_video_loss,
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "action_timestep": float(timestep_action.detach().float().mean().item()),
            "action_weight": float(action_weight.detach().float().mean().item()),
        }
        return (loss_total, loss_dict)

    def training_loss(
        self,
        sample,
        tiled: bool = False,
        training_stage: Optional[str] = None,
        video_feature_callback=None,
    ):
        resolved_stage = "action"
        if video_feature_callback is not None and "action" != "joint":
            raise ValueError("Video feature collection requires joint training")
        inputs = self.build_inputs(sample, tiled=tiled, training_stage="action")
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
        timestep_video = torch.full(
            (batch_size,),
            float(self.train_video_scheduler.num_train_timesteps),
            device=self.device,
            dtype=condition_latents.dtype,
        )
        joint_video_loss = bool("action" == "action" and False)
        latents = noise_video
        target_video = None
        noise_action = torch.randn_like(action)
        if shared_action_timestep is not None:
            timestep_action = shared_action_timestep.to(dtype=action.dtype)
        else:
            timestep_action = self.train_action_scheduler.sample_training_t(
                # BF16 rounds values near 1000 to the endpoint, where the
                # flow weighting function is exactly zero. Preserve sampled
                # continuous timesteps in FP32 even with BF16 model weights.
                batch_size=batch_size, device=self.device, dtype=torch.float32
            )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )
        return self._training_loss_action_only(
            latents=latents,
            noisy_action=noisy_action,
            target_action=target_action,
            target_video=target_video,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            action_is_pad=action_is_pad,
            image_is_pad=image_is_pad,
            condition_latents=condition_latents,
            clip_fea=clip_fea,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            allow_trainable_video="action_only" == "alternating" or bool(False),
        )
