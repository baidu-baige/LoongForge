# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from PIL import Image

from .logging_utils import get_logger

from .helpers.loader import load_wan21_i2v_14b_components
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class Wan21VideoModel(torch.nn.Module):
    """Standalone Wan2.1-I2V model for video-only flow-matching training."""

    def __init__(
        self,
        dit,
        vae,
        text_encoder=None,
        tokenizer=None,
        clip_encoder=None,
        text_dim: int = 4096,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 3.0,
        video_num_train_timesteps: int = 1000,
        video_train_sampling_strategy: str = "beta",
        video_train_beta_alpha: float = 7.0,
        video_train_beta_beta: float = 1.0,
        condition_clip_frame: str = "first",
        condition_latent_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.clip_encoder = clip_encoder
        self.text_dim = int(text_dim)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.train_mode = "video_only"
        self.condition_clip_frame = str(condition_clip_frame).strip().lower()
        if self.condition_clip_frame not in {"first", "latest"}:
            raise ValueError(
                "`condition_clip_frame` must be one of: first, latest; got "
                f"{condition_clip_frame!r}."
            )
        self.condition_latent_loss_weight = float(condition_latent_loss_weight)
        if self.condition_latent_loss_weight < 0.0:
            raise ValueError(
                "`condition_latent_loss_weight` must be non-negative, got "
                f"{condition_latent_loss_weight}."
            )

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
            train_sampling_strategy=video_train_sampling_strategy,
            beta_alpha=video_train_beta_alpha,
            beta_beta=video_train_beta_beta,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.dit.to(device=self.device, dtype=self.torch_dtype)
        if self.text_encoder is not None:
            self.text_encoder.to(device=self.device, dtype=self.torch_dtype)
        self._restore_auxiliary_component_dtypes()

    def set_train_mode(self, train_mode: str):
        normalized = str(train_mode).strip().lower()
        if normalized != "video_only":
            raise ValueError(
                f"Wan21VideoModel only supports train_mode=video_only, got {normalized}"
            )
        self.train_mode = normalized
        return self

    def _restore_auxiliary_component_dtypes(self):
        if self.clip_encoder is not None:
            self.clip_encoder.eval().requires_grad_(False).to(
                device=self.device,
                dtype=self.torch_dtype,
            )
        self.vae.eval().requires_grad_(False).to(
            device=self.device,
            dtype=self.torch_dtype,
        )

    @staticmethod
    def _check_resize_height_width(height: int, width: int, num_frames: int):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires a loaded text encoder/tokenizer; "
                "otherwise provide cached context/context_mask."
            )
        ids, mask = self.tokenizer(
            prompt,
            return_mask=True,
            add_special_tokens=True,
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        sequence_lengths = mask.gt(0).sum(dim=1).long()
        for index, length in enumerate(sequence_lengths):
            prompt_emb[index, length:] = 0
        return prompt_emb.to(device=self.device), torch.ones_like(mask)

    @torch.no_grad()
    def _encode_video_latents(
        self,
        video_tensor,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        self._restore_auxiliary_component_dtypes()
        # DeepSpeed ZeRO-3 may cast/shard auxiliary modules together with the
        # main model.  Feed the VAE using its actual parameter dtype to avoid
        # float32/bfloat16 convolution mismatches after accelerator.prepare.
        # The VAE wrapper may expose auxiliary float32 parameters before the
        # encoder, while ZeRO/autocast can keep convolution weights in bf16.
        # Match the actual first encoder parameter used by conv3d.
        vae_dtype = self.torch_dtype
        videos = [
            video_tensor[index].to(device=self.device, dtype=vae_dtype)
            for index in range(video_tensor.shape[0])
        ]
        latents = self.vae.encode(
            videos,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(latents, list):
            latents = torch.stack(latents)
        return latents.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        self._restore_auxiliary_component_dtypes()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                "`input_image` must have shape [1,3,H,W] or [3,H,W], "
                f"got {tuple(input_image.shape)}"
            )
        vae_dtype = self.torch_dtype
        image_video = input_image.to(device=self.device, dtype=vae_dtype)[0].unsqueeze(1)
        latents = self.vae.encode(
            [image_video],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(latents, list):
            latents = torch.stack(latents)
        return latents.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _encode_separate_condition_latents(
        self,
        condition_video: torch.Tensor,
        condition_observation_video: Optional[torch.Tensor] = None,
        tiled: bool = False,
    ) -> torch.Tensor:
        """Encode history and the observation frame as separate VAE inputs.

        The history prefix is encoded as a normal ``4k+1`` video (two latent
        frames for the current five-RGB-frame contract).  The current
        observation is encoded independently as a one-frame image, yielding
        one additional clean latent.  Keeping this operation separate from
        the full target VAE encode makes train and inference use the same
        three clean condition latents.
        """
        history_latents = self._encode_video_latents(condition_video, tiled=tiled)
        if condition_observation_video is None:
            return history_latents
        if condition_observation_video.ndim == 4:
            condition_observation_video = condition_observation_video.unsqueeze(0)
        if condition_observation_video.ndim != 5 or condition_observation_video.shape[1] != 3:
            raise ValueError(
                "`condition_observation_video` must be [B,3,1,H,W] or [3,1,H,W], got "
                f"{tuple(condition_observation_video.shape)}"
            )
        if condition_observation_video.shape[2] != 1:
            raise ValueError(
                "`condition_observation_video` must contain exactly one RGB frame, got "
                f"T={condition_observation_video.shape[2]}"
            )
        observation_latents = self._encode_video_latents(
            condition_observation_video,
            tiled=tiled,
        )
        if observation_latents.shape[2] != 1:
            raise ValueError(
                "One-frame observation VAE encoding must produce one latent frame, got "
                f"T={observation_latents.shape[2]}"
            )
        if history_latents.shape[0] != observation_latents.shape[0]:
            raise ValueError("History and observation condition batches must match.")
        if history_latents.shape[3:] != observation_latents.shape[3:]:
            raise ValueError(
                "History and observation condition latent spatial grids must match, got "
                f"{tuple(history_latents.shape[3:])} vs {tuple(observation_latents.shape[3:])}."
            )
        return torch.cat([history_latents, observation_latents], dim=2)

    @torch.no_grad()
    def _encode_clip_fea_from_video(self, video_tensor: torch.Tensor):
        self._restore_auxiliary_component_dtypes()
        if self.clip_encoder is None:
            raise ValueError(
                "Wan2.1-I2V-14B requires CLIP image features; load clip_encoder "
                "or provide sample['clip_fea']."
            )
        if video_tensor.ndim != 5:
            raise ValueError(f"`video_tensor` must be [B,C,T,H,W], got {tuple(video_tensor.shape)}")
        clip_dtype = self.torch_dtype
        videos = [
            video_tensor[index].to(device=self.device, dtype=clip_dtype)
            for index in range(video_tensor.shape[0])
        ]
        return self.clip_encoder(videos).to(
            device=self.device,
            dtype=self.torch_dtype,
        )

    @torch.no_grad()
    def _encode_clip_fea_from_image(self, input_image: torch.Tensor):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        return self._encode_clip_fea_from_video(input_image.unsqueeze(2))

    def _select_clip_condition_video(
        self,
        condition_video: torch.Tensor,
    ) -> torch.Tensor:
        if condition_video.ndim != 5:
            raise ValueError(
                f"`condition_video` must be [B,3,T,H,W], got {tuple(condition_video.shape)}"
            )
        if self.condition_clip_frame == "first":
            return condition_video[:, :, :1]
        return condition_video[:, :, -1:]

    @torch.no_grad()
    def _decode_latents(
        self,
        latents,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        self._restore_auxiliary_component_dtypes()
        video_tensor = self.vae.decode(
            latents.to(device=self.device, dtype=torch.float32),
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(video_tensor, list):
            video_tensor = torch.stack(video_tensor)
        if video_tensor.ndim != 5 or video_tensor.shape[0] != 1:
            raise ValueError(
                f"Decoded video must have shape [1,C,T,H,W], got {tuple(video_tensor.shape)}"
            )
        video_tensor = video_tensor[0].detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        return [
            Image.fromarray(video_tensor[:, frame].permute(1, 2, 0).numpy())
            for frame in range(video_tensor.shape[1])
        ]

    def build_inputs(self, sample, tiled: bool = False):
        if "video" not in sample:
            raise ValueError("Wan21VideoModel training requires sample['video'].")
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("Wan21VideoModel training requires cached context/context_mask.")

        video = sample["video"]
        context = sample["context"]
        context_mask = sample["context_mask"]
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"`sample['video']` must be [B,3,T,H,W], got {tuple(video.shape)}")
        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames <= 1 or num_frames % 4 != 1:
            raise ValueError(
                f"Video temporal length must be greater than 1 and satisfy T % 4 == 1, got {num_frames}"
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                "`context/context_mask` must be [B,L,D]/[B,L], "
                f"got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[:2] != context_mask.shape or context.shape[0] != batch_size:
            raise ValueError("Cached text context batch/sequence dimensions do not match.")

        input_video = video.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        condition_video = sample.get("condition_video")
        if condition_video is None:
            condition_latents = input_latents[:, :, :1]
            clip_condition_video = input_video[:, :, :1]
        else:
            if condition_video.ndim != 5 or condition_video.shape[:2] != (
                batch_size,
                3,
            ):
                raise ValueError(
                    "`sample['condition_video']` must be [B,3,T,H,W], got "
                    f"{tuple(condition_video.shape)}"
                )
            if condition_video.shape[-2:] != (height, width):
                raise ValueError(
                    "Condition and target video spatial sizes must match, got "
                    f"{tuple(condition_video.shape[-2:])} vs {(height, width)}."
                )
            condition_num_frames = int(condition_video.shape[2])
            if condition_num_frames <= 1 or condition_num_frames % 4 != 1:
                raise ValueError(
                    "Condition temporal length must be > 1 and satisfy T % 4 == 1, "
                    f"got T={condition_num_frames}."
                )
            condition_video = condition_video.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            condition_observation_video = sample.get("condition_observation_video")
            if condition_observation_video is not None:
                if condition_observation_video.ndim != 5 or condition_observation_video.shape[
                    :2
                ] != (
                    batch_size,
                    3,
                ):
                    raise ValueError(
                        "`sample['condition_observation_video']` must be [B,3,1,H,W], got "
                        f"{tuple(condition_observation_video.shape)}"
                    )
                if condition_observation_video.shape[2] != 1:
                    raise ValueError(
                        "`sample['condition_observation_video']` must contain one frame, got "
                        f"T={condition_observation_video.shape[2]}"
                    )
                condition_observation_video = condition_observation_video.to(
                    device=self.device,
                    dtype=self.torch_dtype,
                    non_blocking=True,
                )
            condition_latents = self._encode_separate_condition_latents(
                condition_video,
                condition_observation_video=condition_observation_video,
                tiled=tiled,
            )
            if condition_latents.shape[3:] != input_latents.shape[3:]:
                raise ValueError(
                    "History condition and target must produce the same latent spatial "
                    "grid for direct channel conditioning, got "
                    f"{tuple(condition_latents.shape[2:])} vs "
                    f"{tuple(input_latents.shape[2:])}."
                )
            if condition_latents.shape[2] > input_latents.shape[2]:
                raise ValueError(
                    "History condition cannot be longer than the target latent sequence, "
                    f"got {condition_latents.shape[2]} > {input_latents.shape[2]}."
                )
            clip_condition_video = self._select_clip_condition_video(condition_video)
            if condition_observation_video is not None:
                input_latents = torch.cat(
                    [
                        condition_latents,
                        input_latents[:, :, condition_latents.shape[2] :],
                    ],
                    dim=2,
                )
        clip_fea = sample.get("clip_fea")
        if clip_fea is None:
            clip_fea = self._encode_clip_fea_from_video(clip_condition_video)
        else:
            clip_fea = clip_fea.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is not None:
            if image_is_pad.shape != (batch_size, num_frames):
                raise ValueError(
                    "`image_is_pad` must match [B,T], "
                    f"got {tuple(image_is_pad.shape)} vs {(batch_size, num_frames)}"
                )
            image_is_pad = image_is_pad.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )

        return {
            "input_latents": input_latents,
            "condition_latents": condition_latents,
            "clip_fea": clip_fea,
            "context": context.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            ),
            "context_mask": context_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            ),
            "image_is_pad": image_is_pad,
        }

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        condition_latent_frames: int = 0,
    ):
        video_loss_token = F.mse_loss(
            pred_video.float(),
            target_video.float(),
            reduction="none",
        ).mean(dim=(1, 3, 4))
        condition_latent_frames = int(condition_latent_frames)
        if condition_latent_frames < 0 or condition_latent_frames > video_loss_token.shape[1]:
            raise ValueError(
                "`condition_latent_frames` must be in [0, latent_frames], got "
                f"{condition_latent_frames} for latent_frames={video_loss_token.shape[1]}"
            )
        token_weight = torch.ones(
            (video_loss_token.shape[1],),
            device=video_loss_token.device,
            dtype=video_loss_token.dtype,
        )
        if condition_latent_frames:
            token_weight[:condition_latent_frames] = self.condition_latent_loss_weight

        if image_is_pad is None:
            weighted = video_loss_token * token_weight.unsqueeze(0)
            return weighted.sum(dim=1) / token_weight.sum().clamp(min=1.0)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(
                f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}"
            )
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align image padding with video latents: "
                f"frames={image_is_pad.shape[1]}, temporal_factor={temporal_factor}"
            )
        latent_tail_is_pad = (
            image_is_pad[:, 1:]
            .reshape(
                image_is_pad.shape[0],
                -1,
                temporal_factor,
            )
            .all(dim=2)
        )
        video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        if video_is_pad.shape != video_loss_token.shape:
            raise ValueError(
                "Video loss padding mask mismatch: "
                f"mask={tuple(video_is_pad.shape)}, loss={tuple(video_loss_token.shape)}"
            )
        valid = (~video_is_pad).to(
            device=video_loss_token.device,
            dtype=video_loss_token.dtype,
        )
        weighted_valid = valid * token_weight.unsqueeze(0)
        return (video_loss_token * weighted_valid).sum(dim=1) / weighted_valid.sum(dim=1).clamp(
            min=1.0
        )

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        noise = torch.randn_like(input_latents)
        timestep = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        noisy_latents = self.train_video_scheduler.add_noise(
            input_latents,
            noise,
            timestep,
        )
        target = self.train_video_scheduler.training_target(
            input_latents,
            noise,
            timestep,
        )
        pred = self.dit(
            x=noisy_latents,
            timestep=timestep,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            clip_fea=inputs["clip_fea"],
            condition_latents=inputs["condition_latents"],
        )
        loss_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred,
            target_video=target,
            image_is_pad=inputs["image_is_pad"],
            condition_latent_frames=inputs["condition_latents"].shape[2],
        )
        weight = self.train_video_scheduler.training_weight(timestep).to(
            device=loss_per_sample.device,
            dtype=loss_per_sample.dtype,
        )
        loss_video = (loss_per_sample * weight).mean()
        return loss_video, {"loss_video": float(loss_video.detach().item())}

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action=None,
        action_horizon=None,
        proprio=None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        generator: Optional[torch.Generator] = None,
        tiled: bool = False,
        condition_video: Optional[torch.Tensor] = None,
        condition_observation_video: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        del action, action_horizon, proprio, negative_prompt
        del text_cfg_scale, action_cfg_scale
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[:2] != (1, 3):
            raise ValueError(
                f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked = self._check_resize_height_width(height, width, int(num_frames))
        if checked != (height, width, int(num_frames)):
            raise ValueError(
                "Input H/W must be multiples of 16 and num_frames must satisfy T % 4 == 1, "
                f"got H={height}, W={width}, T={num_frames}"
            )

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt == use_context:
            raise ValueError("Provide exactly one of prompt or cached context/context_mask.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("context and context_mask must be provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    "`context/context_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)

        latent_t = (int(num_frames) - 1) // int(self.vae.temporal_downsample_factor) + 1
        latent_h = height // int(self.vae.upsampling_factor)
        latent_w = width // int(self.vae.upsampling_factor)
        if generator is None and seed is not None:
            generator = torch.Generator(device=rand_device).manual_seed(seed)
        noise_device = generator.device if generator is not None else torch.device(rand_device)
        latents = torch.randn(
            (1, int(self.vae.model.z_dim), latent_t, latent_h, latent_w),
            generator=generator,
            device=noise_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if condition_video is None:
            if condition_observation_video is not None:
                raise ValueError(
                    "`condition_observation_video` requires `condition_video` history."
                )
            condition_latents = self._encode_input_image_latents_tensor(
                input_image,
                tiled=tiled,
            )
            clip_fea = self._encode_clip_fea_from_image(input_image)
        else:
            if condition_video.ndim == 4:
                condition_video = condition_video.unsqueeze(0)
            if condition_video.ndim != 5 or condition_video.shape[:2] != (1, 3):
                raise ValueError(
                    "`condition_video` must be [1,3,T,H,W] or [3,T,H,W], got "
                    f"{tuple(condition_video.shape)}"
                )
            if condition_video.shape[-2:] != (height, width):
                raise ValueError(
                    "Condition and output spatial sizes must match, got "
                    f"{tuple(condition_video.shape[-2:])} vs {(height, width)}."
                )
            condition_video = condition_video.to(
                device=self.device,
                dtype=self.torch_dtype,
            )
            if condition_observation_video is not None:
                if condition_observation_video.ndim == 4:
                    condition_observation_video = condition_observation_video.unsqueeze(0)
                if condition_observation_video.shape != (1, 3, 1, height, width):
                    raise ValueError(
                        "`condition_observation_video` must be [1,3,1,H,W] with the same "
                        f"spatial size as the output, got {tuple(condition_observation_video.shape)}"
                    )
                condition_observation_video = condition_observation_video.to(
                    device=self.device,
                    dtype=self.torch_dtype,
                )
            condition_latents = self._encode_separate_condition_latents(
                condition_video,
                condition_observation_video=condition_observation_video,
                tiled=tiled,
            )
            if int(condition_latents.shape[2]) > latent_t:
                raise ValueError(
                    "Condition cannot be longer than the output latent sequence for "
                    f"direct conditioning, got {condition_latents.shape[2]} > {latent_t}."
                )
            clip_fea = self._encode_clip_fea_from_video(
                self._select_clip_condition_video(condition_video)
            )
        timesteps, deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.reshape(1).to(
                device=self.device,
                dtype=latents.dtype,
            )
            pred = self.dit(
                x=latents,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                action=None,
                clip_fea=clip_fea,
                condition_latents=condition_latents,
            )
            latents = self.infer_video_scheduler.step(pred, step_delta, latents)
        return {"video": self._decode_latents(latents, tiled=tiled)}

    @classmethod
    def from_wan21_14b_pretrained(
        cls,
        model_id,
        dit_checkpoint_path,
        video_dit_config,
        video_scheduler,
        tokenizer_model_id,
        tokenizer_max_len=512,
        load_text_encoder=False,
        load_clip_encoder=True,
        condition_clip_frame="first",
        condition_latent_loss_weight=1.0,
        redirect_common_files=False,
        torch_dtype=torch.bfloat16,
        device="cpu",
    ):
        del redirect_common_files
        components = load_wan21_i2v_14b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            dit_checkpoint_path=dit_checkpoint_path,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            dit_config=video_dit_config,
            load_text_encoder=load_text_encoder,
            load_clip_encoder=load_clip_encoder,
        )
        model = cls(
            dit=components.dit,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            clip_encoder=components.clip_encoder,
            text_dim=int(video_dit_config.get("text_dim", 4096)),
            video_train_shift=float(video_scheduler["train_shift"]),
            video_infer_shift=float(video_scheduler["infer_shift"]),
            video_num_train_timesteps=int(video_scheduler["num_train_timesteps"]),
            video_train_sampling_strategy=str(video_scheduler["train_sampling_strategy"]),
            video_train_beta_alpha=float(video_scheduler["beta_alpha"]),
            video_train_beta_beta=float(video_scheduler["beta_beta"]),
            condition_clip_frame=condition_clip_frame,
            condition_latent_loss_weight=condition_latent_loss_weight,
            torch_dtype=torch_dtype,
            device=device,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "clip_encoder": components.clip_encoder_path,
        }
        return model

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "format": "wan21_video_v1",
            "dit": self.dit.state_dict(),
            "step": step,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, Path(path))

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        state = payload["dit"] if "dit" in payload else payload
        self.dit.load_state_dict(state, strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
