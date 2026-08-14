# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#
# The config defaults, dual denoising loops, chunk/keyframe accounting and action
# de-normalization follow upstream's ``wan_va/wan_va_server.py`` and
# ``wan_va/configs/va_libero_cfg.py``. Parts were cross-checked against the LingBot-VA
# port in LeRobot (Apache-2.0, Copyright 2024 The HuggingFace Inc. team); that port is not
# authoritative and diverges here where it conflicts with upstream (e.g. it duplicates
# ``grid_id`` for classifier-free guidance, which the native Triton RoPE kernel rejects).

"""Online inference adapter for LingBot-VA behind the shared eval ``predict_action`` contract.

Kept separate from ``modeling_lingbot_va.py`` (training) on purpose: the two share only the
transformer weights, and inference needs a whole layer training does not have — frozen VAE /
text encoder, per-episode autoregressive state, sampling schedules and camera layout. The
training module is not imported here beyond the checkpoint loader.

Reference: the upstream LingBot-VA LIBERO client (``evaluation/libero/client.py``) and its
LeRobot port.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .checkpoint import load_sharded_safetensors
from .modules.flow_match import LingBotVAFlowMatchScheduler
from .modules.rollout_state import RolloutStateStore
from .modules.wan_streaming import (
    WanStreamingTransformer3DModel,
    data_seq_to_patch,
    get_mesh_id_streaming,
)


# Training config keys that mean the same thing under a different name.
_TRAINING_KEY_ALIASES = {
    "lingbot_va_snr_shift": "snr_shift",
    "lingbot_va_action_snr_shift": "action_snr_shift",
    "lingbot_va_sigma_min": "sigma_min",
    "lingbot_va_extra_one_step": "extra_one_step",
    "lingbot_va_num_train_timesteps": "num_train_timesteps",
    "lingbot_va_chunk_size": "frame_chunk_size",
    "lingbot_va_diffusers_checkpoint_path": "checkpoint_path",
}

# Every layout branch is an ``== "robotwin_tshape"`` test with width_concat as the fallback,
# so an unrecognized value would assemble the wrong canvas, run to completion, and only show
# up as a degraded success rate. Validated in ``__post_init__``.
_CAMERA_LAYOUTS = ("width_concat", "robotwin_tshape")


@dataclass
class LingBotVAInferenceConfig:
    """Config for a closed-loop rollout: transformer geometry plus inference-only settings.

    :meth:`from_mapping` pulls the shared values out of a training YAML so the two cannot
    drift; the sampling / autoregression / camera / quantile fields have no training
    counterpart. Defaults match upstream ``wan_va/configs/va_libero_cfg.py``.
    """

    # Transformer geometry — must match the trained checkpoint.
    num_layers: int = 30
    hidden_size: int = 3072
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 24
    latent_in_channels: int = 48
    latent_out_channels: int = 48
    latent_patch_size: Tuple[int, int, int] = (1, 2, 2)
    action_dim: int = 30
    text_dim: int = 4096
    freq_dim: int = 256
    norm_epsilon: float = 1e-6
    rope_max_seq_len: int = 1024
    cross_attn_norm: bool = True

    # Flow matching. The streams are deliberately asymmetric: video is shifted towards high
    # noise and runs few steps, action the other way. ``video_exec_step=-1`` runs the full
    # video schedule; a positive value truncates it.
    snr_shift: float = 5.0
    action_snr_shift: float = 0.05
    sigma_min: float = 0.0
    extra_one_step: bool = True
    num_train_timesteps: int = 1000
    num_inference_steps: int = 20
    action_num_inference_steps: int = 50
    guidance_scale: float = 5.0
    action_guidance_scale: float = 1.0
    video_exec_step: int = -1

    # Autoregression.
    frame_chunk_size: int = 4
    action_per_frame: int = 4
    attn_window: int = 30

    # Observation encoding.
    height: int = 128
    width: int = 128
    camera_layout: str = "width_concat"
    obs_cam_keys: List[str] = field(default_factory=lambda: ["primary", "wrist"])
    # Undoes an extra horizontal flip applied upstream of the model. Upstream's LIBERO client
    # hands the model ``raw[::-1]`` (vertical only), while this repo's shared LIBERO adapter
    # emits ``raw[::-1, ::-1]`` — a 180 degree rotation (``eval/adapters/libero.py:80``). So
    # LIBERO must set this True; otherwise the model sees mirrored images, which degrades it
    # silently rather than erroring. Left False by default so every benchmark's adapter gets
    # checked instead of inheriting a value that only happens to suit LIBERO.
    image_hflip: bool = False

    # Action space: the model always emits ``action_dim`` channels; only these are real.
    used_action_channel_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    norm_q01: List[float] = field(default_factory=list)
    norm_q99: List[float] = field(default_factory=list)

    # ``wan_pretrained_path`` holds the frozen VAE / text encoder / tokenizer subfolders;
    # they are not bundled in the transformer checkpoint.
    checkpoint_path: Optional[str] = None
    wan_pretrained_path: Optional[str] = None
    max_sequence_length: int = 512

    device: str = "cuda"
    dtype: str = "bfloat16"

    @classmethod
    def from_mapping(cls, *sections: Optional[Dict[str, Any]], **overrides: Any):
        """Build a config from training YAML sections (``model``, ``data``, ...).

        Keys matching a field name are taken as-is, ``lingbot_va_*`` keys via
        :data:`_TRAINING_KEY_ALIASES`; unknown keys are ignored, so whole sections can be
        passed in directly.
        """
        known = {f.name for f in fields(cls)}
        values: Dict[str, Any] = {}
        for section in sections:
            for key, value in (section or {}).items():
                key = _TRAINING_KEY_ALIASES.get(key, key)
                if key in known and value is not None:
                    values[key] = value

        # Upstream builds ``inverse_used_action_channel_ids`` as ``[len(used)] * action_dim``
        # and overwrites the used positions with their index, so unused slots hold the
        # sentinel ``len(used)`` — also the maximum. Used channels are those below it.
        for section in sections:
            inverse_ids = (section or {}).get("inverse_used_action_channel_ids")
            if not inverse_ids or "used_action_channel_ids" in overrides:
                continue
            sentinel = max(inverse_ids)
            values["used_action_channel_ids"] = tuple(
                i for i, channel in enumerate(inverse_ids) if channel < sentinel
            )

        values.update(overrides)
        if "latent_patch_size" in values:
            values["latent_patch_size"] = tuple(values["latent_patch_size"])
        return cls(**values)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Resolve the ``dtype`` string to the matching ``torch`` dtype."""
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            self.dtype
        ]

    def __post_init__(self) -> None:
        """Reject an unrecognized camera layout (see :data:`_CAMERA_LAYOUTS`)."""
        if self.camera_layout not in _CAMERA_LAYOUTS:
            raise ValueError(
                f"unknown camera_layout {self.camera_layout!r}; expected one of {sorted(_CAMERA_LAYOUTS)}"
            )

    @property
    def attention_head_dim(self) -> int:
        """Per-head width implied by ``hidden_size`` and the head count."""
        return self.hidden_size // self.num_attention_heads

    @property
    def latent_hw(self) -> Tuple[int, int]:
        """Latent grid of the assembled frame: VAE downsamples 8x, patchify 2x on top."""
        if self.camera_layout == "robotwin_tshape":
            # Full-res head at the bottom, two half-res wrists side by side above it.
            return ((self.height // 16) * 3) // 2, self.width // 16
        return self.height // 16, (self.width // 16) * len(self.obs_cam_keys)

    @property
    def keyframe_stride(self) -> int:
        """Executed sub-steps per buffered keyframe; the VAE temporal downsample is 4."""
        return max(1, self.action_per_frame // 4)


class LingBotVAPredictActionModel(nn.Module):
    """Closed-loop LingBot-VA inference behind the shared ``predict_action`` contract.

    One ``predict_action`` call returns one action. The chunk queue, the transformer KV cache
    and the streaming VAE state live per episode in :mod:`.modules.rollout_state`, so eval
    **must** pass ``disable_action_cache=True``: with the front end's chunk cache enabled the
    model is called only once per chunk and never observes the intermediate frames the
    closed-loop feedback depends on.

    Per chunk: feed the buffered keyframes and executed actions back into the KV cache as
    *real* slots, denoise a fresh video latent, then denoise the action chunk conditioned on
    the same cache.
    """

    def __init__(self, config: LingBotVAInferenceConfig):
        """Build the streaming transformer and the two flow-matching schedulers.

        Weights are not loaded here; use :meth:`from_pretrained`. The frozen VAE and
        text encoder are loaded lazily on the first rollout.
        """
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.transformer = WanStreamingTransformer3DModel(
            patch_size=config.latent_patch_size,
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.latent_in_channels,
            out_channels=config.latent_out_channels,
            action_dim=config.action_dim,
            text_dim=config.text_dim,
            freq_dim=config.freq_dim,
            ffn_dim=config.ffn_hidden_size,
            num_layers=config.num_layers,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.norm_epsilon,
            rope_max_seq_len=config.rope_max_seq_len,
            # Training may use flex attention; the streaming KV-cache path is torch-only.
            attn_mode="torch",
        )

        self.scheduler = self._build_scheduler(config.snr_shift)
        self.action_scheduler = self._build_scheduler(config.action_snr_shift)
        self.scheduler.set_timesteps(config.num_inference_steps)
        self.action_scheduler.set_timesteps(config.action_num_inference_steps)
        self.use_cfg = config.guidance_scale > 1 or config.action_guidance_scale > 1

        self._frozen: Dict[str, Any] = {}
        self._store = RolloutStateStore(on_release=self._release_state)

    def _build_scheduler(self, shift: float) -> LingBotVAFlowMatchScheduler:
        """Instantiate a scheduler at the given SNR shift.

        The video and action streams denoise on deliberately different schedules, so
        each gets its own instance built through this helper.
        """
        return LingBotVAFlowMatchScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            shift=shift,
            sigma_min=self.config.sigma_min,
            extra_one_step=self.config.extra_one_step,
        )

    # ── construction / weights ───────────────────────────────────────
    @classmethod
    def from_pretrained(cls, config: LingBotVAInferenceConfig):
        """Build the model and load the transformer checkpoint onto the target device."""
        model = cls(config)
        if config.checkpoint_path:
            load_sharded_safetensors(model.transformer, config.checkpoint_path)
        model.to(device=config.device, dtype=config.torch_dtype)
        model.eval()
        return model

    def _ensure_frozen_modules(self) -> None:
        """Lazily load the frozen VAE / text encoder / tokenizer.

        They live in ``wan_pretrained_path`` subfolders rather than the transformer
        checkpoint, so loading is deferred to the first rollout instead of being paid for at
        construction.
        """
        if self._frozen:
            return
        from .modules import wan_codec

        path = self.config.wan_pretrained_path
        if not path:
            raise ValueError(
                "wan_pretrained_path must be set to load the frozen VAE / text encoder "
                "(they are not part of the transformer checkpoint)"
            )
        self._frozen = {
            "vae": wan_codec.load_vae(path, self.dtype, self.config.device, subfolder="vae"),
            "text_encoder": wan_codec.load_text_encoder(
                path, self.dtype, self.config.device, subfolder="text_encoder"
            ),
            "tokenizer": wan_codec.load_tokenizer(path, subfolder="tokenizer"),
        }

    # ── per-episode state ────────────────────────────────────────────
    def _release_state(self, state) -> None:
        """Drop the transformer KV cache when an episode ends.

        The cache belongs to the transformer instance rather than the state object, so it has
        to be cleared explicitly; the streaming VAE's ``feat_cache`` goes away with the state.
        """
        self.transformer.clear_cache("pos")

    def _episode_state(self, episode_id: str, episode_step: int):
        """Fetch the active episode state, (re)building it and its VAE wrappers as needed.

        Raises when a second episode is interleaved over this model instance; see
        :meth:`RolloutStateStore.get_or_start`.
        """
        from .modules.wan_codec import WanVAEStreamingWrapper

        state = self._store.get_or_start(episode_id, self.config.keyframe_stride, episode_step)
        if state.streaming_vae is not None:
            return state

        state.streaming_vae = WanVAEStreamingWrapper(self._frozen["vae"])
        if self.config.camera_layout == "robotwin_tshape":
            # The wrists are encoded at half resolution and need their own causal cache: one
            # ``feat_cache`` cannot span two different spatial shapes.
            state.streaming_vae_half = WanVAEStreamingWrapper(self._frozen["vae"])
        return state

    def _init_kv_cache(self) -> None:
        """Allocate the transformer KV pools for one sliding window of this config.

        Sizes the per-chunk token counts from the latent grid and the action layout; the
        batch is doubled when classifier-free guidance is active.
        """
        cfg = self.config
        latent_h, latent_w = cfg.latent_hw
        p = cfg.latent_patch_size
        self.transformer.create_empty_cache(
            "pos",
            cfg.attn_window,
            (cfg.frame_chunk_size * latent_h * latent_w) // (p[0] * p[1] * p[2]),
            cfg.frame_chunk_size * cfg.action_per_frame,
            device=cfg.device,
            dtype=self.dtype,
            batch_size=2 if self.use_cfg else 1,
        )

    @property
    def _action_mask(self) -> torch.Tensor:
        """Boolean mask over the model's 30 action channels selecting the used ones."""
        mask = torch.zeros([self.config.action_dim], dtype=torch.bool)
        mask[list(self.config.used_action_channel_ids)] = True
        return mask

    # ── observation / text encoding ──────────────────────────────────
    def _to_camera_frames(self, views: Sequence[Any]) -> List[torch.Tensor]:
        """Turn one observation's views into per-camera VAE clips ``[1, C, 1, H, W]``.

        ``views`` is ordered to match ``obs_cam_keys``. Images may arrive as HWC uint8 or as
        CHW floats in ``[0, 1]``. Under ``robotwin_tshape`` the wrists go in at half
        resolution.
        """
        from .modules.wan_codec import prepare_camera_frame

        cfg = self.config
        if len(views) < len(cfg.obs_cam_keys):
            raise ValueError(
                f"expected {len(cfg.obs_cam_keys)} camera views, got {len(views)}"
            )

        frames = []
        for idx in range(len(cfg.obs_cam_keys)):
            image = views[idx]
            if not torch.is_tensor(image):
                image = torch.from_numpy(np.ascontiguousarray(image))
            if image.ndim == 3 and image.shape[-1] in (1, 3):
                image = image.permute(2, 0, 1)  # HWC -> CHW
            image = image.float()
            if image.max() > 1.5:
                image = image / 255.0
            size = (cfg.height, cfg.width)
            if cfg.camera_layout == "robotwin_tshape" and idx > 0:
                size = (cfg.height // 2, cfg.width // 2)
            frames.append(
                prepare_camera_frame(
                    image, size, self.dtype, cfg.device, hflip=cfg.image_hflip
                )
            )
        return frames

    @torch.no_grad()
    def _encode_obs(self, state, obs_frames: List[List[torch.Tensor]]) -> torch.Tensor:
        """VAE-encode a temporal clip of observations into a normalized video latent.

        ``obs_frames`` is indexed ``[frame][camera]``; the codec wants ``[camera][frame]``.
        All frames go through one streaming call so the causal ``feat_cache`` carries across
        chunks and the x4 temporal downsample sees a continuous clip.
        """
        from .modules import wan_codec

        if self.config.camera_layout == "robotwin_tshape":
            return wan_codec.encode_frames_tshape(
                state.streaming_vae,
                state.streaming_vae_half,
                [f[0] for f in obs_frames],
                [f[1] for f in obs_frames],
                [f[2] for f in obs_frames],
                self.config.device,
            )
        per_camera = [
            [frame[cam] for frame in obs_frames]
            for cam in range(len(self.config.obs_cam_keys))
        ]
        return wan_codec.encode_frames_width_concat(
            state.streaming_vae, per_camera, self.config.device
        )

    def _encode_prompt(self, state, instruction: str) -> None:
        """Encode the task prompt once per episode (plus the empty prompt when using CFG)."""
        from .modules import wan_codec

        if state.prompt_embeds is not None:
            return
        cfg = self.config
        state.prompt = instruction or ""
        state.prompt_embeds = wan_codec.encode_text(
            self._frozen["tokenizer"],
            self._frozen["text_encoder"],
            state.prompt,
            cfg.max_sequence_length,
            self.dtype,
            cfg.device,
        )
        if self.use_cfg:
            state.negative_prompt_embeds = wan_codec.encode_text(
                self._frozen["tokenizer"],
                self._frozen["text_encoder"],
                "",
                cfg.max_sequence_length,
                self.dtype,
                cfg.device,
            )

    # ── stream input assembly ────────────────────────────────────────
    def _prepare_stream_inputs(
        self,
        state,
        latent_input: Optional[torch.Tensor],
        action_input: Optional[torch.Tensor],
        latent_t=0,
        action_t=0,
        latent_cond: Optional[torch.Tensor] = None,
        action_cond: Optional[torch.Tensor] = None,
        frame_st_id: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Build the per-stream ``input_dict`` the streaming transformer forward expects."""
        cfg = self.config
        p = cfg.latent_patch_size
        device = cfg.device
        out: Dict[str, Dict[str, torch.Tensor]] = {}

        if latent_input is not None:
            out["latent_res_lst"] = {
                "noisy_latents": latent_input,
                "timesteps": torch.ones(
                    [latent_input.shape[2]], dtype=torch.float32, device=device
                )
                * latent_t,
                "grid_id": get_mesh_id_streaming(
                    latent_input.shape[-3] // p[0],
                    latent_input.shape[-2] // p[1],
                    latent_input.shape[-1] // p[2],
                    0,
                    frame_shift=frame_st_id,
                ).to(device),
                "text_emb": state.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                # Frame 0 is the observed conditioning frame: keep it clean and mark it as
                # fully denoised (timestep 0) so the model treats it as known.
                out["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[:, :, 0:1]
                out["latent_res_lst"]["timesteps"][0:1] *= 0

        if action_input is not None:
            out["action_res_lst"] = {
                "noisy_latents": action_input,
                "timesteps": torch.ones(
                    [action_input.shape[2]], dtype=torch.float32, device=device
                )
                * action_t,
                "grid_id": get_mesh_id_streaming(
                    action_input.shape[-3],
                    action_input.shape[-2],
                    action_input.shape[-1],
                    1,
                    frame_shift=frame_st_id,
                    action=True,
                ).to(device),
                "text_emb": state.prompt_embeds.to(self.dtype).clone(),
            }
            if action_cond is not None:
                out["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
                out["action_res_lst"]["timesteps"][0:1] *= 0
            # Unused embodiment channels must stay exactly zero.
            out["action_res_lst"]["noisy_latents"][:, ~self._action_mask] *= 0
        return out

    def _repeat_for_cfg(self, state, stream: Dict[str, torch.Tensor]):
        """Stack the conditional and unconditional branches into one batch.

        ``grid_id`` is deliberately *not* duplicated: both branches occupy the same token
        positions, and this backend's rope is the triton kernel in :mod:`.modules.rope`,
        which requires ``frequencies.shape[0] == 1`` and broadcasts over the value batch
        anyway (it indexes ``cos``/``sin`` as ``[0, :, 0]``). ``timesteps`` does need the
        duplicate, since it feeds the per-sample condition embedder.
        """
        if self.use_cfg:
            stream["noisy_latents"] = stream["noisy_latents"].repeat(2, 1, 1, 1, 1)
            stream["text_emb"] = torch.cat(
                [
                    state.prompt_embeds.to(self.dtype).clone(),
                    state.negative_prompt_embeds.to(self.dtype).clone(),
                ],
                dim=0,
            )
            stream["grid_id"] = stream["grid_id"][None]
            stream["timesteps"] = stream["timesteps"][None].repeat(2, 1)
        else:
            stream["grid_id"] = stream["grid_id"][None]
            stream["timesteps"] = stream["timesteps"][None]
        return stream

    # ── the dual-stream denoising loop (one chunk) ───────────────────
    @torch.no_grad()
    def _infer(self, state, init_latent: Optional[torch.Tensor], frame_st_id: int = 0):
        """Denoise one chunk: video latents first, then actions conditioned on the same cache.

        Both loops append a trailing timestep 0 and treat the final iteration specially: it
        runs the transformer with ``update_cache=1`` so the fully denoised chunk lands in the
        KV cache as a *predicted* slot (later dropped by ``clear_pred_cache`` once the real
        observations arrive), and it does not step the scheduler again.
        """
        cfg = self.config
        device = cfg.device
        latent_h, latent_w = cfg.latent_hw
        chunk = cfg.frame_chunk_size
        cfg_batch = 2 if self.use_cfg else 1

        latents = torch.randn(
            1, cfg.latent_out_channels, chunk, latent_h, latent_w, device=device, dtype=self.dtype
        )
        actions = torch.randn(
            1, cfg.action_dim, chunk, cfg.action_per_frame, 1, device=device, dtype=self.dtype
        )

        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode="constant", value=0)
        if cfg.video_exec_step != -1:
            timesteps = timesteps[: cfg.video_exec_step]
        action_timesteps = F.pad(
            self.action_scheduler.timesteps, (0, 1), mode="constant", value=0
        )

        # 1. Video-latent denoising.
        for i, t in enumerate(timesteps):
            last_step = i == len(timesteps) - 1
            latent_cond = (
                init_latent[:, :, 0:1].to(self.dtype)
                if frame_st_id == 0 and init_latent is not None
                else None
            )
            streams = self._prepare_stream_inputs(
                state, latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id
            )
            noise_pred = self.transformer(
                self._repeat_for_cfg(state, streams["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name="pos",
                action_mode=False,
            )
            if not last_step or cfg.video_exec_step != -1:
                noise_pred = data_seq_to_patch(
                    cfg.latent_patch_size,
                    noise_pred,
                    chunk,
                    latent_h,
                    latent_w,
                    batch_size=cfg_batch,
                )
                if cfg.guidance_scale > 1:
                    noise_pred = noise_pred[1:] + cfg.guidance_scale * (
                        noise_pred[:1] - noise_pred[1:]
                    )
                else:
                    noise_pred = noise_pred[:1]
                latents = self.scheduler.step(noise_pred, t, latents)
            if frame_st_id == 0 and latent_cond is not None:
                latents[:, :, 0:1] = latent_cond

        # 2. Action denoising, conditioned on the video latents now in the cache.
        for i, t in enumerate(action_timesteps):
            last_step = i == len(action_timesteps) - 1
            action_cond = (
                torch.zeros(
                    [1, cfg.action_dim, 1, cfg.action_per_frame, 1],
                    device=device,
                    dtype=self.dtype,
                )
                if frame_st_id == 0
                else None
            )
            streams = self._prepare_stream_inputs(
                state, None, actions, t, t, None, action_cond, frame_st_id=frame_st_id
            )
            noise_pred = self.transformer(
                self._repeat_for_cfg(state, streams["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name="pos",
                action_mode=True,
            )
            if not last_step:
                noise_pred = rearrange(
                    noise_pred, "b (f n) c -> b c f n 1", f=chunk
                )
                if cfg.action_guidance_scale > 1:
                    noise_pred = noise_pred[1:] + cfg.action_guidance_scale * (
                        noise_pred[:1] - noise_pred[1:]
                    )
                else:
                    noise_pred = noise_pred[:1]
                actions = self.action_scheduler.step(noise_pred, t, actions)
            if frame_st_id == 0 and action_cond is not None:
                actions[:, :, 0:1] = action_cond

        actions[:, ~self._action_mask] *= 0
        return actions, latents

    @torch.no_grad()
    def _compute_kv_cache(self, state) -> None:
        """Feed the observed keyframes and executed actions back in as *real* cache slots."""
        if not state.obs_buffer or state.executed_actions is None:
            return
        # The previous chunk's predicted slots are superseded by what actually happened.
        self.transformer.clear_pred_cache("pos")

        latent_input = self._encode_obs(state, state.obs_buffer)
        if state.frame_st_id == 0 and state.init_latent is not None:
            # Upstream prepends the init latent on the first feedback so the latent and
            # action frame counts line up.
            latent_input = torch.cat([state.init_latent, latent_input], dim=2)
        action_input = state.executed_actions.to(latent_input)

        streams = self._prepare_stream_inputs(
            state, latent_input, action_input, frame_st_id=state.frame_st_id
        )
        self.transformer(
            self._repeat_for_cfg(state, streams["latent_res_lst"]),
            update_cache=2,
            cache_name="pos",
            action_mode=False,
        )
        self.transformer(
            self._repeat_for_cfg(state, streams["action_res_lst"]),
            update_cache=2,
            cache_name="pos",
            action_mode=True,
        )
        state.frame_st_id += latent_input.shape[2]

    # ── action post-processing ───────────────────────────────────────
    def _denormalize_actions(self, actions: torch.Tensor) -> np.ndarray:
        """Undo the dataset's quantile normalization on the used channels only.

        Training normalized actions to ``[-1, 1]`` via
        ``(a - q01) / (q99 - q01 + 1e-6) * 2 - 1``; this is upstream's exact inverse,
        ``(a + 1) / 2 * (q99 - q01 + 1e-6) + q01``. Unused channels have ``q01 == q99 == 0``
        (the ``1e-6`` is what keeps upstream from dividing by zero on them), so they are
        excluded here instead.
        """
        cfg = self.config
        used = list(cfg.used_action_channel_ids)
        out = actions.float()
        if not len(cfg.norm_q01) or not len(cfg.norm_q99):
            raise ValueError(
                "norm_q01 / norm_q99 are empty: the model would return actions still in the "
                "normalized [-1, 1] range, which runs to completion but never reaches the "
                "env's action scale. Set them from the dataset's quantile statistics "
                "(upstream keeps them in the task config as norm_stat)."
            )
        q01 = torch.as_tensor(cfg.norm_q01, dtype=torch.float32, device=out.device)[used]
        q99 = torch.as_tensor(cfg.norm_q99, dtype=torch.float32, device=out.device)[used]
        return (((out + 1.0) / 2.0) * (q99 - q01 + 1e-6) + q01).cpu().numpy()

    def _chunk_to_steps(self, actions: torch.Tensor, drop_first_frame: bool) -> np.ndarray:
        """Flatten a chunk ``[1, action_dim, F, action_per_frame, 1]`` to ``[n_steps, n_used]``.

        On the first chunk frame 0 is the conditioning frame — already "known" — so upstream's
        LIBERO client starts executing at frame 1 and its actions are dropped here.
        """
        a = actions[:, list(self.config.used_action_channel_ids)]
        if drop_first_frame:
            a = a[:, :, 1:]
        a = a.squeeze(-1).flatten(2)  # [1, n_used, n_steps]
        a = a.transpose(1, 2).contiguous()[0]  # [n_steps, n_used]
        return self._denormalize_actions(a)

    def _refill_queue(self, state, actions: torch.Tensor, drop_first_frame: bool) -> None:
        """Replace the queue with a freshly predicted chunk and keep it for the next feedback.

        ``actions`` stays in normalized space on the state because the next chunk feeds it
        back into the KV cache; only the queued copy is denormalized to env scale.
        """
        steps = self._chunk_to_steps(actions, drop_first_frame)
        state.action_queue = deque(steps)
        state.executed_actions = actions
        state.begin_chunk()

    # ── the eval entry point ─────────────────────────────────────────
    @torch.no_grad()
    def predict_action(
        self,
        images,
        instructions,
        state=None,
        dataset_stats=None,
        episode_id: str = "default",
        episode_step: int = 0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the action for the current env step as ``[1, n_used]``.

        Requires ``disable_action_cache=True`` on the eval side so this is called once per
        env step: every observation either conditions the first chunk, or is buffered as a
        keyframe for the closed-loop feedback.

        ``state`` (proprioception) and ``dataset_stats`` are part of the shared contract but
        unused here — LingBot-VA conditions on images and text only, and the action quantiles
        come from the config.
        """
        del state, dataset_stats  # unused by this model; see docstring
        self.eval()
        self._ensure_frozen_modules()

        rollout = self._episode_state(episode_id, episode_step)
        views = images[0] if len(images) and isinstance(images[0], (list, tuple)) else images
        instruction = instructions[0] if isinstance(instructions, (list, tuple)) else instructions
        self._encode_prompt(rollout, instruction)
        frames = self._to_camera_frames(views)

        if not rollout.started:
            # First observation conditions the first chunk; it is not a keyframe.
            rollout.started = True
            rollout.init_latent = self._encode_obs(rollout, [frames])
            self._init_kv_cache()
            actions, _latents = self._infer(rollout, rollout.init_latent, frame_st_id=0)
            rollout.first_chunk = False
            self._refill_queue(rollout, actions, drop_first_frame=True)
        else:
            # This observation is the result of the action just executed.
            if rollout.should_buffer_keyframe():
                rollout.obs_buffer.append(frames)
            if not rollout.action_queue:
                self._compute_kv_cache(rollout)
                actions, _latents = self._infer(
                    rollout, None, frame_st_id=rollout.frame_st_id
                )
                self._refill_queue(rollout, actions, drop_first_frame=False)

        action = rollout.action_queue.popleft()
        rollout.advance_exec_step(self.config.action_per_frame)
        return np.asarray(action, dtype=np.float32).reshape(1, -1)
