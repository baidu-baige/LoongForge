# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from FastWAM (https://github.com/yuantianyuan01/FastWAM).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastWAM policy for LoongForge embodied training and evaluation."""

from __future__ import annotations

import json
import logging
import os
import math
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from loongforge.embodied.model.registry import register_model

logger = logging.getLogger(__name__)


def _resolve_dtype_from_str(dtype_str: str) -> torch.dtype:
    """Resolve torch dtype from a string alias."""
    name = dtype_str.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported FastWAM dtype: {dtype_str!r}")


# ── Inference helpers (official LIBERO rollout protocol) ─────────────────────
# Official references: experiments/libero/eval_libero_single.py,
# src/fastwam/models/wan22/fastwam_optional_idm.py (first_frame branch),
# configs/sim_libero.yaml, configs/train.yaml, configs/data/libero_2cam.yaml.
# Official LIBERO constants: eval_num_inference_steps=10, sigma_shift=1.0
# (current checkpoints; 5.0 is the legacy uncond-checkpoint value),
# action_horizon=32 (= num_frames 33 - 1), 224x224 per camera.
EVAL_NUM_INFERENCE_STEPS = 10
EVAL_SIGMA_SHIFT = 1.0
EVAL_ACTION_HORIZON = 32
EVAL_INPUT_SIZE = 224
# IDM mode: video latents for (33 - 1) // 4 + 1 = 9 latent frames.
IDM_NUM_VIDEO_FRAMES = 9


class _MinMaxStats:
    """Per-dimension min/max normalizer (processor ``norm_default_mode: min/max``)."""

    def __init__(self, vmin: np.ndarray, vmax: np.ndarray):
        self.vmin = np.asarray(vmin, dtype=np.float32).reshape(-1)
        self.vmax = np.asarray(vmax, dtype=np.float32).reshape(-1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """x -> [-1, 1] via (x - min) / (max - min) * 2 - 1."""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.vmin.size:
            raise ValueError(
                f"min/max normalize: input dim {x.shape[1]} != stats dim {self.vmin.size}"
            )
        span = self.vmax - self.vmin
        span = np.where(span == 0.0, 1.0, span)
        return ((x - self.vmin) / span * 2.0 - 1.0).astype(np.float32)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`normalize`."""
        x = np.asarray(x, dtype=np.float32)
        span = (self.vmax - self.vmin).reshape(1, -1)
        span = np.where(span == 0.0, 1.0, span)
        return ((x + 1.0) / 2.0 * span + self.vmin.reshape(1, -1)).astype(np.float32)


class _ZScoreStats:
    """Per-dimension z-score normalizer (processor ``norm_default_mode: z-score``).

    Mirrors the official ``SingleFieldLinearNormalizer`` z-score branch:
    ``scale = 1 / (std + 1e-8)``, ``offset = -mean / (std + 1e-8)`` — i.e.
    normalize is ``(x - mean) / (std + 1e-8)`` (no [-1, 1] output range).
    """

    std_reg = 1e-8

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """x -> (x - mean) / (std + 1e-8)."""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.mean.size:
            raise ValueError(
                f"z-score normalize: input dim {x.shape[1]} != stats dim {self.mean.size}"
            )
        return ((x - self.mean) / (self.std + self.std_reg)).astype(np.float32)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`normalize`."""
        x = np.asarray(x, dtype=np.float32)
        return (x * (self.std + self.std_reg).reshape(1, -1) + self.mean.reshape(1, -1)).astype(np.float32)


def load_fastwam_dataset_stats(
    stats_path: Union[str, Path],
    norm_mode: str = "min/max",
) -> tuple[Union[_MinMaxStats, _ZScoreStats], Union[_MinMaxStats, _ZScoreStats]]:
    """Load state/action normalization stats from the checkpoint's stats json.

    Layout of ``<ckpt>_dataset_stats.json``:
    ``{"state": {"default": {"global_min": [D], "global_max": [D],
    "global_mean": [D], "global_std": [D], ...}},
    "action": {"default": {...}}}`` (``stepwise_*`` entries carry an extra
    leading dim; the global entries are authoritative because stepwise action
    norm is disabled).

    ``norm_mode`` selects the per-dataset ``processor.norm_default_mode``:
    ``"min/max"`` (LIBERO configs) or ``"z-score"`` (RoboTwin config).
    """
    with open(stats_path, "r") as f:
        stats = json.load(f)
    try:
        state, action = stats["state"]["default"], stats["action"]["default"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"dataset_stats.json missing {exc!r}: {stats_path}") from exc
    if norm_mode == "min/max":
        return (
            _MinMaxStats(state["global_min"], state["global_max"]),
            _MinMaxStats(action["global_min"], action["global_max"]),
        )
    if norm_mode == "z-score":
        return (
            _ZScoreStats(state["global_mean"], state["global_std"]),
            _ZScoreStats(action["global_mean"], action["global_std"]),
        )
    raise ValueError(f"Unsupported fastwam norm_mode: {norm_mode!r} (min/max | z-score)")


def _quat_to_axisangle(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> rotation vector (official ``quat2axisangle``)."""
    w, x, y, z = (float(v) for v in np.asarray(quat_wxyz, dtype=np.float64).reshape(-1))
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(norm, w)
    return (np.array([x, y, z], dtype=np.float64) / norm * angle).astype(np.float32)


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Cover-style bilinear resize then center crop.

    Exact port of the official FastWAM eval implementation: PIL BILINEAR on
    uint8 (NOT torch F.interpolate — the two differ numerically, and the
    official uint8 rounding is part of the training-time preprocessing).
    """
    from PIL import Image

    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _resize_bilinear(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Plain bilinear resize to ``(width, height)`` (official ``_resize_rgb``)."""
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _build_robotwin_t_image(views: list[np.ndarray]) -> np.ndarray:
    """Pack the RoboTwin 3-cam T layout (official ``_build_robotwin_image_tensor``).

    ``views`` is ``[head, left_wrist, right_wrist]`` at env-native resolution.
    head -> (320, 256) top row; each wrist -> (160, 128); bottom = [left|right]
    horizontally; image = [head; bottom] vertically -> ``[384, 320, 3]``.
    """
    head, left, right = views
    head_r = _resize_bilinear(head, (320, 256))
    left_r = _resize_bilinear(left, (160, 128))
    right_r = _resize_bilinear(right, (160, 128))
    bottom = np.concatenate([left_r, right_r], axis=1)
    return np.concatenate([head_r, bottom], axis=0)  # [384, 320, 3]


@register_model("fastwam")
class FastWAMPolicy(nn.Module):
    """LoongForge wrapper for the local FastWAM implementation.

    ``BCTrainer`` expects ``model(batch)`` to return a dictionary containing
    ``action_loss``. FastWAM returns ``(loss, metrics)`` from ``training_loss``;
    this wrapper adapts the return value and exposes common checkpoint methods.
    """

    def __init__(self, core: nn.Module):
        """Wrap a FastWAM core module for the embodied trainer interface."""
        super().__init__()
        self.core = core
        self.dit = getattr(core, "dit", None)
        # Eval rollout protocol (configured via :meth:`set_eval_protocol`).
        self._eval_action_infer_mode = "first_frame"
        self._eval_num_inference_steps = EVAL_NUM_INFERENCE_STEPS
        self._eval_sigma_shift = EVAL_SIGMA_SHIFT
        self._eval_num_video_frames = IDM_NUM_VIDEO_FRAMES
        self._eval_input_size = EVAL_INPUT_SIZE
        self._eval_camera_layout = "horizontal2"
        self._eval_action_dim = 7
        self._eval_state_stats: _MinMaxStats | _ZScoreStats | None = None
        self._eval_action_stats: _MinMaxStats | _ZScoreStats | None = None

    def set_eval_protocol(
        self,
        stats_path: Union[str, Path, None] = None,
        action_infer_mode: str = "first_frame",
        num_inference_steps: int = EVAL_NUM_INFERENCE_STEPS,
        sigma_shift: float = EVAL_SIGMA_SHIFT,
        num_video_frames: int = IDM_NUM_VIDEO_FRAMES,
        input_size: int = EVAL_INPUT_SIZE,
        camera_layout: str = "horizontal2",
        action_dim: int = 7,
        norm_mode: str = "min/max",
        noise_seed: Union[int, None] = None,
        noise_dump_path: Union[str, Path, None] = None,
    ) -> None:
        """Configure the official rollout protocol on this policy.

        ``action_infer_mode``:
        - ``"first_frame"`` (Fast-WAM): condition the action head on the
          current frame's VAE latents only — no test-time future video
          imagination (``FastWAM.infer_action``).
        - ``"idm"``: two-stage joint denoising — Stage1 video diffusion fills
          the future video latents, Stage2 teacher-forces them into the action
          denoising (``FastWAMIDM.infer_action`` -> ``infer_joint``).
        Both modes run per call with no cross-call state, so episodes can mix
        modes freely.

        ``camera_layout`` selects the official per-benchmark image packing:
        - ``"horizontal2"`` (LIBERO 2cam224): per-camera cover-style resize +
          center crop to ``input_size`` square, horizontal concat
          (agent | wrist) -> ``input_size x 2*input_size``.
        - ``"robotwin_t"`` (RoboTwin 3cam384): plain bilinear resize — head
          cam to ``W x input_size``-wide top row and the two wrist cams to
          ``W/2 x input_size/2`` bottom row, vertical concat (head on top)
          -> ``input_size x W`` (official deploy_policy ``_build_robotwin_image_tensor``:
          head (320,256), wrists (160,128) each, bottom = [left|right],
          image = [head; bottom] -> [384,320,3], no crop).

        ``action_dim`` is the checkpoint's action width (7 LIBERO / 14
        RoboTwin); ``norm_mode`` the dataset's processor normalization
        (``min/max`` LIBERO / ``z-score`` RoboTwin). ``stats_path`` points at
        the checkpoint's ``*_dataset_stats.json``; normalization is part of
        the model contract (eval passes raw observations only).
        """
        if action_infer_mode not in {"first_frame", "idm"}:
            raise ValueError(
                f"action_infer_mode must be 'first_frame' or 'idm', got {action_infer_mode!r}"
            )
        if camera_layout not in {"horizontal2", "robotwin_t"}:
            raise ValueError(
                f"camera_layout must be 'horizontal2' or 'robotwin_t', got {camera_layout!r}"
            )
        self._eval_action_infer_mode = action_infer_mode
        self._eval_noise_seed = noise_seed
        self._eval_noise_dump_path = None if noise_dump_path is None else Path(noise_dump_path)
        self._eval_num_inference_steps = int(num_inference_steps)
        self._eval_sigma_shift = float(sigma_shift)
        self._eval_num_video_frames = int(num_video_frames)
        self._eval_input_size = int(input_size)
        self._eval_camera_layout = camera_layout
        self._eval_action_dim = int(action_dim)
        if stats_path is not None:
            self._eval_state_stats, self._eval_action_stats = load_fastwam_dataset_stats(
                stats_path, norm_mode=norm_mode
            )

    @torch.no_grad()
    def predict_action(
        self,
        images: Any,
        instructions: Any,
        state: Any = None,
        dataset_stats: Any = None,
        episode_id: str = "default",
        episode_step: int = 0,
        **_: Any,
    ) -> np.ndarray:
        """Predict a denormalized action chunk for the eval protocol.

        Args:
            images: batched views (``[[view, ...], ...]``), each a uint8 HWC
                array at env-native resolution. Layout depends on the
                configured ``camera_layout``:

                - ``"horizontal2"`` (LIBERO): ``[agent_view, wrist_view]``,
                  cover-style resize + center crop to ``input_size`` square
                  each, horizontal concat (agent left | wrist right) ->
                  224x448.
                - ``"robotwin_t"`` (RoboTwin): ``[head, left_wrist,
                  right_wrist]``, T-layout pack -> 384x320 (see
                  :func:`_build_robotwin_t_image`).

                Pixels are mapped to ``x * (2/255) - 1``.
            instructions: list-of-str batch (batch=1 supported).
            state: proprio per batch item — dim matches the checkpoint
                (8D LIBERO / 14D RoboTwin joint); normalized inside.
            dataset_stats: unused — FastWAM normalization stats come from the
                checkpoint's own ``*_dataset_stats.json`` via
                :meth:`set_eval_protocol`.

        Returns:
            ``[batch, 32, action_dim]`` float32 — 7D: dims 0-5 delta EEF
            (pos + axis-angle) + dim 6 absolute gripper (LIBERO); 14D:
            absolute joint targets (RoboTwin). Min/max or z-score
            denormalized per the checkpoint's stats.
        """
        from loongforge.embodied.model.fastwam.mot.fastwam import FastWAM

        if self._eval_state_stats is None:
            raise RuntimeError(
                "call set_eval_protocol(stats_path=...) before predict_action: "
                "state/action normalization stats are required"
            )
        if not isinstance(images[0], (list, tuple)):
            images = [list(images)]
        batch = len(images)
        if batch != 1:
            raise ValueError(f"FastWAM predict_action supports batch=1, got {batch}")
        if self.training or not self.core.eval:
            # §2.2 of docs/推理正确性验证方法.md: assert eval mode so dropout /
            # checkpointing branches cannot silently diverge from rollout.
            self.eval()

        views = images[0]
        if self._eval_camera_layout == "horizontal2":
            if len(views) != 2:
                raise ValueError(
                    f"horizontal2 layout expects [agent, wrist], got {len(views)} views"
                )
            size = self._eval_input_size
            agent = _center_crop_resize(np.asarray(views[0]), size, size)
            wrist = _center_crop_resize(np.asarray(views[1]), size, size)
            rgb = np.concatenate([agent, wrist], axis=1)
        else:  # robotwin_t
            if len(views) != 3:
                raise ValueError(
                    f"robotwin_t layout expects [head, left, right], got {len(views)} views"
                )
            rgb = _build_robotwin_t_image(
                [np.asarray(v) for v in views]
            )
        input_image = (
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(
                device=self.core.device, dtype=self.core.torch_dtype
            )
            / 255.0
            * 2.0
            - 1.0
        )
        state_np = np.asarray(state, dtype=np.float32).reshape(1, -1)
        state_norm = torch.from_numpy(self._eval_state_stats.normalize(state_np)).to(
            device=self.core.device, dtype=self.core.torch_dtype
        )

        call_seed = None
        action_noise = None
        if self._eval_noise_seed is not None:
            # Fixed seed for every call (official eval semantics): each replan
            # starts from the same initial action noise, so paired closed-loop
            # arms stay noise-aligned step by step.
            call_seed = int(self._eval_noise_seed)
            if self._eval_action_infer_mode == "first_frame":
                # Build the noise here (same construction as FastWAM.infer_action:
                # cpu generator, float32) so it can be dumped to disk and injected
                # into both arms — bit-identical regardless of torch version.
                g = torch.Generator(device="cpu").manual_seed(call_seed)
                action_noise = torch.randn(
                    (1, EVAL_ACTION_HORIZON, self._eval_action_dim),
                    generator=g,
                    device="cpu",
                    dtype=torch.float32,
                )
                dump_path = getattr(self, "_eval_noise_dump_path", None)
                if dump_path is not None and not dump_path.exists():
                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(dump_path, action_noise.numpy())
                    logger.info(
                        "Dumped shared action noise to %s (seed=%d, shape=%s)",
                        dump_path,
                        call_seed,
                        tuple(action_noise.shape),
                    )

        if self._eval_action_infer_mode == "first_frame":
            # ``self.core`` may be a FastWAMIDM, whose overridden ``infer_action``
            # routes to ``infer_joint`` (IDM mode). The first-frame rollout calls
            # the base ``FastWAM.infer_action`` instead — same as the official
            # ``action_infer_mode='first_frame'`` branch in
            # fastwam_optional_idm.py.
            out = FastWAM.infer_action(
                self.core,
                prompt=instructions[0],
                input_image=input_image,
                action_horizon=EVAL_ACTION_HORIZON,
                proprio=state_norm,
                num_inference_steps=self._eval_num_inference_steps,
                sigma_shift=self._eval_sigma_shift,
                seed=call_seed,
            )
        else:
            out = self.core.infer_action(
                prompt=instructions[0],
                input_image=input_image,
                action_horizon=EVAL_ACTION_HORIZON,
                num_video_frames=self._eval_num_video_frames,
                proprio=state_norm,
                num_inference_steps=self._eval_num_inference_steps,
                sigma_shift=self._eval_sigma_shift,
                seed=call_seed,
            )
        chunk = np.asarray(out["action"], dtype=np.float32).reshape(-1, self._eval_action_dim)
        denorm = self._eval_action_stats.unnormalize(chunk)

        trace_dir = os.environ.get("FASTWAM_TRACE_DIR")
        if trace_dir and instructions[0] != "warmup":  # skip server-startup dummy call
            d = Path(trace_dir)
            d.mkdir(parents=True, exist_ok=True)
            call_idx = len(list(d.glob("call_*.npz")))
            np.savez(
                d / f"call_{call_idx:04d}.npz",
                input_image=input_image.detach().to(torch.float32).cpu().numpy(),
                proprio=state_norm.detach().to(torch.float32).cpu().numpy(),
                raw_chunk=chunk.astype(np.float32),
                denorm_chunk=denorm.astype(np.float32),
            )
        return denorm[None, :, :]  # [batch=1, horizon, action_dim]


    @classmethod
    def from_pretrained(cls, cfg: Any) -> "FastWAMPolicy":
        """Build a FastWAM policy from a typed FastWAMConfig instance."""
        from loongforge.embodied.model.fastwam.modeling_configuration_fastwam import FastWAMModelConfig
        from loongforge.embodied.model.fastwam.mot.fastwam import FastWAM
        from loongforge.embodied.model.fastwam.mot.idm import FastWAMIDM
        from loongforge.embodied.model.fastwam.mot.joint import FastWAMJoint

        if not isinstance(cfg, FastWAMModelConfig):
            raise TypeError(
                "FastWAMPolicy.from_pretrained expects a typed FastWAMModelConfig instance; "
                f"got {type(cfg).__name__}. build_model now passes ModelConfig directly."
            )
        config = cfg

        variant_map = {
            "base": FastWAM,
            "uncond": FastWAM,
            "joint": FastWAMJoint,
            "idm": FastWAMIDM,
        }
        cls_map = variant_map[config.variant]  # variant already validated in __post_init__

        # dtype is config-driven (training uses bf16; eval may set float32 via
        # model.dtype in the YAML to bypass HEAD flash_attn rope's bf16 issue).
        model_dtype = _resolve_dtype_from_str(config.dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        core = cls_map.from_wan22_pretrained(
            device=device,
            torch_dtype=model_dtype,
            model_id=config.model_id,
            tokenizer_model_id=config.tokenizer_model_id,
            tokenizer_max_len=config.tokenizer_max_len,
            load_text_encoder=config.load_text_encoder,
            proprio_dim=config.proprio_dim,
            redirect_common_files=config.redirect_common_files,
            video_dit_config=config.video_dit_config,
            action_dit_config=config.action_dit_config,
            action_dit_pretrained_path=config.action_dit_pretrained_path,
            skip_dit_load_from_pretrain=config.skip_dit_load_from_pretrain,
            mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
            drop_all_true_cross_attn_mask=config.drop_all_true_cross_attn_mask,
            compile_mot_blocks=config.mot_compile_blocks,
            compile_dynamic=config.compile_dynamic,
            video_train_shift=float(config.video_scheduler["train_shift"]),
            video_infer_shift=float(config.video_scheduler["infer_shift"]),
            video_num_train_timesteps=int(config.video_scheduler["num_train_timesteps"]),
            action_train_shift=float(config.action_scheduler["train_shift"]),
            action_infer_shift=float(config.action_scheduler["infer_shift"]),
            action_num_train_timesteps=int(config.action_scheduler["num_train_timesteps"]),
            loss_lambda_video=float(config.loss["lambda_video"]),
            loss_lambda_action=float(config.loss["lambda_action"]),
        )
        if config.compile_vae_encode:
            core.vae.encode = torch.compile(core.vae.encode, dynamic=config.compile_dynamic)
            logger.info("[compile] torch.compile on VAE encode (dynamic=%s)", config.compile_dynamic)
        return cls(core)

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Run FastWAM training loss and adapt metrics to trainer output dict."""
        sample = batch.to_sample() if hasattr(batch, "to_sample") else batch
        loss, metrics = self.core.training_loss(sample)
        output: Dict[str, torch.Tensor] = {"action_loss": loss}
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if torch.is_tensor(value):
                    output[key] = value.detach()
                else:
                    output[key] = torch.tensor(float(value), device=loss.device)

        return loss, {k: v for k, v in output.items() if k != "action_loss"}

    def load_pretrained(self, path: str, device=None):
        """Load a FastWAM checkpoint through the wrapped core module."""
        del device
        return self.core.load_checkpoint(path)

    def save_checkpoint(self, *args, **kwargs):
        """Save a FastWAM checkpoint through the wrapped core module."""
        return self.core.save_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs):
        """Load a FastWAM checkpoint through the wrapped core module."""
        return self.core.load_checkpoint(*args, **kwargs)
