# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM evaluation factory (LIBERO + RoboTwin).

Server-side eval boundary for FastWAM:

- Builds ``FastWAMPolicy`` via ``build_model`` / ``FastWAMPolicy.from_pretrained``
  (Wan2.2-TI2V-5B backbone: video expert 5B + action expert 1.02B + Wan VAE +
  UMT5 text encoder) and loads an official released checkpoint through
  ``load_checkpoint`` (``libero_optional_idm_2cam224.pt`` /
  ``robotwin_uncond_3cam_384.pt``).
- Configures the rollout protocol via ``FastWAMPolicy.set_eval_protocol``:
  per-checkpoint camera packing (horizontal2 2cam224 / robotwin_t 3cam384),
  action width (7D / 14D) and stats normalization (min/max / z-score), plus
  action_infer_mode first_frame / idm, 10 flow-matching steps and
  action_horizon 32. State/action stats come from the checkpoint's
  ``*_dataset_stats.json`` and are attached to the instance there, not passed
  per RPC.
- Truncates each 32-step chunk to the official replan horizon (10 LIBERO /
  24 RoboTwin steps executed open-loop per chunk) so the generic chunk-cached
  policy replans on the official cadence.

Official LIBERO eval parameters (FastWAM ``fastwam_optional_idm.py``):
action_infer_mode='first_frame', num_inference_steps=10, sigma_shift=1.0,
action chunk 32 with 10 executed, 2 cams 224x224 cover-crop + horizontal
concat, proprio 8D min/max normalized, bf16.

Official RoboTwin eval parameters (``deploy_policy`` + sim_robotwin.yaml):
uncond checkpoint, sigma_shift=5.0, num_inference_steps=10, replan_steps=24,
3 cams T-packed to 384x320, proprio 14D joint pass-through, z-score
normalized, absolute 14D qpos actions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from loongforge.embodied.eval.factories.registry import register_factory
from loongforge.embodied.eval.servers.eval_server_config import EvalServerArgs
from loongforge.embodied.eval.servers.loongforge_policy import PredictActionModelSpec
from loongforge.embodied.model.fastwam.modeling_configuration_fastwam import FastWAMModelConfig
from loongforge.embodied.model.registry import build_model

logger = logging.getLogger(__name__)

# Official FastWAM replan horizons (``replan_steps``): LIBERO rolls 10 env
# steps per 32-step chunk, RoboTwin 24 (configs/sim_libero.yaml vs
# configs/sim_robotwin.yaml).
FASTWAM_CHUNK_EXECUTE_STEPS = 10
FASTWAM_ROBOTWIN_CHUNK_EXECUTE_STEPS = 24


@dataclass(frozen=True)
class FastWAMEvalConfig(FastWAMModelConfig):
    """FastWAM model config extended with eval-only semantics."""

    # ``idm`` variant: FastWAMIDM carries both rollouts — first-frame via the
    # base ``FastWAM.infer_action`` and two-stage IDM via its override — so a
    # single optional-IDM checkpoint serves both eval modes. ``uncond``
    # variant: plain FastWAM base (RoboTwin checkpoint; prompt is still sent
    # per the official deploy_policy — text_cfg_scale stays 1.0).
    variant: str = "idm"
    # LIBERO rollout needs the live UMT5 text encoder (prompt -> conditioning).
    # RoboTwin's official deploy_policy also loads it (uncond ckpt + prompt).
    load_text_encoder: bool = True
    # The optional-IDM checkpoint carries the full MoT (video + action expert);
    # nothing is loaded from the Wan DiT pretrain at build time.
    skip_dit_load_from_pretrain: bool = True
    action_dit_pretrained_path: str | None = None
    # Offline Wan release dir (model_id / tokenizer_model_id point into it via
    # DIFFSYNTH_MODEL_BASE_PATH); no modelscope redirect.
    redirect_common_files: bool = False
    # Eval protocol knobs (consumed by set_eval_protocol in build()).
    action_infer_mode: str = "first_frame"
    num_inference_steps: int = 10
    sigma_shift: float = 1.0
    num_video_frames: int = 9
    # Per-checkpoint eval geometry (official data configs):
    #   LIBERO 2cam224:  horizontal2 pack, 224 square, 7D action, min/max stats.
    #   RoboTwin 3cam384: T-layout pack (384x320), 14D action, z-score stats.
    camera_layout: str = "horizontal2"
    norm_mode: str = "min/max"
    # Per-call sampling seed: None keeps the legacy global-RNG behaviour; an int
    # fixes the same initial action noise for every call (official semantics).
    noise_seed: Optional[int] = None
    # When set (and noise_seed is an int, first_frame mode), the generated
    # initial action noise is written once to this .npy path so the official
    # eval arm can inject the bit-identical tensor.
    noise_dump_path: Optional[str] = None


@register_factory("fastwam")
class FastWAMModelFactory:
    """Build a FastWAM model instance implementing the predict_action interface."""

    model_config_cls = FastWAMEvalConfig

    @classmethod
    def build(
        cls,
        model_cfg: FastWAMEvalConfig,
        server_args: EvalServerArgs,
    ) -> PredictActionModelSpec:
        """Create the FastWAM LIBERO eval model and its metadata."""
        import torch

        ckpt_path = str(Path(server_args.ckpt_path).expanduser()) if server_args.ckpt_path else ""
        resolved_device = torch.device(
            server_args.device
            if torch.cuda.is_available() or not server_args.device.startswith("cuda")
            else "cpu"
        )

        model = build_model(model_cfg)
        model = model.to(resolved_device)
        model.eval()
        if not server_args.random_init:
            if not ckpt_path:
                raise ValueError("fastwam eval requires ckpt_path (or server.random_init)")
            model.load_checkpoint(ckpt_path)
        # Rollout protocol + checkpoint-owned normalization stats (attached to the
        # instance; predict_action consumes raw observations only).
        model.set_eval_protocol(
            stats_path=server_args.dataset_statistics_path or None,
            action_infer_mode=model_cfg.action_infer_mode,
            num_inference_steps=model_cfg.num_inference_steps,
            sigma_shift=model_cfg.sigma_shift,
            num_video_frames=model_cfg.num_video_frames,
            camera_layout=model_cfg.camera_layout,
            action_dim=model_cfg.action_dim,
            norm_mode=model_cfg.norm_mode,
            noise_seed=model_cfg.noise_seed,
            noise_dump_path=model_cfg.noise_dump_path,
        )

        # Truncate each 32-step chunk to the official replan horizon: 0 ->
        # benchmark default (LIBERO 10 / RoboTwin 24), N>0 -> N, N<0 -> full
        # 32 (no truncation).
        _orig_predict_action = model.predict_action
        _raw_chunk_steps = int(getattr(server_args, "chunk_execute_steps", 0) or 0)
        if _raw_chunk_steps == 0:
            _default_chunk_steps = (
                FASTWAM_ROBOTWIN_CHUNK_EXECUTE_STEPS
                if model_cfg.camera_layout == "robotwin_t"
                else FASTWAM_CHUNK_EXECUTE_STEPS
            )
            _chunk_execute_steps = _default_chunk_steps
        elif _raw_chunk_steps < 0:
            _chunk_execute_steps = 0  # disabled
        else:
            _chunk_execute_steps = _raw_chunk_steps

        def _predict_action_wrapper(images, instructions, state=None, dataset_stats=None, **kwargs):
            result = _orig_predict_action(
                images, instructions, state=state, dataset_stats=dataset_stats
            )
            # predict_action returns [B, H, 7]; index 0 is batch, not horizon.
            if _chunk_execute_steps > 0 and getattr(result, "ndim", 0) >= 2:
                if result.ndim == 3 and result.shape[1] > _chunk_execute_steps:
                    result = result[:, :_chunk_execute_steps]
                elif result.ndim == 2 and result.shape[0] > _chunk_execute_steps:
                    result = result[:_chunk_execute_steps]
            return result

        model.predict_action = _predict_action_wrapper

        metadata: Dict[str, Any] = {
            "framework": "loongforge",
            "model_type": "fastwam",
            "ckpt_path": ckpt_path if not server_args.random_init else "random_init://fastwam",
            "random_init": bool(server_args.random_init),
            "loongforge_root": server_args.loongforge_root,
            "action_dim": model_cfg.action_dim,
            "action_horizon": model_cfg.action_horizon,
            "action_infer_mode": model_cfg.action_infer_mode,
            # The server warmup uses this to construct the same number of
            # camera views as the real RoboTwin payload.
            "num_camera_views": 3 if model_cfg.camera_layout == "robotwin_t" else 2,
            "state_dim": model_cfg.proprio_dim,
            "chunk_execute_steps": _chunk_execute_steps if _chunk_execute_steps > 0 else None,
            "dataset_statistics_path": server_args.dataset_statistics_path,
            "tokenizer_path": server_args.tokenizer_path or os.environ.get("TOKENIZER_PATH", ""),
        }
        return PredictActionModelSpec(model=model, metadata=metadata)
