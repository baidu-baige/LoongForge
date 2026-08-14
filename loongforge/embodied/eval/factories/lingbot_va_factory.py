# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA model factory for the LoongForge eval server.

LingBot-VA serves eval through the inference-side model
(``modeling_lingbot_va_infer.py``), not the training adapter: the two share only the
transformer weights. ``LingBotVAInferenceConfig`` doubles as the registered
``model_config_cls``, so YAML ``model:`` keys map 1:1 onto it with no parallel
eval-side config dataclass.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict

from loongforge.embodied.eval.factories.registry import register_factory
from loongforge.embodied.eval.servers.eval_server_config import EvalServerArgs
from loongforge.embodied.eval.servers.loongforge_policy import PredictActionModelSpec
from loongforge.embodied.model.lingbot_va.modeling_lingbot_va_infer import (
    LingBotVAInferenceConfig,
    LingBotVAPredictActionModel,
)


@register_factory("lingbot_va")
class LingBotVAModelFactory:
    """Build the LingBot-VA inference model behind the shared predict_action contract."""

    model_config_cls = LingBotVAInferenceConfig

    @classmethod
    def build(
        cls,
        model_cfg: LingBotVAInferenceConfig,
        server_args: EvalServerArgs,
    ) -> PredictActionModelSpec:
        """Create the model and return it with metadata for the generic eval policy.

        ``server.ckpt_path`` points at the checkpoint root, which holds the
        ``transformer/`` weights plus the frozen ``vae/`` / ``text_encoder/`` /
        ``tokenizer/`` subfolders. Both config paths are derived from it so the YAML
        only carries one path.
        """
        import torch

        ckpt_root = str(Path(server_args.ckpt_path).expanduser()) if server_args.ckpt_path else ""
        device = (
            server_args.device
            if torch.cuda.is_available() or not server_args.device.startswith("cuda")
            else "cpu"
        )
        wan_path = ckpt_root or model_cfg.wan_pretrained_path or ""
        config = dataclasses.replace(
            model_cfg,
            checkpoint_path="" if server_args.random_init else ckpt_root,
            wan_pretrained_path=wan_path,
            device=device,
            dtype="bfloat16" if server_args.use_bf16 else model_cfg.dtype,
        )

        if server_args.random_init:
            model = LingBotVAPredictActionModel(config)
            model.to(device=device, dtype=config.torch_dtype)
            model.eval()
        else:
            model = LingBotVAPredictActionModel.from_pretrained(config)

        metadata: Dict[str, Any] = {
            "framework": "loongforge",
            "model_type": "lingbot_va",
            "ckpt_path": ckpt_root if not server_args.random_init else "random_init://lingbot_va",
            "random_init": bool(server_args.random_init),
            "loongforge_root": server_args.loongforge_root,
            "action_dim": len(config.used_action_channel_ids),
            # One env step per call: the model owns the action queue, so the eval-side
            # chunk cache is bypassed (see the PayloadBuilder's disable_action_cache).
            "action_horizon": 1,
            "wan_pretrained_path": config.wan_pretrained_path,
            "camera_layout": config.camera_layout,
            # Both views are concatenated into one frame, so a single-view
            # warmup call would be rejected (see loongforge_server._warmup_model).
            "num_camera_views": len(config.obs_cam_keys),
            "image_hflip": bool(config.image_hflip),
            "num_inference_steps": config.num_inference_steps,
            "action_num_inference_steps": config.action_num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "action_guidance_scale": config.action_guidance_scale,
            "dataset_statistics_path": server_args.dataset_statistics_path,
        }
        return PredictActionModelSpec(model=model, metadata=metadata)
