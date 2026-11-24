# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VLA v2 model factory for the LoongForge eval server.

Eval goes through ``modeling_lingbot_vla_v2_infer.py`` (the upstream policy served
the way the community RoboTwin server does), not the training adapter; the two
share the ``hf_ckpt`` weights. ``LingBotVLAV2InferenceConfig`` is the registered
``model_config_cls``, so YAML ``model:`` keys map 1:1 onto it.

``server.ckpt_path`` points at ``.../checkpoints/global_step_N/hf_ckpt``; the
training config is read from three levels up unless
``model.training_config_path`` is set. The model returns the full 50-step chunk
and the generic chunk-cached policy executes all of it before replanning, which
is the community client's open-loop protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from loongforge.embodied.eval.factories.registry import register_factory
from loongforge.embodied.eval.servers.eval_server_config import EvalServerArgs
from loongforge.embodied.eval.servers.loongforge_policy import PredictActionModelSpec
from loongforge.embodied.model.lingbot_vla_v2.modeling_lingbot_vla_v2_infer import (
    STATE_DIM,
    IMAGE_KEYS,
    LingBotVLAV2InferenceConfig,
    LingBotVLAV2PredictActionModel,
)


@register_factory("lingbot_vla_v2")
class LingBotVLAV2ModelFactory:
    """Build the LingBot-VLA v2 inference model behind the shared predict_action contract."""

    model_config_cls = LingBotVLAV2InferenceConfig

    @classmethod
    def build(
        cls,
        model_cfg: LingBotVLAV2InferenceConfig,
        server_args: EvalServerArgs,
    ) -> PredictActionModelSpec:
        """Create the model and return it with metadata for the generic eval policy."""
        import torch

        if server_args.random_init:
            raise ValueError("lingbot_vla_v2 eval needs a real hf_ckpt; random_init is not supported")
        if not server_args.ckpt_path:
            raise ValueError("lingbot_vla_v2 eval requires server.ckpt_path (.../hf_ckpt)")
        ckpt_path = str(Path(server_args.ckpt_path).expanduser())
        device = (
            server_args.device
            if torch.cuda.is_available() or not server_args.device.startswith("cuda")
            else "cpu"
        )
        model = LingBotVLAV2PredictActionModel.from_pretrained(
            ckpt_path, model_cfg, use_bf16=bool(server_args.use_bf16), device=device
        )
        metadata: Dict[str, Any] = {
            "framework": "loongforge",
            "model_type": "lingbot_vla_v2",
            "ckpt_path": ckpt_path,
            "random_init": False,
            "loongforge_root": server_args.loongforge_root,
            "action_dim": STATE_DIM,
            "action_horizon": int(model_cfg.use_length),
            # Warmup must send the same three views as the real payload: the upstream
            # vision tower caches its split sizes on the first call.
            "num_camera_views": len(IMAGE_KEYS),
            "state_dim": STATE_DIM,
            "precision": "bf16" if server_args.use_bf16 else "fp32",
            "use_compile": bool(model_cfg.use_compile),
            "noise_seed": model_cfg.noise_seed,
            "tokenizer_path": model.config.tokenizer_path,
        }
        return PredictActionModelSpec(model=model, metadata=metadata)
