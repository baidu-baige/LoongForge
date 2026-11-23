# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 model factory for the LoongForge eval server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from loongforge.embodied.eval.factories.registry import register_factory
from loongforge.embodied.eval.servers.eval_server_config import EvalServerArgs
from loongforge.embodied.eval.servers.loongforge_policy import PredictActionModelSpec
from loongforge.embodied.model.cosmos3.modeling_configuration_cosmos3 import Cosmos3ModelConfig
from loongforge.embodied.model.registry import build_model


@register_factory("cosmos3")
class Cosmos3ModelFactory:
    """Build a Cosmos3 model instance that implements the common predict_action interface."""

    model_config_cls = Cosmos3ModelConfig

    @classmethod
    def build(
        cls,
        model_cfg: Cosmos3ModelConfig,
        server_args: EvalServerArgs,
    ) -> PredictActionModelSpec:
        """Create Cosmos3, load the checkpoint and return it for the generic eval policy.

        Args:
            model_cfg: Typed Cosmos3ModelConfig resolved from the eval YAML model section.
            server_args: Typed EvalServerArgs with runtime/infra options.
        """
        import torch

        if server_args.random_init:
            raise ValueError("model_type 'cosmos3' does not support server.random_init")
        if not server_args.ckpt_path:
            raise ValueError("model_type 'cosmos3' requires server.ckpt_path")
        pretrained_path = str(Path(server_args.ckpt_path).expanduser())
        device = torch.device(server_args.device)

        # Cosmos3 is built on the meta device; materialize before loading weights.
        model = build_model(model_cfg)
        model.materialize(device, "bfloat16" if server_args.use_bf16 else "float32")
        model.load_pretrained(pretrained_path, device=device)
        model.eval()

        original_predict_action = model.predict_action

        @torch.no_grad()
        def _predict_action_wrapper(images, instructions, state=None, dataset_stats=None):
            return original_predict_action(images, instructions, state=state)

        model.predict_action = _predict_action_wrapper

        metadata: Dict[str, Any] = {
            "framework": "loongforge",
            "model_type": "cosmos3",
            "ckpt_path": pretrained_path,
            "loongforge_root": server_args.loongforge_root,
            "action_dim": int(model_cfg.action_dim),
            "action_horizon": int(model_cfg.action_horizon),
            "state_dim": int(model_cfg.state_dim),
            "num_camera_views": 1,
            # One request per observation; no batch contract yet.
            "supports_batching": False,
            "noise_seed": None,
        }
        return PredictActionModelSpec(model=model, metadata=metadata)
