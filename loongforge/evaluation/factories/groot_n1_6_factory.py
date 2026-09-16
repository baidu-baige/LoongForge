# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.6 model factory for the LoongForge eval server.

GR00T-N1.6 is a multi-embodiment model. Its training-side
``GrootN1d6Policy.predict_action(images, instructions, state=None,
dataset_stats=None)`` already matches the shared eval contract, so this factory
only:

1. builds the policy (loading the Eagle VLM backbone from ``model.model_name``),
2. optionally loads a fine-tuned checkpoint (skipped for ``random_init``),
3. calls ``configure_predict_action(...)`` to switch the active embodiment
   (e.g. ``libero_panda`` for LIBERO), load statistics and the Eagle processor.

All inference logic (state normalization, flow-matching sampling, action
unnormalization) lives inside the training-side ``predict_action`` and is
consumed as-is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from loongforge.evaluation.factories.registry import register_factory
from loongforge.evaluation.servers.eval_server_config import EvalServerArgs
from loongforge.evaluation.servers.loongforge_policy import PredictActionModelSpec
from loongforge.models.vla.groot_n1_6.model_configuration_groot_n1_6 import (
    GrootN1d6ModelConfig,
)
from loongforge.models.native_registry import build_model


def _load_statistics(path: str) -> Dict[str, Any]:
    """Load the statistics JSON required by ``configure_predict_action``.

    GR00T-N1.6 cannot run without statistics even under ``random_init``: the
    embodiment's state/action normalization parameters are needed to build the
    ``StateActionProcessor``. Accepts either checkpoint-style
    ``{embodiment: {state, action, ...}}`` or raw LeRobot
    ``{observation.state, action, ...}`` stats — the model coerces both.
    """
    if not path:
        raise ValueError(
            "GR00T-N1.6 eval requires server.dataset_statistics_path (state/action "
            "normalization stats for the target embodiment). Provide the LIBERO "
            "dataset_statistics.json even for a random_init link smoke."
        )
    stats_path = Path(path).expanduser()
    if not stats_path.exists():
        raise FileNotFoundError(f"GR00T-N1.6 dataset statistics not found: {path}")
    with stats_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@register_factory("gr00tn1d6")
class GrootN1d6ModelFactory:
    """Build a GR00T-N1.6 policy exposing the shared predict_action interface."""

    model_config_cls = GrootN1d6ModelConfig

    @classmethod
    def build(
        cls,
        model_cfg: GrootN1d6ModelConfig,
        server_args: EvalServerArgs,
    ) -> PredictActionModelSpec:
        """Create GrootN1d6Policy and return it with eval metadata.

        Args:
            model_cfg: Typed GrootN1d6ModelConfig resolved from the YAML ``model:``.
            server_args: Typed EvalServerArgs with runtime/infra options.
        """
        import torch

        pretrained_path = (
            str(Path(server_args.ckpt_path).expanduser()) if server_args.ckpt_path else ""
        )
        resolved_device = torch.device(
            server_args.device
            if torch.cuda.is_available() or not server_args.device.startswith("cuda")
            else "cpu"
        )

        # Builds GrootN1d6Policy (loads the Eagle backbone from model_cfg.model_name).
        model = build_model(model_cfg)
        if not server_args.random_init:
            model.load_pretrained(pretrained_path, device=resolved_device)

        # Switch the active embodiment + load statistics / Eagle processor.
        statistics = _load_statistics(server_args.dataset_statistics_path)
        eagle_assets_path = model_cfg.vlm_tokenizer_path or model_cfg.model_name
        embodiment_tag = server_args.embodiment_tag
        model.configure_predict_action(
            checkpoint_statistics=statistics,
            eagle_assets_path=eagle_assets_path,
            embodiment_tag=embodiment_tag,
            use_bf16=server_args.use_bf16,
            # Allow the startup warmup (state=None) to run; real eval steps always
            # pass an encoded state from the PayloadBuilder.
            validation_zero_state=True,
        )

        model = model.to(resolved_device)
        model.eval()
        if server_args.use_bf16 and resolved_device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)

        # Open-loop execution horizon: keep only the first N steps of each
        # predicted chunk before the policy replans (official SimplerEnv WidowX
        # uses n_action_steps=4). 0 -> no truncation (full chunk). The generic
        # policy then caches the truncated chunk and steps through it.
        _chunk_execute_steps = int(getattr(server_args, "chunk_execute_steps", 0) or 0)
        if _chunk_execute_steps > 0:
            _orig_predict_action = model.predict_action

            def _predict_action_truncate(images, instructions, state=None, dataset_stats=None, **kwargs):
                # Extra eval payload keys (cfg_scale, unnorm_key, ...) are swallowed
                # here and NOT forwarded: GrootN1d6Policy.predict_action does not
                # consume them.
                result = _orig_predict_action(
                    images, instructions, state=state, dataset_stats=dataset_stats
                )
                arr = np.asarray(result)
                if arr.ndim == 3 and arr.shape[1] > _chunk_execute_steps:
                    arr = arr[:, :_chunk_execute_steps]
                elif arr.ndim == 2 and arr.shape[0] > _chunk_execute_steps:
                    arr = arr[:_chunk_execute_steps]
                return arr

            model.predict_action = _predict_action_truncate

        metadata: Dict[str, Any] = {
            "framework": "loongforge",
            "model_type": "gr00tn1d6",
            "embodiment_tag": embodiment_tag,
            "ckpt_path": pretrained_path if not server_args.random_init else "random_init://gr00tn1d6",
            "random_init": bool(server_args.random_init),
            "loongforge_root": server_args.loongforge_root,
            # Decoded LIBERO action dim (x, y, z, roll, pitch, yaw, gripper).
            "action_dim": int(model.action_dim),
            "action_horizon": int(model.action_horizon),
            "chunk_execute_steps": _chunk_execute_steps if _chunk_execute_steps > 0 else None,
            "dataset_statistics_path": server_args.dataset_statistics_path,
        }
        return PredictActionModelSpec(model=model, metadata=metadata)
