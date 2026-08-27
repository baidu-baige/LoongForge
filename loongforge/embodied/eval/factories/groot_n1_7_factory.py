# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.7 model factory for the LoongForge eval server.

GR00T-N1.7 is a multi-embodiment model. Its training-side
``GrootN1d7Policy.predict_action(images, instructions, state=None,
dataset_stats=None)`` already matches the shared eval contract, so this factory
only:

1. builds the policy (loading the Qwen3-VL/Cosmos backbone from
   ``model.model_name``),
2. optionally loads a fine-tuned checkpoint (skipped for ``random_init``),
3. calls ``configure_predict_action(...)`` to switch the active embodiment
   (e.g. ``libero_sim`` for LIBERO) and load statistics + the Qwen3VL processor.

Unlike N1.6 there is no ``eagle_assets_path``: the Qwen3-VL processor is loaded
from ``model.model_name`` by ``Gr00tN1d7DataCollator``, so the same local backbone
dir serves both the weights and the tokenizer/image processor.

All inference logic (state normalization, flow-matching sampling, action
unnormalization) lives inside the training-side ``predict_action`` and is
consumed as-is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from loongforge.embodied.eval.factories.registry import register_factory
from loongforge.embodied.eval.servers.eval_server_config import EvalServerArgs
from loongforge.embodied.eval.servers.loongforge_policy import PredictActionModelSpec
from loongforge.embodied.model.groot_n1_7.model_configuration_groot_n1_7 import (
    GrootN1d7Config,
)
from loongforge.embodied.model.registry import build_model


def _load_statistics(path: str) -> Dict[str, Any]:
    """Load the statistics JSON required by ``configure_predict_action``.

    GR00T-N1.7 cannot run without statistics even under ``random_init``: the
    embodiment's state/action normalization parameters are needed to build the
    ``StateActionProcessor``. Accepts either checkpoint-style
    ``{embodiment: {state, action, relative_action}}`` or raw LeRobot
    ``{observation.state, action, ...}`` stats — the model coerces both.
    """
    if not path:
        raise ValueError(
            "GR00T-N1.7 eval requires server.dataset_statistics_path (state/action "
            "normalization stats for the target embodiment). Provide the checkpoint's "
            "statistics.json even for a random_init link smoke."
        )
    stats_path = Path(path).expanduser()
    if not stats_path.exists():
        raise FileNotFoundError(f"GR00T-N1.7 dataset statistics not found: {path}")
    with stats_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_processor_kwargs(ckpt_path: str) -> Dict[str, Any]:
    """Read the checkpoint's ``processor_config.json`` ``processor_kwargs``.

    This is the serialized eval-time processor spec — the same file the official
    Isaac-GR00T server loads — and it is the only trustworthy source for the
    image pipeline: ``crop_fraction`` is 0.95 (not the 0.898 you get from
    ``image_crop_size / image_target_size``) and ``letter_box_transform`` is
    False (the training launch ``conf.yaml`` says true, and its backbone dims
    disagree with the released weights, so it is a stale template).

    Returns an empty dict when the file is absent (e.g. ``random_init`` against a
    bare dir), letting the model fall back to the released defaults.
    """
    if not ckpt_path:
        return {}
    config_path = Path(ckpt_path).expanduser() / "processor_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("processor_kwargs") or {}


@register_factory("gr00tn1d7")
class GrootN1d7ModelFactory:
    """Build a GR00T-N1.7 policy exposing the shared predict_action interface."""

    model_config_cls = GrootN1d7Config

    @classmethod
    def build(
        cls,
        model_cfg: GrootN1d7Config,
        server_args: EvalServerArgs,
    ) -> PredictActionModelSpec:
        """Create GrootN1d7Policy and return it with eval metadata.

        Args:
            model_cfg: Typed GrootN1d7Config resolved from the YAML ``model:``.
            server_args: Typed EvalServerArgs with runtime/infra options.
        """
        import os

        import torch

        pretrained_path = (
            str(Path(server_args.ckpt_path).expanduser()) if server_args.ckpt_path else ""
        )
        resolved_device = torch.device(
            server_args.device
            if torch.cuda.is_available() or not server_args.device.startswith("cuda")
            else "cpu"
        )

        # Builds GrootN1d7Policy (loads the Qwen3-VL backbone from model_cfg.model_name).
        model = build_model(model_cfg)
        if not server_args.random_init:
            # Load on CPU first: the backbone is already materialized on CPU here,
            # and the checkpoint carries every backbone tensor, so a GPU-side load
            # would double-allocate the 3B backbone before the .to(device) below.
            model.load_pretrained(pretrained_path, device=torch.device("cpu"))

        # Switch the active embodiment + load statistics / Qwen3VL processor.
        statistics = _load_statistics(server_args.dataset_statistics_path)
        embodiment_tag = server_args.embodiment_tag
        model.configure_predict_action(
            checkpoint_statistics=statistics,
            embodiment_tag=embodiment_tag,
            use_bf16=server_args.use_bf16,
            # Mirror the checkpoint's own processor spec (image crop/resize) verbatim
            # instead of re-deriving it; see _load_processor_kwargs.
            processor_kwargs=_load_processor_kwargs(pretrained_path),
            # Allow the startup warmup (state=None) to run; real eval steps always
            # pass an encoded state from the PayloadBuilder.
            validation_zero_state=True,
        )

        model = model.to(resolved_device)
        model.eval()
        if server_args.use_bf16 and resolved_device.type == "cuda":
            # Eval runs the whole model in bf16 like the official server. Without
            # this flag, GrootN1d7Policy._restore_precision_after_apply would cast
            # the trainable action head back to fp32 right after .to(bf16) below,
            # which is the training-side precision policy, not the eval behavior.
            os.environ.setdefault("GROOT_ALLOW_TRAINABLE_PARAM_BF16", "1")
            model = model.to(dtype=torch.bfloat16)

        # Open-loop execution horizon: keep only the first N steps of each
        # predicted chunk before the policy replans (official Isaac-GR00T LIBERO
        # client uses --n_action_steps 8). 0 -> no truncation (full chunk). The
        # generic policy then caches the truncated chunk and steps through it.
        _chunk_execute_steps = int(getattr(server_args, "chunk_execute_steps", 0) or 0)
        if _chunk_execute_steps > 0:
            _orig_predict_action = model.predict_action

            def _predict_action_truncate(images, instructions, state=None, dataset_stats=None, **kwargs):
                # Extra eval payload keys (cfg_scale, unnorm_key, ...) are swallowed
                # here and NOT forwarded: GrootN1d7Policy.predict_action does not
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
            "model_type": "gr00tn1d7",
            "embodiment_tag": embodiment_tag,
            "ckpt_path": pretrained_path if not server_args.random_init else "random_init://gr00tn1d7",
            "random_init": bool(server_args.random_init),
            "loongforge_root": server_args.loongforge_root,
            # Decoded LIBERO action dim (x, y, z, axis-angle x3, gripper).
            "action_dim": int(model.action_dim),
            # Per-embodiment decoded horizon (len(action delta_indices)), not the
            # model's DiT action_horizon.
            "action_horizon": int(model.action_horizon),
            "chunk_execute_steps": _chunk_execute_steps if _chunk_execute_steps > 0 else None,
            "dataset_statistics_path": server_args.dataset_statistics_path,
        }
        return PredictActionModelSpec(model=model, metadata=metadata)
