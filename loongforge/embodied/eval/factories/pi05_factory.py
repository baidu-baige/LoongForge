# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""PI05 model factory for the LoongForge eval server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from loongforge.embodied.eval.factories.registry import register_factory
from loongforge.embodied.eval.servers.eval_server_config import EvalServerArgs
from loongforge.embodied.eval.servers.loongforge_policy import (
    GenericPredictActionPolicy,
    PredictActionModelSpec,
)
from loongforge.embodied.model.pi05.model_configuration_pi05 import Pi05ModelConfig
from loongforge.embodied.model.registry import build_model


@register_factory("pi05")
class PI05ModelFactory:
    """Build a PI05 model instance that implements the common predict_action interface."""

    model_config_cls = Pi05ModelConfig

    @classmethod
    def build(
        cls,
        model_cfg: Pi05ModelConfig,
        server_args: EvalServerArgs,
    ) -> PredictActionModelSpec:
        """Create PI05 and return it with metadata for the generic eval policy.

        Args:
            model_cfg: Typed Pi05ModelConfig resolved from the eval YAML model section.
            server_args: Typed EvalServerArgs with runtime/infra options.
        """
        import torch

        pretrained_path = str(Path(server_args.ckpt_path).expanduser()) if server_args.ckpt_path else ""
        resolved_device = torch.device(
            server_args.device
            if torch.cuda.is_available() or not server_args.device.startswith("cuda")
            else "cpu"
        )

        model = build_model(model_cfg)
        model._tokenizer_path = server_args.tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
        if not server_args.random_init:
            model.model.load_pretrained(pretrained_path, device=resolved_device)
        model = model.to(resolved_device)
        model.eval()
        if server_args.use_bf16 and resolved_device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)

        # Keep only the first N steps of each chunk so the policy replans every
        # N steps. 0 / negative = no truncation (full action_horizon).
        chunk_execute_steps = int(server_args.chunk_execute_steps)
        image_norm_mode = server_args.eval_image_normalize_mode
        norm_mode = server_args.eval_normalization_mode

        from loongforge.embodied.data.datasets.transforms.utils.image_transform import ImageTransform
        from loongforge.embodied.data.datasets.transforms.utils.normalizer import Normalizer

        if image_norm_mode not in ImageTransform.NORMALIZE_MODES:
            raise ValueError(
                f"Invalid eval_image_normalize_mode={image_norm_mode!r}; "
                f"expected one of {ImageTransform.NORMALIZE_MODES}"
            )
        if norm_mode not in Normalizer.VALID_MODES:
            raise ValueError(
                f"Invalid eval_normalization_mode={norm_mode!r}; "
                f"expected one of {Normalizer.VALID_MODES}"
            )

        original_predict_action = model.predict_action

        @torch.no_grad()
        def _predict_action_wrapper(images, instructions, state=None, dataset_stats=None):
            out = original_predict_action(
                images, instructions, state=state, dataset_stats=dataset_stats,
                image_normalize_mode=image_norm_mode, normalization_mode=norm_mode,
            )
            return out[:, :chunk_execute_steps] if chunk_execute_steps > 0 else out

        model.predict_action = _predict_action_wrapper

        metadata: Dict[str, Any] = {
            "framework": "loongforge",
            "model_type": "pi05",
            "ckpt_path": pretrained_path if not server_args.random_init else "random_init://pi05",
            "random_init": bool(server_args.random_init),
            "loongforge_root": server_args.loongforge_root,
            "action_dim": model_cfg.action_dim,
            "action_horizon": model_cfg.action_horizon,
            "dataset_statistics_path": server_args.dataset_statistics_path,
            "tokenizer_path": model._tokenizer_path,
            "chunk_execute_steps": chunk_execute_steps if chunk_execute_steps > 0 else None,
        }
        return PredictActionModelSpec(model=model, metadata=metadata)


class LoongForgePI05Policy(GenericPredictActionPolicy):
    """Backward-compatible PI05 policy built from the generic predict_action policy."""

    def __init__(
        self,
        ckpt_path: str,
        loongforge_root: str = "",
        device: str = "cuda",
        use_bf16: bool = True,
        dataset_statistics_path: str = "",
        tokenizer_path: str = "",
        action_dim: int = 7,
        state_dim: int = 7,
        action_horizon: int = 50,
        max_action_dim: int = 32,
        max_state_dim: int = 32,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        random_init: bool = False,
    ) -> None:
        """Build PI05 through its factory and attach it to the generic eval policy."""
        model_cfg = Pi05ModelConfig(
            action_dim=action_dim,
            state_dim=state_dim,
            action_horizon=action_horizon,
            max_action_dim=max_action_dim,
            max_state_dim=max_state_dim,
            compile_model=compile_model,
            compile_mode=compile_mode,
        )
        server_args = EvalServerArgs(
            ckpt_path=ckpt_path,
            loongforge_root=loongforge_root,
            device=device,
            use_bf16=use_bf16,
            dataset_statistics_path=dataset_statistics_path,
            tokenizer_path=tokenizer_path,
            random_init=random_init,
        )
        spec = PI05ModelFactory.build(model_cfg, server_args)
        super().__init__(
            model=spec.model,
            metadata=spec.metadata,
            dataset_statistics_path=dataset_statistics_path,
            action_dim=action_dim,
            request_id_prefix="loongforge-pi05",
        )
